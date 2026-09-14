"""Bounded-memory access to the large canonical prediction parquet."""
from __future__ import annotations

from pathlib import Path
import json
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


class DiskBackedPredictionStore:
    """Read only requested model IDs row-group by row-group from SSD."""

    columns = ("decision_date", "ticker", "score", "model_artifact_id")

    def __init__(self, path: str | Path, *, row_group_cache_size: int = 1):
        self.path = Path(path)
        parquet = pq.ParquetFile(self.path)
        self._num_row_groups = parquet.num_row_groups
        self._schema_names = tuple(parquet.schema_arrow.names)
        self._model_row_groups: dict[str, tuple[int, ...]] = {}
        self._model_row_group_index_complete = True
        model_field = parquet.schema_arrow.get_field_index("model_artifact_id")
        for row_group in range(parquet.num_row_groups):
            if parquet.metadata.row_group(row_group).num_rows == 0:
                continue
            statistics = parquet.metadata.row_group(row_group).column(model_field).statistics
            if statistics is None or statistics.min is None or statistics.max is None:
                # Some Parquet writers omit string statistics.  Recover the
                # exact single-ID invariant from only this row group's ID
                # column; never infer a partial index from incomplete stats.
                values = parquet.read_row_group(row_group, columns=["model_artifact_id"])["model_artifact_id"]
                unique_values = values.unique().to_pylist()
                if len(unique_values) != 1 or unique_values[0] is None:
                    self._model_row_group_index_complete = False
                    continue
                model_id = str(unique_values[0])
                self._model_row_groups[model_id] = self._model_row_groups.get(model_id, ()) + (row_group,)
                continue
            # The canonical store is written one authoritative model artifact per
            # row group.  Use the exact min=max statistics as a safe index; if a
            # future artifact violates that invariant, iter_model_ids falls back
            # to the complete scan below rather than risking dropped evidence.
            if str(statistics.min) != str(statistics.max):
                self._model_row_group_index_complete = False
                continue
            model_id = str(statistics.min)
            self._model_row_groups[model_id] = self._model_row_groups.get(model_id, ()) + (row_group,)
        parquet.close()
        self._cache: dict[tuple[str, ...], pd.DataFrame] = {}
        self._cache_limit = max(0, int(row_group_cache_size))

    def max_date(self):
        parquet = pq.ParquetFile(self.path)
        field = parquet.schema_arrow.get_field_index("decision_date")
        values = []
        try:
            for index in range(parquet.num_row_groups):
                statistics = parquet.metadata.row_group(index).column(field).statistics
                if statistics is not None and statistics.max is not None:
                    values.append(statistics.max)
        finally:
            parquet.close()
        return pd.to_datetime(max(values)) if values else pd.NaT

    def read_model_ids(self, model_ids) -> pd.DataFrame:
        if not model_ids:
            return pd.DataFrame(columns=list(self.columns))
        chunks = list(self.iter_model_ids(model_ids))
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=list(self.columns))

    def iter_model_ids(self, model_ids):
        """Yield matching prediction rows one parquet row group at a time."""
        wanted = tuple(sorted({str(x) for x in model_ids}))
        if not wanted:
            return
        wanted_set = set(wanted)
        parquet = pq.ParquetFile(self.path)
        try:
            if self._model_row_group_index_complete:
                row_groups = sorted({index for model_id in wanted_set for index in self._model_row_groups.get(model_id, ())})
            else:
                row_groups = range(parquet.num_row_groups)
            for index in row_groups:
                frame = parquet.read_row_group(index, columns=list(self.columns)).to_pandas()
                mask = frame["model_artifact_id"].astype(str).isin(wanted_set)
                if mask.any():
                    yield frame.loc[mask, list(self.columns)].copy()
        finally:
            parquet.close()

    def __contains__(self, column: str) -> bool:
        return column in self.columns

    def __iter__(self):
        return iter(self.columns)


class MaturedPredictionStore:
    """Partitioned, immutable-once-published matured prediction evidence.

    The full matured panel is intentionally not represented by a pandas
    DataFrame.  Each family is an independent parquet object and the manifest
    is published only after the corresponding object has been atomically
    completed and validated.  Consumers can therefore retain at most one
    family in memory while preserving the exact candidate x fold evidence.
    """

    columns = (
        "decision_date", "terminal_date", "ticker", "family_id",
        "generation_id", "model_artifact_id", "score", "realized_excess",
    )
    schema_version = "DYNAMIC_QBD_MATURED_PARTITION_CACHE_V1"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"MATURED_CACHE_MANIFEST_MISSING:{self.root}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != self.schema_version:
            raise ValueError("MATURED_CACHE_SCHEMA_MISMATCH")
        self.cache_fingerprint = str(self.manifest.get("cache_fingerprint", ""))
        self._family_meta = dict(self.manifest.get("families", {}))
        self._validate_manifest()

    @classmethod
    def empty(cls, root: str | Path, *, cache_fingerprint: str, schema: pa.Schema):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        return cls._from_manifest(root, {
            "schema_version": cls.schema_version,
            "cache_fingerprint": str(cache_fingerprint),
            "columns": list(schema.names),
            "families": {},
            "status": "BUILDING",
        })

    @classmethod
    def _from_manifest(cls, root: Path, manifest: dict):
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f"manifest.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, root / "manifest.json")
        return cls(root)

    def _validate_manifest(self) -> None:
        expected = set(self.columns)
        declared = set(self.manifest.get("columns", []))
        if not expected.issubset(declared):
            raise ValueError("MATURED_CACHE_COLUMNS_MISSING")
        for family_id, meta in self._family_meta.items():
            path = (self.root / str(meta["path"])).resolve()
            try:
                path.relative_to(self.root.resolve())
            except ValueError as exc:
                raise ValueError("MATURED_CACHE_PATH_ESCAPE") from exc
            if not path.is_file():
                raise FileNotFoundError(f"MATURED_CACHE_PARTITION_MISSING:{family_id}")
            parquet = pq.ParquetFile(path)
            if not expected.issubset(set(parquet.schema_arrow.names)):
                raise ValueError(f"MATURED_CACHE_PARTITION_SCHEMA_MISMATCH:{family_id}")
            if int(parquet.metadata.num_rows) != int(meta["rows"]):
                raise ValueError(f"MATURED_CACHE_PARTITION_ROWCOUNT_MISMATCH:{family_id}")

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._family_meta))

    @property
    def row_count(self) -> int:
        return sum(int(meta["rows"]) for meta in self._family_meta.values())

    def max_terminal_date(self):
        values = []
        for meta in self._family_meta.values():
            parquet = pq.ParquetFile(self.root / str(meta["path"]))
            index = parquet.schema_arrow.get_field_index("terminal_date")
            for row_group in range(parquet.num_row_groups):
                statistics = parquet.metadata.row_group(row_group).column(index).statistics
                if statistics is not None and statistics.max is not None:
                    values.append(statistics.max)
        return pd.to_datetime(max(values)) if values else pd.NaT

    def read_family(self, family_id: str, *, columns=None) -> pd.DataFrame:
        meta = self._family_meta.get(str(family_id))
        if meta is None:
            return pd.DataFrame(columns=list(columns or self.columns))
        requested = list(columns or self.columns)
        missing = set(requested) - set(self.columns)
        if missing:
            raise ValueError(f"MATURED_CACHE_COLUMNS_UNKNOWN:{sorted(missing)}")
        frame = pd.read_parquet(self.root / str(meta["path"]), columns=requested)
        if not frame.empty and "family_id" in frame:
            if not frame["family_id"].astype(str).eq(str(family_id)).all():
                raise ValueError(f"MATURED_CACHE_FAMILY_PARTITION_MISMATCH:{family_id}")
        return frame

    def __contains__(self, column: str) -> bool:
        return column in set(self.manifest.get("columns", []))

    def publish_family(self, family_id: str, frame: pd.DataFrame) -> None:
        """Atomically publish one complete family partition and checkpoint it."""
        self.publish_family_chunks(family_id, (frame,))

    def publish_family_chunks(self, family_id: str, chunks) -> int:
        """Atomically publish a family from bounded DataFrame chunks."""
        family_id = str(family_id)
        filename = f"family-{family_id}.parquet"
        final = self.root / filename
        temporary = self.root / f".{filename}.{os.getpid()}.tmp"
        writer = None
        rows = 0
        try:
            for frame in chunks:
                missing = set(self.columns) - set(frame.columns)
                if missing:
                    raise ValueError(f"MATURED_CACHE_WRITE_COLUMNS_MISSING:{sorted(missing)}")
                table = pa.Table.from_pandas(frame.loc[:, list(self.columns)], preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                rows += int(table.num_rows)
            if writer is None:
                return 0
            writer.close()
            writer = None
            check = pq.ParquetFile(temporary)
            if set(self.columns) - set(check.schema_arrow.names):
                del check
                raise ValueError(f"MATURED_CACHE_WRITE_SCHEMA_MISMATCH:{family_id}")
            validated_rows = int(check.metadata.num_rows)
            del check
            if validated_rows != rows:
                raise ValueError(f"MATURED_CACHE_WRITE_ROWCOUNT_MISMATCH:{family_id}")
            os.replace(temporary, final)
            families = dict(self._family_meta)
            families[family_id] = {"path": filename, "rows": rows}
            self.manifest["families"] = families
            self.manifest["completed_family_count"] = len(families)
            self.manifest["status"] = "BUILDING"
            self._write_manifest()
            self._family_meta = families
            return rows
        finally:
            if writer is not None:
                writer.close()
            if temporary.exists():
                temporary.unlink()

    def publish_family_group_from_base(self, family_specs, base_chunks) -> dict[str, int]:
        """Publish sibling N-partitions while reading a shared base once."""
        family_specs = {str(family_id): schedule for family_id, schedule in family_specs.items()}
        family_ids = tuple(family_specs)
        if not family_ids:
            return {}
        temporary_paths = {
            family_id: self.root / f".family-{family_id}.parquet.{os.getpid()}.tmp"
            for family_id in family_ids
        }
        final_paths = {
            family_id: self.root / f"family-{family_id}.parquet"
            for family_id in family_ids
        }
        writers = {}
        rows = {family_id: 0 for family_id in family_ids}
        try:
            for base_frame in base_chunks:
                missing = set(self.columns) - set(base_frame.columns) - {"family_id", "generation_id"}
                if missing:
                    raise ValueError(f"MATURED_CACHE_WRITE_COLUMNS_MISSING:{sorted(missing)}")
                for family_id in family_ids:
                    schedule = family_specs[family_id]
                    decisions = pd.DataFrame({"decision_date": sorted(base_frame["decision_date"].unique())})
                    authority = pd.merge_asof(
                        decisions,
                        schedule[["activation_date", "generation_id", "model_artifact_id"]],
                        left_on="decision_date", right_on="activation_date", direction="backward",
                    ).dropna(subset=["generation_id"])
                    family_frame = base_frame.merge(
                        authority[["decision_date", "generation_id", "model_artifact_id"]],
                        on=["decision_date", "model_artifact_id"], how="inner",
                    )
                    if family_frame.empty:
                        del decisions, authority, family_frame
                        continue
                    family_frame["family_id"] = family_id
                    family_frame = family_frame.loc[:, list(self.columns)]
                    table = pa.Table.from_pandas(family_frame, preserve_index=False)
                    writer = writers.get(family_id)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary_paths[family_id], table.schema, compression="zstd")
                        writers[family_id] = writer
                    writer.write_table(table)
                    rows[family_id] += int(table.num_rows)
                    del decisions, authority, family_frame, table
            written_family_ids = set(writers)
            for writer in writers.values():
                writer.close()
            writers.clear()
            if any(family_id not in written_family_ids or rows[family_id] == 0 for family_id in family_ids):
                missing = [family_id for family_id in family_ids
                           if family_id not in written_family_ids or rows[family_id] == 0]
                raise ValueError(f"MATURED_CACHE_GROUP_EMPTY_FAMILIES:{missing[:10]}")
            families = dict(self._family_meta)
            for family_id in family_ids:
                check = pq.ParquetFile(temporary_paths[family_id])
                if int(check.metadata.num_rows) != rows[family_id] or set(self.columns) - set(check.schema_arrow.names):
                    check.close()
                    raise ValueError(f"MATURED_CACHE_GROUP_VALIDATION_FAILED:{family_id}")
                check.close()
            for family_id in family_ids:
                os.replace(temporary_paths[family_id], final_paths[family_id])
                families[family_id] = {"path": final_paths[family_id].name, "rows": rows[family_id]}
            self.manifest["families"] = families
            self.manifest["completed_family_count"] = len(families)
            self.manifest["status"] = "BUILDING"
            self._write_manifest()
            self._family_meta = families
            return rows
        finally:
            for writer in writers.values():
                writer.close()
            for path in temporary_paths.values():
                if path.exists():
                    path.unlink()

    def finalize(self) -> None:
        self.manifest["status"] = "COMPLETE"
        self.manifest["row_count"] = self.row_count
        self._write_manifest()

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_name(self.manifest_path.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def copy_to_parquet(self, destination: str | Path) -> Path:
        """Materialize a flat compatibility file by streaming partitions."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and self._flat_file_matches_manifest(destination):
            return destination
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        writer = None
        try:
            for family_id in self.families:
                table = pa.Table.from_pandas(self.read_family(family_id), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
            if writer is None:
                schema = pa.schema([(name, pa.string()) for name in self.columns])
                writer = pq.ParquetWriter(temporary, schema, compression="zstd")
            writer.close()
            writer = None
            check = pq.ParquetFile(temporary)
            flat_rows = int(check.metadata.num_rows)
            del check
            if flat_rows != self.row_count:
                raise ValueError("MATURED_CACHE_FLAT_ROWCOUNT_MISMATCH")
            os.replace(temporary, destination)
        finally:
            if writer is not None:
                writer.close()
            if temporary.exists():
                temporary.unlink()
        return destination

    def _flat_file_matches_manifest(self, path: Path) -> bool:
        """Validate an existing flat export without reading its value columns."""
        try:
            parquet = pq.ParquetFile(path)
            if set(self.columns) - set(parquet.schema_arrow.names):
                return False
            family_index = parquet.schema_arrow.get_field_index("family_id")
            observed: dict[str, int] = {}
            for row_group in range(parquet.num_row_groups):
                metadata = parquet.metadata.row_group(row_group)
                statistics = metadata.column(family_index).statistics
                if statistics is None or statistics.min is None or statistics.max is None:
                    return False
                if str(statistics.min) != str(statistics.max):
                    return False
                family_id = str(statistics.min)
                observed[family_id] = observed.get(family_id, 0) + int(metadata.num_rows)
            expected = {str(family_id): int(meta["rows"]) for family_id, meta in self._family_meta.items()}
            return int(parquet.metadata.num_rows) == self.row_count and observed == expected
        except (OSError, ValueError, TypeError):
            return False
