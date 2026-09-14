> **Scope note — 2026-08-27:** This file remains the human-readable authority for the protected V4/N25, execution, V4.5 and V5 research IDs. It is **not** the current Dynamic-QBD project-status handoff. For active QBD work read [TOBECONTINUED.md](TOBECONTINUED.md) and [research/DYNAMIC_QBD_CURRENT_STATE.md](research/DYNAMIC_QBD_CURRENT_STATE.md).

# Current research model

Stand: 2026-07-31

Diese Datei ist die verbindliche menschlich lesbare Kurzfassung. Die maschinenlesbare Single Source of Truth ist `stock_predictor/research_model_registry.json`. Die Dokumentenrangfolge steht in `DOCUMENTATION_AUTHORITY.md`.

## Verbindlicher aktueller Stack

```text
CURRENT_V4_CANDIDATE=V4_N25_T212_RESEARCH_V1
CURRENT_EXECUTION_CANDIDATE=EXEC_N25_DUAL_VENUE_RESEARCH_V1
CURRENT_V45_CANDIDATE=V45_N25_HYBRID_STOP_RESEARCH_V1
CURRENT_V5_RESEARCH=V5_N25_SHADOW_V1
CURRENT_BENCHMARK_POLICY=MSCI_WORLD_IMPLEMENTABLE_PROXY_V1
LEGACY_V4_BASELINE=V4_FROZEN_10K_LEGACY
PAPER_TRADING_ONLY
```

## N25: aktueller V4-Kandidat

`V4_N25_T212_RESEARCH_V1` ist der einzige aktuelle V4-Research-Kandidat:

- 25 gleichgewichtete Aktien;
- 100 % aktiver Depotanteil, wenn der Marktfilter freigegeben ist;
- reguläre Prüfung ungefähr alle 15 Handelstage;
- Haltepuffer bis Rang 38;
- Rebalancingband 10 %;
- freiwillige Änderung erst ab 5 % des Depotwerts;
- Marktfilter: Benchmarkrendite der vergangenen 20 Handelstage größer als 0;
- keine feste Orderprovision;
- ausschließlich Research, Shadow und Demo.

Der Ausdruck `V4` allein ist nur als Familienname zulässig. Neue Resultate müssen mindestens `N25 (V4_N25_T212_RESEARCH_V1)` nennen.

## Ausführung

`EXEC_N25_DUAL_VENUE_RESEARCH_V1` vergleicht bei identischem N25-Signal und identischem Zielgewicht ex ante:

1. deutsche EUR-Linie derselben ISIN;
2. US-Primärlisting mit EUR-Autokonvertierung;
3. US-Primärlisting mit persistentem USD-Guthaben;
4. den vor Orderabgabe günstigsten zulässigen Pfad.

Die Implementierung liegt in `stock_predictor/n25_dual_venue.py`. Sie berücksichtigt Spread, Slippage, Gebühren, tatsächliche FX-Konvertierung, Quotegröße, Ausführungsverzögerung, Nichtausführung, Basisrisiko und Unsicherheit. Zukünftige Preise oder Fills sind für die historische Routenwahl verboten. Ohne ausreichende verbleibende Nettokante gilt `NO_TRADE_OR_DEFER`.

Die Policy ist implementiert, aber noch nicht mit vollständigen US-/deutschen Intraday-Daten und Forward-Fills validiert. Keine Route ist live freigegeben.

## Benchmark

`MSCI_WORLD_IMPLEMENTABLE_PROXY_V1` bildet eine investierbare MSCI-World-ETF-Umsetzung ab.

- Forschungsannahme: 0,20 % TER pro Jahr;
- TER wird nur separat abgezogen, wenn sie nicht bereits in ETF-Kurs oder NAV enthalten ist;
- Spread und Slippage bleiben getrennte Handelskosten;
- doppelte TER-Belastung ist verboten;
- Umsetzung: `stock_predictor/benchmark_policy.py`.

Historische N25-Ergebnisse sind noch nicht mit der aktuellen TER- und Dual-Venue-Policy reproduziert. Sie bleiben Kandidatendiagnosen und dürfen nicht umbenannt werden.

## V4.5 und V5

`V45_N25_HYBRID_STOP_RESEARCH_V1` ist ausschließlich eine Stop-/Crash-/Ersatzschicht über N25. `NONE` bleibt die aktive Kontrolle, bis ein eingefrorener Final-Holdout und Forward-Test bestanden sind.

`V5_N25_SHADOW_V1` ist eine ML-Research-Erweiterung von N25. Seine Portfoliokonfiguration verwendet dieselbe N25-Hülle: 25 Positionen, 100 % maximal aktiv, 400 EUR Zielnotional je Aktie bei 10.000 EUR Modellportfolio. V5 ist nicht promotionsfähig und nicht live freigegeben.

## Historische Legacy-V4

`V4_FROZEN_10K_LEGACY` bleibt unverändert als historische Kontrollstrategie:

- drei Aktien;
- maximal 50 % aktiv;
- mindestens 500 EUR freiwillige Order;
- 25-%-Rebalancingband;
- Rest im MSCI-World-Proxy.

Diese Strategie ist niemals der aktuelle V4-Kandidat. Der German-only-T212-Replay bleibt ausschließlich ein Legacy-Ausführungsbenchmark.

## Daten- und Evidenzstatus

- vorhandene Alpaca-Minutenbatches im Integrationsstand: 0000–0003 und 0006–0008;
- 0004 und 0005 sind nicht integriert;
- vollständige Point-in-Time-, deutsche Intraday- und Forward-Fill-Abdeckung fehlt;
- historische N25-Ergebnisse sind nicht TER- oder Dual-Venue-neuberechnet;
- Promotion bleibt blockiert;
- lokale Unit- und Syntaxprüfungen sind kein Ersatz für GitHub-Actions- oder Marktdatenvalidierung;
- keine Liveorders, kein VPS- oder Produktionsrollout.

## Benennungsregeln

Jedes neue Artefakt nennt `model_id`, `execution_policy_id` und `benchmark_policy_id`. V4.5 nennt zusätzlich `base_model_id`; V5 nennt N25 ausdrücklich als Basis. Legacy-Ergebnisse dürfen nicht unter aktuellen IDs neu veröffentlicht werden.
