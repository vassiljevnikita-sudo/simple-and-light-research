# Dynamic-QBD Process Parallelism

The shared opportunity-portfolio multicore backend is bounded at 24 processes.
The affinity planner fills one logical processor per physical core first and
only uses SMT siblings if a caller explicitly requests more capacity than the
physical-core plan can provide. Numeric native libraries remain at one thread
per worker, so process parallelism is not multiplied by hidden BLAS threads.

The ABC recalibration runner uses the same execution principle. Independent
fresh fits run in a Windows-spawned `ProcessPoolExecutor` with 24 configured
workers and a bounded queue of at most `2 * active_workers` submitted tasks.
The parent process owns telemetry aggregation and deterministic result ordering;
workers return the completed fit artifact plus process id, logical processor,
affinity status, compute time and idle time before the task.

This is execution-only infrastructure. It does not alter recipes, model
hyperparameters, dates, target definitions, holdout state or promotion
authority. The existing Dynamic-QBD memory Job Object remains suite-wide and
is inherited by spawned workers.
