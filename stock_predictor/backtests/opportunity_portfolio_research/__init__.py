"""Chronological V5 signal-to-portfolio research suite."""

# Dynamic-QBD runtime changes are installed in a strict order. Scheduler
# efficiency patches the original coordinator functions first; recovery
# hardening then wraps those patched functions with attempt fencing and lane
# repair; the post-hardening efficiency pass finally binds the hardened
# watchdog plus batched/Step-9 runner hooks. None of these layers changes
# scientific payloads, causal dependencies, RAM thresholds, numerical GPU
# kernels or holdout authority.
from .dynamic_qbd_scheduler_efficiency import (
    install_scheduler_efficiency,
    install_scheduler_efficiency_post_hardening,
)
from .dynamic_qbd_runtime_hardening import install_runtime_hardening
from .dynamic_qbd_runtime_hardening_extensions import (
    install_runtime_hardening_extensions,
)

install_scheduler_efficiency()
install_runtime_hardening()
install_runtime_hardening_extensions()
install_scheduler_efficiency_post_hardening()
