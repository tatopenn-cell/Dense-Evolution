# Dashboard Core — System Limits

Real, per-machine qubit limits — detects actual available RAM instead of
a number picked to fit whatever machine this was developed on, so
Composer refuses an allocation that would actually exhaust memory rather
than crashing partway through.

::: dashboard_core.system_limits
