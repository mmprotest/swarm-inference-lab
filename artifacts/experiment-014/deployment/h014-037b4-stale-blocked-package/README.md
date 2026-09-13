# Experiment 015 deployment package — NOT READY

This directory is intentionally fail-closed. It is not the self-contained deployment package required to start Experiment 015.

`DEPLOYMENT-BLOCKED.json` records the failed Experiment 014 certification. `canary.sh` exits before provisioning or fleet activation. Do not remove the marker manually; regenerate the package only after all hard gates in `../experiment-014/acceptance-gates.json` pass.

See `../../docs/experiment-014-kimi-k3-precluster-certification.md` for measured evidence and `../../docs/experiment-015-physical-kimi-k3-plan.md` for the blocked handoff.
