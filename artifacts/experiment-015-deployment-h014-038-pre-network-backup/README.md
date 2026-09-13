# Experiment 015: physical Kimi K3 cluster handoff

This is a no-checkout, hash-locked release for the selected 93-worker whole-layer topology.
Experiment 014 local pre-canary certification is complete. This package is ready to run the
preregistered single RTX 3090 canary; it is not proof that the physical canary passed.

1. Verify `package-lock.json` with the installed package validator.
2. Run `scripts/install-worker.sh` and `scripts/build-runtime.sh` on the canary RTX 3090.
3. Prepare workers 000, 089, 091 and 092 with `scripts/prepare-worker.sh`.
4. Run `scripts/run-3090-canary.sh`. Only this may emit the Linux physical certificate.
5. Publish the canary-built ELF and certificate immutably; record the certificate SHA out of band.
6. Rent the remaining nodes only if the canary passes and price is at most USD 0.05/GPU-hour.
7. On every fleet node run `install-qualified-runtime.sh`, `prepare-worker.sh`, then
   `qualify-worker.sh`; never rebuild the fleet ELF independently.
8. Exchange public identity fingerprints, start all admitted workers, then run
   `bind-and-deploy.sh`.

The selected initial fleet does not use sub-layer microworkers. Their measured logical result is
functional but uneconomic; the optional two-GPU script is deliberately non-activating.
