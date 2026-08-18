"""Run E024 local physical work only after Phase 0 admission."""

from __future__ import annotations

from experiment_024_calibrate import main

if __name__ == "__main__":
    raise SystemExit(main())
