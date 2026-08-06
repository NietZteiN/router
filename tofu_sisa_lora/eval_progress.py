"""Write eval progress to stdout (unbuffered) and a .progress.json file."""
import json
import sys
import time
from typing import Any


class ProgressLogger:
    def __init__(self, path: str | None, label: str):
        self.path = path
        self.label = label
        self.t0 = time.time()
        self.state: dict[str, Any] = {
            "label": label,
            "status": "starting",
            "step": None,
            "elapsed_s": 0,
            "metrics": {},
        }
        self._flush()

    def _flush(self):
        self.state["elapsed_s"] = round(time.time() - self.t0, 1)
        msg = f"[{self.label}] {self.state['status']}"
        if self.state.get("step"):
            msg += f" | {self.state['step']}"
        if self.state.get("metrics"):
            parts = [f"{k}={v}" for k, v in self.state["metrics"].items()]
            msg += " | " + ", ".join(parts)
        print(msg, flush=True)
        if self.path:
            with open(self.path, "w") as f:
                json.dump(self.state, f, indent=2)

    def step(self, status: str, detail: str | None = None):
        self.state["status"] = status
        self.state["step"] = detail
        self._flush()

    def metric(self, key: str, value: Any):
        self.state["metrics"][key] = value
        self._flush()

    def done(self, final: dict):
        self.state["status"] = "done"
        self.state["step"] = None
        self.state["results"] = final
        self._flush()
