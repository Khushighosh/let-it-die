import json
import os
from pathlib import Path
from temporalio import activity

LOG = Path("actions.jsonl")

def _log(row: dict) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

@activity.defn
async def archive_email(mid: str) -> str:
    attempt = activity.info().attempt
    if os.environ.get("FLAKY") and attempt == 1:
        raise RuntimeError("simulated flaky archive call")
    _log({"mid": mid, "action": "archived", "attempt": attempt})
    return "archived"

@activity.defn
async def escalate_email(mid: str) -> str:
    _log({"mid": mid, "action": "escalated", "attempt": activity.info().attempt})
    return "escalated"
