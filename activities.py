import json
from pathlib import Path
from temporalio import activity

LOG = Path("actions.jsonl")

def _log(row: dict) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

@activity.defn
async def archive_email(mid: str) -> str:
    _log({"mid": mid, "action": "archived"})
    return "archived"

@activity.defn
async def escalate_email(mid: str) -> str:
    _log({"mid": mid, "action": "escalated"})
    return "escalated"
