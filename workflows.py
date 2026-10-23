import asyncio
from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities import archive_email, escalate_email

@workflow.defn
class ParkedEmail:
    def __init__(self) -> None:
        self._followup = False

    @workflow.signal
    def followup_arrived(self) -> None:
        self._followup = True

    @workflow.run
    async def run(self, mid: str, wait_days: int) -> str:
        try:
            await workflow.wait_condition(
                lambda: self._followup, timeout=timedelta(days=wait_days))
        except asyncio.TimeoutError:
            return await workflow.execute_activity(
                archive_email, mid, start_to_close_timeout=timedelta(seconds=10))
        return await workflow.execute_activity(
            escalate_email, mid, start_to_close_timeout=timedelta(seconds=10))
