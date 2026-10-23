import asyncio, email, hashlib, json, os, time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import ollama
import pandas as pd
import sentry_sdk
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities import archive_email, escalate_email
from workflows import ParkedEmail; from textclean import strip_forwarded

ROOT = Path(r"\\?\D:\enron\maildir\kaminski-v")
MODEL, AGENT, PIPELINE = "gemma3:4b", "Let It Die", "let-it-die"
N_SHOW, N_PARK, WAIT_DAYS, QUEUE = 5, 8, 7, "let-it-die"
CACHE = Path("digest_cache.json")

# Prompts and outputs are sent to Sentry because the data is the public Enron corpus.
# Set send_default_pii=False if you ever run this on real mail.
DSN = os.environ.get("SENTRY_DSN", ""); DSN = DSN if "@" in DSN else None; sentry_sdk.init(dsn=DSN, traces_sample_rate=1.0, send_default_pii=True)

PROMPT = ("You are an email triage assistant. Read the email below and reply with exactly two lines:\n"
          "Summary: <one or two sentences>\n"
          "Action: <what the recipient most likely needs to do>\n"
          "Use only what the email says. Do not invent details.\n\n"
          "Subject: {subject}\n\n{body}")


def span(op, name, attrs):
    s = sentry_sdk.start_span(op=op, name=name)
    for k, v in attrs.items():
        s.set_data(k, v)
    return s


def fetch_bodies(mids):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    missing = set(mids) - set(cache)
    if missing:
        print("Scanning mailbox for %d emails..." % len(missing))
        for p in ROOT.rglob("*"):
            if not p.is_file():
                continue
            try:
                m = email.message_from_bytes(p.read_bytes())
            except Exception:
                continue
            mid = str(m["Message-ID"] or "").strip()
            if mid in missing:
                body = m.get_payload() if not m.is_multipart() else ""
                cache[mid] = {"subject": str(m["Subject"] or "").strip(), "body": str(body)[:1500]}
                missing.discard(mid)
                if not missing:
                    break
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def summarize(c):
    content = PROMPT.format(subject=c["subject"], body=strip_forwarded(c["body"]))
    attrs = {
        "gen_ai.operation.name": "chat", "gen_ai.provider.name": "ollama",
        "gen_ai.request.model": MODEL, "gen_ai.agent.name": AGENT,
        "gen_ai.pipeline.name": PIPELINE, "gen_ai.request.temperature": 0.2,
        "gen_ai.input.messages": json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": content}]}]),
    }
    with span("gen_ai.chat", "chat " + MODEL, attrs) as s:
        try:
            t = time.time()
            resp = ollama.chat(model=MODEL, options={"temperature": 0.2},
                               messages=[{"role": "user", "content": content}])
            dt = time.time() - t
        except Exception as exc:
            s.set_status("internal_error")
            s.set_data("error.type", type(exc).__name__)
            raise
        text = resp["message"]["content"].strip()
        tin = getattr(resp, "prompt_eval_count", None) or 0
        tout = getattr(resp, "eval_count", None) or 0
        s.set_data("gen_ai.response.model", getattr(resp, "model", None) or MODEL)
        s.set_data("gen_ai.output.messages", json.dumps(
            [{"role": "assistant", "parts": [{"type": "text", "content": text}]}]))
        s.set_data("gen_ai.usage.input_tokens", tin)
        s.set_data("gen_ai.usage.output_tokens", tout)
    return text, dt, tin, tout


async def park(sample):
    Path("actions.jsonl").unlink(missing_ok=True)
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(env.client, task_queue=QUEUE, workflows=[ParkedEmail],
                          activities=[archive_email, escalate_email]):
            handles = {}
            for r in sample.itertuples():
                handles[r.mid] = await env.client.start_workflow(
                    ParkedEmail.run, args=[r.mid, WAIT_DAYS], task_queue=QUEUE,
                    id="park-" + hashlib.md5(r.mid.encode()).hexdigest()[:12])
            await env.sleep(timedelta(days=2))              # simulated day 2
            for r in sample.itertuples():
                if r.followup == 1:                         # sender really wrote again in the data
                    await handles[r.mid].signal(ParkedEmail.followup_arrived)
            return [await handles[m].result() for m in sample["mid"]]


async def main():
    preds = pd.read_csv("preds.csv")
    feats = pd.read_csv("features.csv")[["mid", "sender", "followup"]]
    df = preds.merge(feats, on="mid")

    with sentry_sdk.start_transaction(name="let-it-die morning run", op="agent.run"):
        agent_attrs = {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": AGENT,
                       "gen_ai.pipeline.name": PIPELINE}
        with span("gen_ai.invoke_agent", "invoke_agent " + AGENT, agent_attrs) as agent:
            tool = {"gen_ai.operation.name": "execute_tool", "gen_ai.agent.name": AGENT,
                    "gen_ai.tool.name": "triage_buckets",
                    "gen_ai.tool.description": "Split emails into need-you / low-priority / let-it-die by TabPFN score",
                    "gen_ai.tool.call.arguments": json.dumps({"emails": len(df)})}
            with span("gen_ai.execute_tool", "execute_tool triage_buckets", tool) as s:
                hi, lo = df.tabpfn.quantile(0.8), df.tabpfn.quantile(0.2)
                act, parked = df[df.tabpfn >= hi], df[df.tabpfn <= lo]
                middle = len(df) - len(act) - len(parked)
                s.set_data("gen_ai.tool.call.result", json.dumps(
                    {"need_you": len(act), "low_priority": middle, "let_it_die": len(parked)}))

            show = act.sample(N_SHOW, random_state=0).sort_values("tabpfn", ascending=False)
            sample = parked.sample(N_PARK, random_state=0).copy()
            cache = fetch_bodies(show.mid)

            lines = ["MORNING DIGEST (test slice of %d emails)" % len(df),
                     "%d need you | %d low-priority | %d parked on a timer (let it die)"
                     % (len(act), middle, len(parked)),
                     "In the 'need you' bucket, %.0f%% were actually replied to (overall: %.0f%%)\n"
                     % (100 * act.y.mean(), 100 * df.y.mean())]
            lat = []
            for r in show.itertuples():
                c = cache.get(r.mid)
                if not c:
                    continue
                text, dt, tin, tout = summarize(c)
                lat.append(dt)
                lines.append("- %s | %s | score %.2f | he %s" % (
                    r.sender, c["subject"] or "(no subject)", r.tabpfn,
                    "replied" if r.y == 1 else "did not reply"))
                lines.append("  " + text.replace("\n", "\n  "))
                lines.append("  [%.1fs, %s prompt tokens, %s output tokens]\n" % (dt, tin, tout))
            if lat:
                lines.append("Local Gemma latency: avg %.1fs per email\n" % (sum(lat) / len(lat)))

            park_tool = {"gen_ai.operation.name": "execute_tool", "gen_ai.agent.name": AGENT,
                         "gen_ai.tool.name": "temporal_park",
                         "gen_ai.tool.description": "Park an email on a durable Temporal timer; archive on timeout, escalate on follow-up",
                         "gen_ai.tool.call.arguments": json.dumps({"emails": len(sample), "wait_days": WAIT_DAYS})}
            with span("gen_ai.execute_tool", "execute_tool temporal_park", park_tool) as s:
                outcomes = await park(sample)
                s.set_data("gen_ai.tool.call.result", json.dumps(dict(Counter(outcomes))))

            sample["outcome"] = outcomes
            arch = sample[sample.outcome == "archived"]
            lines.append("Parked %d emails for up to %d days: archived %d, escalated %d; "
                         "of the archived, %d truly needed no reply"
                         % (len(sample), WAIT_DAYS, len(arch), (sample.outcome == "escalated").sum(),
                            int((arch.y == 0).sum())))
            out = "\n".join(lines)
            agent.set_data("gen_ai.output.messages", json.dumps(
                [{"role": "assistant", "parts": [{"type": "text", "content": out}]}]))

    print(out)
    Path("digest.txt").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
    sentry_sdk.flush()
    print("\nSentry:", "trace sent" if DSN else "SENTRY_DSN not set, tracing skipped")
