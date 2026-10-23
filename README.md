# Let It Die: an agent whose main skill is deciding not to act

Built for the Hacktoberfest Open-Source AI Challenge, Week 3 ("Slow and Steady").

Most "lazy" agents do your chores faster. This one asks whether the chore needs doing at all. It scores each incoming email on how likely ignoring it is to cost you something, then:

- **Need you** (top 20% by score): a local Gemma model writes a two-line summary and a suggested action for the morning digest.
- **Low priority** (middle 60%): left alone, counted in the digest.
- **Let it die** (bottom 20%): parked on a durable Temporal timer. If the sender follows up within 7 days, the email is escalated. If nothing arrives, it is archived.

```
emails ──> TabPFN score ──> triage ──┬─ need you ──> Gemma summary ──> morning digest
                                     ├─ low priority ──> counted only
                                     └─ let it die ──> Temporal timer (7 days)
                                                         ├─ follow-up signal ──> escalate
                                                         └─ timeout ──────────> archive
```

## Results

Data: one public mailbox from the CMU Enron corpus (May 7, 2015 release, `kaminski-v`). I have no two-sided mailbox of my own, so I used Enron, which includes sent folders and lets reply labels be derived.

Label: an email counts as "needed action" if the owner replied in the same thread within 7 days. The data is cut to Nov 1999 through May 2001, because sent mail stops being archived after that and every later email would look unanswered. Emails in the last 7 days of that window are dropped, since their outcome isn't observable yet.

Split: time-based. Train on the earliest 80% (7,921 emails), test on the latest 20% (1,981 emails, 70.5% of which got no reply).

| Model | ROC AUC | Ignore lowest 10%: truly no reply / replied-to mail missed | Ignore lowest 20% |
|---|---|---|---|
| Sender reply-rate rule | 0.579 | 68.2% / 10.8% | 73.7% / 17.8% |
| Gradient boosting | 0.662 | 92.4% / 2.6% | 88.4% / 7.9% |
| TabPFN | 0.687 | 92.9% / 2.4% | 84.6% / 10.4% |

TabPFN minus gradient boosting on AUC: +0.024 (bootstrap 95% CI 0.003 to 0.043). Both models far outperform the sender-history rule. The gap between the two models is small and mixed across buckets, so treat them as comparable.

Triage on the test slice: 397 "need you", 1,186 low priority, 398 "let it die". In the "need you" bucket 46% were actually replied to, against 30% overall. In the "let it die" bucket, about 15% were (from the 20% row above).

Parking demo: 8 randomly chosen emails from the "let it die" bucket were parked for 7 simulated days. 7 were archived, and all 7 truly needed no reply. 1 was escalated; it came from a mailing list that re-sent the same subject line, which shows the weakness of "sender wrote again" as a follow-up signal.

Local Gemma (`gemma3:4b` via Ollama): about 8 to 9 seconds per email on my laptop, with 90 to 649 prompt tokens and 30 to 59 output tokens per summary across two runs.

## Why open

- **Gemma runs locally** through Ollama, so email text never leaves the machine for summarization.
- **Temporal** (open source) makes the "wait and re-check" step durable: the timer survives restarts, and there is no cron job or queue to maintain.
- **TabPFN** needs no training step, which suits a small personal mailbox. Honest caveat: I used the hosted API (`tabpfn-client`), so in this pipeline the 5,000 feature rows used to fit the model are sent to Prior Labs' servers. I only did that with the public Enron data.
- No per-token bill: the only cost is electricity.

## Limitations

- "Replied within 7 days" is a proxy for "needed action", and "sender wrote again" is a noisy proxy for "ignoring it cost something".
- One mailbox, one time-based split. The bootstrap interval treats emails as independent, though threads and senders cluster, so it is probably a little too tight.
- The parking demo uses 8 emails and a simulated clock (Temporal's time-skipping test server), and "archive" writes to a log file instead of moving real mail.
- Gemma summaries are not deterministic and can go wrong on forwarded chains: for one email it summarized the oldest forwarded message instead of the sender's one-line reply. Cutting each email at the first forwarded or quoted-message marker before summarizing (`textclean.py`) fixed that case: the prompt for that email dropped from 649 to 90 tokens, the call from 10.9s to 3.2s, and the summary now describes the sender's actual reply. That is one email and one run per version, not a benchmark, and the suggested-action line for it is still a mild inference.
- Attachments are not in the email text, so suggested actions for attachment-only emails are generic.

## Debugging notes

1. Enron filenames end in a dot (`1.`, `10.`). Windows silently strips the dot, so Python reported every file as missing until paths used the `\\?\` prefix.
2. The first test split had zero replied-to emails. A month-by-month table showed reply rates falling to exactly 0 from June 2001, because the archive stops containing sent mail. The fix was to cut the timeline where sent mail ends.
3. A forwarded-chain failure in the digest (see Limitations) was found by reading the cached email text next to the summary, and fixed with `textclean.py`.

## Run it (Windows, Python 3.12)

```
pip install pandas scikit-learn tabpfn-client temporalio ollama sentry-sdk
ollama pull gemma3:4b
```

1. Download the CMU Enron corpus and extract one mailbox, for example `tar -xzf enron_mail_20150507.tar.gz -C D:\enron maildir/kaminski-v`.
2. `python build.py` builds labels and features and a gradient-boosting baseline.
3. `python compare.py` compares the heuristic, gradient boosting and TabPFN (needs a free Prior Labs account) and writes `preds.csv`.
4. `python agent.py` runs the full agent: triage, Gemma digest, Temporal parking. Set `SENTRY_DSN` first to send a trace to Sentry.

Files: `build.py`, `compare.py`, `activities.py`, `workflows.py`, `agent.py`, `textclean.py`. The Enron data, `features.csv`, `preds.csv` and any tokens or DSNs are not committed.

## Partner categories

- **TabPFN** (Prior Labs): the ignore-vs-act predictor.
- **Gemma**: local summaries.
- **Temporal**: durable wait-and-re-check workflows.
- **Sentry**: TODO, add screenshots of the agent trace and what it showed, or remove this line if tracing isn't shipped.

## Data credit

Enron Email Dataset, CMU / William W. Cohen, May 7, 2015 release.
