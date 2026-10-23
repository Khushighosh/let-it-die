import email, email.utils, re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(r"\\?\D:\enron\maildir\kaminski-v")
ME = "vince.kaminski@enron.com"
LABEL = "replied"          # or "followup" to test the other proxy
W = timedelta(days=7)
LO = datetime(1998, 1, 1, tzinfo=timezone.utc)
HI = datetime(2002, 12, 31, tzinfo=timezone.utc)

def addrs(m, *names):
    vals = [str(v) for n in names for v in (m.get_all(n) or [])]
    return [a.lower() for _, a in email.utils.getaddresses(vals) if a]

def parse_date(s):
    try:
        d = email.utils.parsedate_to_datetime(s)
    except Exception:
        return None
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)

PREFIX = re.compile(r"^((re|fw|fwd):\s*)+", re.I)

# ---------- 1. parse ----------
rows, seen = [], set()
for p in ROOT.rglob("*"):
    if not p.is_file():
        continue
    try:
        m = email.message_from_bytes(p.read_bytes())
    except Exception:
        continue
    mid = str(m["Message-ID"] or "").strip()
    d = parse_date(str(m["Date"] or ""))
    if not mid or mid in seen or d is None or not (LO <= d <= HI):
        continue
    seen.add(mid)
    raw_subj = str(m["Subject"] or "").strip()
    body = m.get_payload() if not m.is_multipart() else ""
    rows.append({
        "mid": mid,
        "sender": email.utils.parseaddr(str(m["From"] or ""))[1].lower(),
        "to": addrs(m, "To"), "cc": addrs(m, "Cc"),
        "date": d,
        "subj": PREFIX.sub("", raw_subj).strip().lower(),
        "is_re": int(raw_subj.lower().startswith("re:")),
        "is_fwd": int(raw_subj.lower().startswith(("fw:", "fwd:"))),
        "irt": str(m["In-Reply-To"] or "").strip(),
        "body_len": len(body), "body_q": body.count("?"),
    })

sent = [r for r in rows if r["sender"] == ME]
recv = [r for r in rows if r["sender"] != ME]
print("parsed:", len(rows), "| sent:", len(sent), "| received:", len(recv))

# ---------- 2. indexes ----------
all_by_subj, sent_by_subj = defaultdict(list), defaultdict(list)
from_by, sent_to = defaultdict(list), defaultdict(list)
for r in rows:
    if r["subj"]:
        all_by_subj[r["subj"]].append(r["date"])
for r in sent:
    if r["subj"]:
        sent_by_subj[r["subj"]].append(r["date"])
    for a in set(r["to"] + r["cc"]):
        sent_to[a].append(r["date"])
for r in recv:
    if r["subj"]:
        from_by[(r["subj"], r["sender"])].append(r["date"])
for dct in (all_by_subj, sent_by_subj, from_by, sent_to):
    for v in dct.values():
        v.sort()
irt_set = {r["irt"] for r in sent if r["irt"]}

def in_win(lst, d):
    i = bisect_right(lst, d)
    return i < len(lst) and lst[i] <= d + W

# ---------- 3. labels ----------
max_date = max(r["date"] for r in sent)
recv = [r for r in recv if r["date"] <= max_date - W]   # drop emails whose window isn't fully observed
for r in recv:
    s, d = r["subj"], r["date"]
    r["replied"] = int(r["mid"] in irt_set or (bool(s) and in_win(sent_by_subj[s], d)))
    r["followup"] = int(bool(s) and in_win(from_by[(s, r["sender"])], d))
recv.sort(key=lambda r: r["date"])

# per-sender history, using only past emails whose window already closed (no leakage)
hist = {}
by_sender = defaultdict(list)
for r in recv:
    by_sender[r["sender"]].append(r)
for s, lst in by_sender.items():
    cum = [0]
    for x in lst:
        cum.append(cum[-1] + x[LABEL])
    hist[s] = ([x["date"] for x in lst], cum)

# ---------- 4. features ----------
feats = []
for r in recv:
    s, d = r["sender"], r["date"]
    dates, cum = hist[s]
    j = bisect_right(dates, d - W)
    feats.append({
        "mid": r["mid"], "sender": s, "date": d,
        "replied": r["replied"], "followup": r["followup"],
        "sender_prior_n": bisect_left(dates, d),
        "sender_reply_rate": cum[j] / j if j else float("nan"),
        "me_sent_to_sender": bisect_left(sent_to[s], d),
        "n_to": len(r["to"]), "n_cc": len(r["cc"]),
        "me_in_to": int(ME in r["to"]), "me_in_cc": int(ME in r["cc"]),
        "hour": d.hour, "weekday": d.weekday(),
        "subj_len": len(r["subj"]), "is_re": r["is_re"], "is_fwd": r["is_fwd"],
        "body_len": r["body_len"], "body_q": r["body_q"],
        "thread_prior": bisect_left(all_by_subj[r["subj"]], d) if r["subj"] else 0,
        "sender_internal": int(s.endswith("@enron.com")),
    })
df = pd.DataFrame(feats)
df.to_csv("features.csv", index=False)
print("rows used:", len(df), "| dates:", df.date.min().date(), "to", df.date.max().date())
print("replied rate: %.3f | followup rate: %.3f" % (df.replied.mean(), df.followup.mean()))

# ---------- 5. time-based split + baseline ----------
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

cols = [c for c in df.columns if c not in ("mid", "sender", "date", "replied", "followup")]
cut = int(0.8 * len(df))
tr, te = df.iloc[:cut], df.iloc[cut:]
clf = HistGradientBoostingClassifier(random_state=0).fit(tr[cols], tr[LABEL])
p = clf.predict_proba(te[cols])[:, 1]
print("train:", len(tr), "| test:", len(te), "| test positive rate: %.3f" % te[LABEL].mean())
print("ROC AUC: %.3f" % roc_auc_score(te[LABEL], p))
print(classification_report(te[LABEL], (p >= 0.5).astype(int),
                            target_names=["let it die", "needs action"], zero_division=0))
