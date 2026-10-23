import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

LABEL = "replied"
TRAIN_N = 5000   # lower to 3000 if the TabPFN API rejects the training size

df = pd.read_csv("features.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
cols = [c for c in df.columns if c not in ("mid", "sender", "date", "replied", "followup")]
cut = int(0.8 * len(df))
tr, te = df.iloc[:cut], df.iloc[cut:]
Xtr, ytr, Xte, yte = tr[cols], tr[LABEL], te[cols], te[LABEL]
print("test n=%d | no-reply base rate=%.3f" % (len(te), 1 - yte.mean()))

def report(name, p):
    p = np.asarray(p)
    print("\n%s | ROC AUC %.3f" % (name, roc_auc_score(yte, p)))
    order = np.argsort(p, kind="stable")      # lowest reply-probability first
    y = yte.values
    for cov in (0.1, 0.2, 0.3, 0.4, 0.5):
        k = int(cov * len(p))
        missed = y[order[:k]].sum()
        print("  ignore lowest %2d%% (%4d emails): %.1f%% truly needed no reply | missed %.1f%% of all replied-to mail"
              % (cov * 100, k, 100 * (1 - missed / k), 100 * missed / y.sum()))

report("Heuristic: sender reply rate", te["sender_reply_rate"].fillna(ytr.mean()))

hgb = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
report("HistGradientBoosting", hgb.predict_proba(Xte)[:, 1])

from tabpfn_client import TabPFNClassifier
sub = tr.tail(TRAIN_N)
tab = TabPFNClassifier()
tab.fit(sub[cols], sub[LABEL])
report("TabPFN", tab.predict_proba(Xte)[:, 1])

ph = hgb.predict_proba(Xte)[:, 1]
pt = tab.predict_proba(Xte)[:, 1]
pd.DataFrame({"mid": te["mid"].values, "y": yte.values, "hgb": ph, "tabpfn": pt}).to_csv("preds.csv", index=False)
rng = np.random.default_rng(0)
y = yte.values
d = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    d.append(roc_auc_score(y[i], pt[i]) - roc_auc_score(y[i], ph[i]))
print("\nTabPFN minus HGB AUC: mean %.3f, 95%% CI [%.3f, %.3f]" % (np.mean(d), *np.percentile(d, [2.5, 97.5])))
