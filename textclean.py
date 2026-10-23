import re

# Cut an email at the first forwarded/quoted-message marker so the model summarizes
# what the sender actually wrote, not the chain underneath.
_CUT = re.compile(r"^[ \t>-]*(forwarded by|original message)", re.I | re.M)


def strip_forwarded(body: str) -> str:
    m = _CUT.search(body)
    top = body[:m.start()].strip() if m else body
    return top if len(top) >= 20 else body    # forward with no comment: keep everything
