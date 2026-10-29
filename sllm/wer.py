def normalize(text: str) -> list[str]:
    return "".join(c if c.isalnum() or c in " '" else " " for c in text.upper()).split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    errors = words = 0
    for r, h in zip(refs, hyps):
        r, h = normalize(r), normalize(h)
        errors += edit_distance(r, h)
        words += len(r)
    return errors / max(words, 1)
