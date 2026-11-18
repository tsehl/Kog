"""Progress of the translator, beyond the number of cards translated at 100 %.

The card counter only moves when ALL the effects of a card pass; an added mechanic advances hundreds of cards by one
notch without finishing any. Here, each effect registered by `initial_effect` is translated separately (the
translator is not modified), which gives:
- the share of translated effects, over the whole catalog (the mechanical progress);
- the distance of each card: number of its effects still rejected, and number of distinct refusal reasons;
- the refusal reasons, ranked by number of cards touched and by number of cards they block alone;
- with `--ref`, what moved since a previous snapshot: finished cards, cards brought closer, effects gained.

Limits: a rejected effect gives only its FIRST reason (the interpreter stops there): "blocked by a single reason" is
an upper bound, another reason may hide behind it. A card whose `initial_effect` is rejected has no effects counted
(distance unknown). A card whose effects all pass may remain rejected by a card-level rule (too many effects,
procedure...).

Usage: uv run --with luaparser --with jax python translate/progress.py [--json translate/progress.json.gz]
            [--ref translate/progress.json.gz] [--md translate/PROGRESS.md] [--codes a,b,...] [--top 40] [--jobs 4]"""
import argparse
import collections
import gzip
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _open(path, mode):
    """JSON snapshot, compressed if the name ends in .gz."""
    return gzip.open(path, mode + "t", encoding="utf-8") if path.endswith(".gz") else open(path, mode)


def family(reason):
    """Reason without its details (names, numbers): "unhandled call: Duel.X" -> "unhandled call"."""
    r = re.split(r" ?: ", reason, maxsplit=1)[0]
    return re.sub(r"\d+", "N", r)[:70]


def kind_of(code, tr):
    try:
        t = tr.card_type(code)
    except tr.Unsupported:
        return "?"
    core = tr.core
    if t & (core.TYPE_FUSION | core.TYPE_SYNCHRO | core.TYPE_XYZ | core.TYPE_LINK):
        return "extra"
    return "monster" if t & core.TYPE_MONSTER else "spell" if t & core.TYPE_SPELL else "trap" if t & core.TYPE_TRAP else "?"


def one(code):
    """-> (code, type, ok, effects, translated effects, [reasons of the rejected effects], card-level reason)."""
    import translate as tr
    K = tr.K
    try:
        tr.translate(code)
        ok, why = True, None
    except tr.Unsupported as e:
        ok, why = False, str(e)
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"BUG {type(e).__name__}"
    kind = kind_of(code, tr)
    t = tr.Translator(code)
    try:
        regs = t.effects()
    except tr.Unsupported as e:
        return code, kind, ok, -1, 0, [], str(e)
    except Exception as e:  # noqa: BLE001
        return code, kind, ok, -1, 0, [], f"BUG {type(e).__name__}"
    effs = [e for e in regs if isinstance(e, tr.Effect)]
    hand = any(e.f.get("SetCode") == [K["EFFECT_SPSUMMON_PROC"]] for e in effs)
    if hand:  # the static restriction that accompanies a hand procedure takes up no effect (translate())
        effs = [e for e in effs if e.f.get("SetCode") != [K["EFFECT_SPSUMMON_CONDITION"]]]
    n_tuple = sum(isinstance(e, tuple) for e in regs)
    bad = []
    if not ok:
        for e in effs:
            try:
                t.allocate(t.effect(e))
            except tr.Unsupported as x:
                bad.append(str(x))
            except Exception as x:  # noqa: BLE001
                bad.append(f"BUG {type(x).__name__}")
    n = len(effs) + n_tuple
    return code, kind, ok, n, n - len(bad), bad, (None if ok or bad else why)


def scan(codes, jobs):
    if jobs <= 1:
        return [one(c) for c in codes]
    with ProcessPoolExecutor(jobs) as ex:
        return list(ex.map(one, codes, chunksize=64))


def summarize(rows, top):
    out = []
    P = out.append
    n = len(rows)
    ok = [r for r in rows if r[2]]
    known = [r for r in rows if r[3] >= 0]
    ne, nok = sum(r[3] for r in known), sum(r[4] for r in known)
    P(f"- Cards 100% translated: **{len(ok)} / {n}** ({100 * len(ok) / n:.1f} %)")
    P(f"- Effects translated: **{nok} / {ne}** ({100 * nok / ne:.1f} %), over the {len(known)} cards whose `initial_effect` is read "
      f"({n - len(known)} cards rejected at `initial_effect`, effects not counted)")
    P(f"- Mean share of effects translated per card: {100 * sum(r[4] / max(r[3], 1) for r in known) / len(known):.1f} %")
    P("")
    P("## By card type")
    P("")
    P("| type | cards | 100% | effects translated |")
    P("|---|---|---|---|")
    for k in ("monster", "extra", "spell", "trap", "?"):
        rs = [r for r in rows if r[1] == k]
        kn = [r for r in rs if r[3] >= 0]
        if rs:
            e, o = sum(r[3] for r in kn), sum(r[4] for r in kn)
            P(f"| {k} | {len(rs)} | {sum(r[2] for r in rs)} ({100 * sum(r[2] for r in rs) / len(rs):.1f} %) | "
              f"{o} / {e} ({100 * o / max(e, 1):.1f} %) |")
    P("")
    P("## Distance of the untranslated cards")
    P("")
    P("Effects still rejected per card (0 = all effects pass, the card is rejected by a card-level rule):")
    P("")
    P("| rejected effects | cards | cumulative with the 100% cards |")
    P("|---|---|---|")
    hist = collections.Counter(min(r[3] - r[4], 6) for r in known if not r[2])
    cum = len(ok)
    for d in range(7):
        cum += hist[d]
        P(f"| {d if d < 6 else '6+'} | {hist[d]} | {100 * cum / n:.1f} % |")
    P(f"| `initial_effect` rejected | {n - len(known)} | 100 % |")
    P("")
    # reasons: each untranslated card has a set of reasons (its rejected effects, or its card-level reason)
    touched, sole, sole_fam, touched_fam, ex = (collections.Counter() for _ in range(5))
    ex = {}
    for r in rows:
        if r[2]:
            continue
        rs = set(r[5]) or {r[6]}
        fs = {family(x) for x in rs}
        for x in rs:
            touched[x] += 1
            ex.setdefault(x, r[0])
        for x in fs:
            touched_fam[x] += 1
        if len(rs) == 1:
            sole[next(iter(rs))] += 1
        if len(fs) == 1:
            sole_fam[next(iter(fs))] += 1
    one_reason = sum(sole.values())
    P(f"Untranslated cards with a single visible reason: **{one_reason}** ({100 * one_reason / n:.1f} % of the catalogue); "
      f"a single family of reasons: {sum(sole_fam.values())}.")
    P("")
    P("## Rejection reasons: cards blocked by this reason alone")
    P("")
    P("Upper bound of what removing the reason finishes at once (another reason may hide behind it).")
    P("")
    P("| sole visible reason | cards affected | reason | example |")
    P("|---|---|---|---|")
    for x, v in sole.most_common(top):
        P(f"| {v} | {touched[x]} | {x[:110]} | {ex[x]} |")
    P("")
    P("## Families of reasons")
    P("")
    P("| cards affected | sole visible family | family |")
    P("|---|---|---|")
    for x, v in touched_fam.most_common(top):
        P(f"| {v} | {sole_fam[x]} | {x} |")
    return "\n".join(out)


def diff(rows, ref):
    """Movement since a snapshot: cards finished, lost, brought closer (fewer rejected effects), effects gained."""
    old = {int(k): v for k, v in ref.get("cards", ref.get("cartes", {})).items()}  # "cartes": older snapshots
    won = lost = closer = farther = 0
    de = 0
    for code, kind, ok, n, nok, bad, why in rows:
        o = old.get(code)
        if o is None:
            continue
        o_ok, o_n, o_nok = o[0], o[1], o[2]
        won += ok and not o_ok
        lost += o_ok and not ok
        if n >= 0 and o_n >= 0:
            de += nok - o_nok
            if not ok and not o_ok:
                closer += (n - nok) < (o_n - o_nok)
                farther += (n - nok) > (o_n - o_nok)
    return (f"Since snapshot {ref.get('commit', '?')}: **{won} cards finished**, {lost} lost; "
            f"**{closer} cards brought closer** (fewer rejected effects, not yet finished), {farther} moved away; "
            f"effects translated {de:+d}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--json", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    import translate as tr
    codes = [int(x) for x in a.codes.split(",")] if a.codes else tr.all_codes()
    rows = scan(codes, a.jobs)
    try:
        import subprocess
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "?"
    text = f"# Translator progress (generated by translate/progress.py, commit {commit})\n\n"
    if a.ref:
        text += diff(rows, json.load(_open(a.ref, "r"))) + "\n\n"
    text += summarize(rows, a.top) + "\n"
    print(text)
    if a.md:
        open(a.md, "w").write(text)
    if a.json:
        reasons = sorted({x for r in rows for x in r[5]} | {r[6] for r in rows if r[6]})
        rid = {x: i for i, x in enumerate(reasons)}
        cards = {str(r[0]): [int(r[2]), r[3], r[4], sorted(rid[x] for x in set(r[5])) or ([rid[r[6]]] if r[6] else [])]
                 for r in rows}
        with _open(a.json, "w") as fh:
            json.dump(dict(commit=commit, reasons=reasons, cards=cards), fh, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
