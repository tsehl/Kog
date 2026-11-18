"""Effect skeletons: the SHAPE of the effects the translator knows how to read, without their parameters.

The hand-written generator (cardgen.Gen) falls behind the engine with each mechanic the translator adds. Here, the
effects of the translated cards (translate.translate) are filed in a catalog of skeletons, and the generator draws new
cards from it: the structure of the effect is kept, its parameters are redrawn.

Kept: the kind of the effect, the sequence of instructions of the three programs, the events, the locations, the
counters. Redrawn: attributes, monster types, archetypes (replaced by the procedural archetypes), ATK / DEF / Level
bounds of the filters, amounts (damage, recovery, LP payment, bonus), and the whole frame of the card (Level, stats,
attribute, type). The real cards do not enter training: neither their codes, nor their names, nor their values; an
effect that designates a card by its name is dropped from the catalog.

Draw: uniform over the SHAPES (kind + opcodes), not over the original cards. The frequency of a shape in the official
catalog says nothing about its frequency in a deck; a flattened draw also shows the rare shapes.

The catalog is a file (gen/skeletons.pkl.gz) built once by `build`: training needs neither the official scripts nor
luaparser.

Usage: uv run --with luaparser --with jax python gen/skeleton.py        (rebuilds the catalog, prints a summary)"""
import collections
import gzip
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import cards as C  # noqa: E402
import core  # noqa: E402

PATH = os.path.join(HERE, "skeletons.pkl.gz")
BIG = core.BIG
# positions of the fields of a filter (cards.F)
F_ATK_LO, F_ATK_HI, F_DF_LO, F_DF_HI, F_ATTR, F_RACE, F_SETC, F_CODE0, F_CODE1, F_LV_LO, F_LV_HI = 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16
FLEN = 24
GROUPS = ("monster", "spell", "quick-play spell", "continuous spell", "trap", "continuous trap", "counter trap")


def is_filter(x):
    return isinstance(x, tuple) and len(x) == FLEN and all(isinstance(v, (int, np.integer)) for v in x)


def shape_of(e):
    """Shape of an effect: kind + opcodes of the three programs."""
    return (int(e[0]),) + tuple(tuple(int(i[0]) for i in p) for p in e[4:7])


def _filters(e):
    out = [i[2] for p in e[4:7] for i in p if i[0] == core.OP_FILTER and is_filter(i[2])]
    if len(e) > 7:
        out += [f for f in e[7].get("filters", ()) if is_filter(f)]
    return out


def group_of(ctype):
    if ctype & core.TYPE_MONSTER:
        return "monster"
    if ctype & core.TYPE_SPELL:
        return ("quick-play spell" if ctype & core.TYPE_QUICKPLAY else "continuous spell" if ctype & core.TYPE_CONTINUOUS
                else "spell")
    return "counter trap" if ctype & core.TYPE_COUNTER else "continuous trap" if ctype & core.TYPE_CONTINUOUS else "trap"


def build(path=PATH, codes=None, verbose=True):
    """Catalog from the translated cards: {group: [unit]}, unit = list of the effects of an original card."""
    sys.path.insert(0, os.path.join(HERE, "..", "translate"))
    import translate as tr
    why = collections.Counter()
    cat = {g: [] for g in GROUPS}
    seen = set()
    for code in (codes or tr.all_codes()):
        try:
            effs, notes, proc, descs = tr.translate(code)
            ctype = tr.card_type(code)
        except tr.Unsupported:
            continue
        except Exception:  # noqa: BLE001
            why["translator exception"] += 1
            continue
        effs = [tuple(e) for e in effs]
        if not effs:
            why["no effect"] += 1
        elif ctype & (core.TYPE_FUSION | core.TYPE_SYNCHRO | core.TYPE_XYZ | core.TYPE_LINK | core.TYPE_RITUAL
                      | core.TYPE_PENDULUM | 0x40000 | 0x80000):   # Extra Deck, Ritual, Pendulum, Equip, Field
            why["frame not taken (Extra Deck, Ritual, Pendulum, Equip, Field)"] += 1
        elif proc is not None or getattr(effs, "revive", 0) or any(len(e) > 7 and e[7].get("revive") for e in effs):
            why["procedure or restricted summon"] += 1
        elif any(f[F_CODE0] or f[F_CODE1] for e in effs for f in _filters(e)):
            why["designates a card by name"] += 1
        elif any(e[0] == core.K_PROC for e in effs):
            why["hand procedure"] += 1
        else:
            unit = [(e if len(e) > 7 else e + ({},)) for e in effs]
            key = repr(unit)
            if key in seen:
                why["exact duplicate"] += 1
                continue
            seen.add(key)
            cat[group_of(ctype)].append(unit)
            why["kept"] += 1
    with gzip.open(path, "wb") as fh:
        pickle.dump(cat, fh, protocol=4)
    if verbose:
        for k, v in why.most_common():
            print(f"{v:6d}  {k}")
        for g in GROUPS:
            sh = {tuple(shape_of(e) for e in u) for u in cat[g]}
            print(f"  {g:16s} {len(cat[g]):4d} units, {len(sh):4d} card shapes, "
                  f"{len({shape_of(e) for u in cat[g] for e in u}):4d} effect shapes")
    return cat


_CAT = {}


def catalog(path=PATH):
    """{group: {card shape: [units]}}; loaded once."""
    if path not in _CAT:
        with gzip.open(path, "rb") as fh:
            raw = pickle.load(fh)
        out = {}
        for g, units in raw.items():
            by = collections.OrderedDict()
            for u in units:
                by.setdefault(tuple(shape_of(e) for e in u), []).append(u)
            out[g] = by
        _CAT[path] = out
    return _CAT[path]


# ---------------------------------------------------------------------- redrawing of the parameters

def _bound(rng, v, step, lo, hi):
    """Finite bound of a filter, shifted by one or two steps."""
    if abs(v) >= BIG:
        return v
    return int(np.clip(v + step * int(rng.integers(-2, 3)), lo, hi))


def refilter(rng, f, attrs, races, setmap):
    """Filter with the qualifiers redrawn. attrs / races / setmap: draws memorized for the card (the same original
    value gives the same drawn value in all the effects of the card)."""
    f = list(f)
    if f[F_ATTR]:
        f[F_ATTR] = attrs.setdefault(f[F_ATTR], attrs["pick"]())
    if f[F_RACE]:
        f[F_RACE] = races.setdefault(f[F_RACE], races["pick"]())
    if f[F_SETC]:
        f[F_SETC] = setmap.setdefault(f[F_SETC], setmap["pick"]())
    f[F_ATK_LO], f[F_ATK_HI] = sorted((_bound(rng, f[F_ATK_LO], 100, 0, 5000), _bound(rng, f[F_ATK_HI], 100, 0, 5000)))
    f[F_DF_LO], f[F_DF_HI] = sorted((_bound(rng, f[F_DF_LO], 100, 0, 5000), _bound(rng, f[F_DF_HI], 100, 0, 5000)))
    lo, hi = _bound(rng, f[F_LV_LO], 1, 1, 12), _bound(rng, f[F_LV_HI], 1, 1, 12)
    f[F_LV_LO], f[F_LV_HI] = (lo, hi) if lo <= hi else (hi, lo)
    return tuple(int(v) for v in f)


def _amount(rng, n, step=100, lo=100, hi=4000):
    """Amount (LP, damage, bonus): between half and one and a half times the original, rounded."""
    if n == 0:
        return 0
    s = -1 if n < 0 else 1
    return s * int(np.clip(round(abs(n) * float(rng.choice([0.5, 0.75, 1.0, 1.0, 1.25, 1.5])) / step) * step, lo, hi))


def reparam(rng, unit, attrs, races, setmap):
    """Unit (effects of an original card) -> same effects, parameters redrawn."""
    out = []
    for e in unit:
        progs = []
        for p in e[4:7]:
            ins = []
            for i in p:
                op, a, b, c = i
                if op == core.OP_FILTER and is_filter(b):
                    b = refilter(rng, b, attrs, races, setmap)
                elif op in (core.OP_DAMAGE, core.OP_RECOVER) and isinstance(b, (int, np.integer)) and abs(b) >= 100:
                    b = _amount(rng, int(b))
                elif op == core.OP_PAY_LP and isinstance(a, (int, np.integer)) and a >= 100:
                    a = _amount(rng, int(a))
                elif op == core.OP_MODIFY and isinstance(c, (int, np.integer)) and (b & 31) in (core.MC_ATK, core.MC_DEF) \
                        and ((b >> 5) & 15) == core.MV_CONST and abs(c) >= 100:
                    c = _amount(rng, int(c))
                ins.append((op, a, b, c))
            progs.append(ins)
        extra = dict(e[7])
        if "filters" in extra:
            extra["filters"] = [refilter(rng, f, attrs, races, setmap) if is_filter(f) else f for f in extra["filters"]]
        out.append((e[0], e[1], e[2], e[3], progs[0], progs[1], progs[2], extra))
    return out


def draw(rng, group, attr_pick, race_pick, set_pick, max_effects):
    """A re-parameterized unit of the group, card shape drawn uniformly. None if the group is empty."""
    by = catalog().get(group)
    if not by:
        return None
    shapes = [s for s in by if len(s) <= max_effects]
    if not shapes:
        return None
    units = by[shapes[int(rng.integers(len(shapes)))]]
    unit = units[int(rng.integers(len(units)))]
    return reparam(rng, unit, {"pick": attr_pick}, {"pick": race_pick}, {"pick": set_pick})


if __name__ == "__main__":
    build()
