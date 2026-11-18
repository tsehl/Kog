"""Inventory: what share of the 13,541 official cards can the engine play, and which features are missing,
ranked by number of cards unlocked?

Input: translate/cache/extract.json.gz (translate/extract.py). Each card is a set of items (API calls, methods,
constants). An item is:
- "ignored": display, Lua plumbing, generic data (Duel.Hint, aux.Stringid, s.filter, RACE_*, SET_*...);
- "covered": present in the script of at least one card that the engine implements (engine/cards.py CARDS,
  checked against the oracle). Approximation: a covered card may use only part of a feature
  (e.g. EFFECT_UPDATE_ATTACK yes, but not with all its values); the real measure will come from the translator;
- "missing" otherwise.
A card is playable (in the sense of this inventory) if it has no missing item.

Ranking: greedy. At each step, the feature that makes the most cards playable is added (counting the cards it
unlocks alone), which gives the coverage curve. With --popular, the cards are weighted by their presence in
decks (no data yet: to come).

Usage: uv run --with jax python translate/inventory.py [--steps 150] [--out translate/INVENTORY.md]"""
import argparse
import collections
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "engine")]
IN = os.path.join(HERE, "cache", "extract.json.gz")

# display, Lua plumbing and generic data (the engine has them as card fields, or they change nothing in the game)
IGNORE = re.compile(
    r"^(GetID|aux\.Stringid|Duel\.Hint|Duel\.HintSelection|Duel\.ConfirmCards|:SetDescription|:SetHintTiming|"
    r"HINTMSG_.*|aux\.RegisterClientHint|EFFECT_FLAG_CLIENT_HINT|Duel\.BreakEffect|"
    r"s\..*|[a-z_]+\.[a-z_]+|math\..*|table\..*|string\..*|ipairs|pairs|type|tostring|tonumber|select|next|print|"
    r"RACE_.*|ATTRIBUTE_.*|SET_.*|Effect\.CreateEffect|:RegisterEffect|:SetType|:SetCode|:SetOperation|:SetTarget|"
    r":SetCondition|:SetCost|:SetCategory|:GetHandler|:GetHandlerPlayer|:Clone|CATEGORY_.*|Duel\.SetOperationInfo|"
    r"Duel\.SetPossibleOperationInfo|:GetFirst|:GetNext|:Iter|aux\.Next|:Filter|:FilterCount|:IsExists|:Select|"
    r":FilterSelect|:GetCount|:Merge|:AddCard|:RemoveCard|:Sub|:Clear|:KeepAlive|:DeleteGroup|Group\.CreateGroup|"
    r"Group\.FromCards|:IsContains|:Match|:GetClassCount|:GetSum|:SetLabel|:GetLabel|:SetLabelObject|"
    r":GetLabelObject|aux\.FilterBoolFunction|aux\.FilterBoolFunctionEx|aux\.FaceupFilter|aux\.AND|aux\.OR|aux\.NOT|"
    r"aux\.TargetBoolFunction|Duel\.ShuffleHand|Duel\.ShuffleDeck|:GetOwner|:GetControler|:GetCode|:IsCode|"
    r":IsSetCard|:IsRace|:IsAttribute|:GetRace|:GetAttribute|:IsType|:IsMonster|:IsSpell|:IsTrap|:IsSpellTrap|"
    r":IsFaceup|:IsFacedown|:IsControler|:IsLocation|:GetLocation|:GetAttack|:GetDefense|:GetLevel|"
    r":IsLevelBelow|:IsLevelAbove|:IsAttackBelow|:IsAttackAbove|:IsDefenseBelow|:IsDefenseAbove|"
    r"TYPE_.*|POS_.*|LOCATION_.*|PLAYER_.*|REASON_EFFECT|RESET_EVENT|RESETS_STANDARD|RESET_PHASE|PHASE_END)$")

PLUMBING_NOTE = ("generic plumbing and data (groups, basic filters, display): the translator must "
                 "understand them, but they are not engine features")


def load():
    with gzip.open(IN, "rt") as fh:
        d = json.load(fh)
    cards = {}
    for code, v in d.items():
        if "err" in v:
            continue
        s = set()
        for items in v["fn"].values():
            s.update(items)
        cards[int(code)] = s
    return cards


def supported_codes():
    import cards as engine_cards
    return set(engine_cards.CARDS)


def greedy(missing, steps):
    """missing: {code: set of missing items}. Returns [(item, cards unlocked at this step, total playable)]."""
    left = {c: set(m) for c, m in missing.items() if m}
    done = sum(1 for m in missing.values() if not m)
    out = []
    added = set()
    for _ in range(steps):
        # score of an item: cards for which it is the only gap (after the previous additions); on a tie, number of
        # cards that use it
        alone = collections.Counter()
        uses = collections.Counter()
        for m in left.values():
            uses.update(m)
            if len(m) == 1:
                alone.update(m)
        if not uses:
            break
        best = max(uses, key=lambda k: (alone[k], uses[k]))
        added.add(best)
        freed = []
        for c in list(left):
            left[c].discard(best)
            if not left[c]:
                freed.append(c)
                del left[c]
        done += len(freed)
        out.append((best, len(freed), uses[best], done))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(HERE, "INVENTORY.md"))
    a = ap.parse_args()
    cards = load()
    sup = supported_codes()
    known = [c for c in sup if c in cards]
    ok = set().union(*(cards[c] for c in known))
    ok = {i for i in ok if not IGNORE.match(i)}
    missing = {c: {i for i in s if not IGNORE.match(i) and i not in ok} for c, s in cards.items()}
    n = len(cards)
    playable = sum(1 for m in missing.values() if not m)
    freq = collections.Counter()
    for m in missing.values():
        freq.update(m)
    nmiss = collections.Counter(min(len(m), 20) for m in missing.values())
    steps = greedy(missing, a.steps)

    L = [f"# Feature inventory (generated by translate/inventory.py)\n",
         f"{n} official scripts; {len(known)} cards implemented in the engine (engine/cards.py); "
         f"{len(ok)} covered items (present in their scripts), "
         f"{len({i for s in cards.values() for i in s if not IGNORE.match(i)})} non-ignored items in total.\n",
         f"Cards with no missing item: **{playable} ({100 * playable / n:.1f} %)**. Ignored: {PLUMBING_NOTE}.\n",
         "Approximation: 'covered' = seen in an implemented card; the exact measure will come from the translator "
         "(cards translated and checked against the oracle).\n",
         "## Number of missing items per card\n", "| missing | cards | cumulative |", "|---|---|---|"]
    cum = 0
    for k in range(21):
        cum += nmiss[k]
        L.append(f"| {k}{'+' if k == 20 else ''} | {nmiss[k]} | {100 * cum / n:.1f} % |")
    L += ["\n## Greedy curve (the feature that unlocks the most cards at each step)\n",
          "| step | item | unlocked | used by (remaining) | playable | % |", "|---|---|---|---|---|---|"]
    for i, (item, fr, us, done) in enumerate(steps):
        L.append(f"| {i + 1} | `{item}` | {fr} | {us} | {done} | {100 * done / n:.1f} % |")
    L += ["\n## Most frequent missing items (all cards)\n", "| item | cards |", "|---|---|"]
    for item, k in freq.most_common(200):
        L.append(f"| `{item}` | {k} |")
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"{n} cards, {len(known)} implemented, playable {playable} ({100 * playable / n:.1f} %) → {a.out}")
    for i, (item, fr, us, done) in enumerate(steps[:40]):
        print(f"{i + 1:3d} {item:45s} +{fr:5d} (used {us:5d}) → {done:5d} ({100 * done / n:.1f} %)")


if __name__ == "__main__":
    main()
