"""Benchmark on real cards (never seen in training): bank and reference decks.

The bank is built from the oracle's database (`python train/realcards.py` -> train/realbank.npz), then reloaded
without the oracle (on GPU, for example). It is padded to the shapes of the procedural banks (C = 480 cards, tprog at
2 x PT): same shapes, hence no new compilation of the rollout functions.

Four decks of 40 cards + Extra Deck of 15, built by hand with the cards the engine knows how to play:
control (generic Spells / Traps), Synchro / Xyz / Link, Ritual / Fusion, Pendulum. The fictitious test cards
(Test Link) are never in them."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", d) for d in ("engine", "gen", "oracle")]
PATH = os.path.join(HERE, "realbank.npz")
C_PAD = 480  # cards of a procedural bank (ppo.py: --main 400 --extra 80)
# fields of core.Bank whose axis 1 is the effect (K_NONE = 0: a zero effect is empty)
E_FIELDS = ("ekind", "prog", "tprog", "econd", "filt", "etrig", "eopt", "erng", "emod", "edmg", "einline", "erefund")

LINKS = [("Knightmare Phoenix", 1), ("Knightmare Cerberus", 1), ("LANphorhynchus", 1),
         ("Gaia Saber, the Lightning Shadow", 1), ("Traffic Ghost", 1), ("Berserker of the Tenyi", 1)]
DECKS = {
    "controle": dict(main=[
        ("Sangan", 1), ("Witch of the Black Forest", 1), ("Mystic Tomato", 2), ("Exiled Force", 1), ("D.D. Crow", 1),
        ("Effect Veiler", 1), ("Command Knight", 2), ("Cannon Soldier", 1), ("Lord of D.", 1),
        ("Obnoxious Celtic Guard", 2), ("Man-Eater Bug", 1), ("Dark Magician Girl", 1), ("Kazejin", 1),
        ("Arsenal Bug", 2), ("Winged Rhynos", 1), ("Big-Tusked Mammoth", 1),
        ("Pot of Greed", 1), ("Graceful Charity", 1), ("Raigeki", 1), ("Dark Hole", 1), ("Monster Reborn", 1),
        ("Harpie's Feather Duster", 1), ("Mystical Space Typhoon", 2), ("Upstart Goblin", 1), ("Shrink", 1),
        ("Forbidden Chalice", 1),
        ("Mirror Force", 1), ("Torrential Tribute", 1), ("Trap Hole", 2), ("Sakuretsu Armor", 2),
        ("Magic Jammer", 1), ("Seven Tools of the Bandit", 1), ("Jar of Greed", 1)],
        extra=LINKS + [("Gem-Knight Pearl", 3), ("Gagagigo the Risen", 3), ("Number 17: Leviathan Dragon", 3)]),
    "synchro": dict(main=[
        ("Tune Warrior", 3), ("Genex Controller", 2), ("Ally Mind", 1), ("Water Spirit", 1), ("Flamvell Guard", 2),
        ("Command Knight", 2), ("Obnoxious Celtic Guard", 2), ("Lord of D.", 1), ("Arsenal Bug", 2),
        ("Mystic Tomato", 2), ("Big-Tusked Mammoth", 1), ("Ally of Justice Quarantine", 2),
        ("Elemental HERO Wildheart", 2), ("Ghost Ogre & Snow Rabbit", 1),
        ("Pot of Greed", 1), ("Monster Reborn", 1), ("E - Emergency Call", 2), ("Rush Recklessly", 2),
        ("Banner of Courage", 1), ("Mystical Space Typhoon", 2), ("Upstart Goblin", 2),
        ("Mirror Force", 1), ("Trap Hole", 2), ("Reverse Glasses", 1), ("Just Desserts", 1)],
        extra=[("Gaia Knight, the Force of Earth", 2), ("Scrap Archfiend", 2), ("Naturia Leodrake", 1),
               ("Gem-Knight Pearl", 2), ("Gagagigo the Risen", 1), ("Number 17: Leviathan Dragon", 1)] + LINKS),
    "rituel_fusion": dict(main=[
        ("Black Luster Soldier", 2), ("Black Luster Ritual", 2), ("Demise, King of Armageddon", 2),
        ("End of the World", 2), ("Polymerization", 3), ("Flame Manipulator", 2), ("Masaki the Legendary Swordsman", 2),
        ("Gaia The Fierce Knight", 2), ("Curse of Dragon", 2), ("Blue-Eyes White Dragon", 3),
        ("Sangan", 1), ("Witch of the Black Forest", 2), ("Mystic Tomato", 2), ("Man-Eater Bug", 1),
        ("Pot of Greed", 1), ("Graceful Charity", 1), ("Monster Reborn", 1), ("Dark Hole", 1), ("Upstart Goblin", 2),
        ("Gravity Bind", 1), ("Castle Walls", 1), ("Mirror Force", 1), ("Dian Keto the Cure Master", 1),
        ("Mystical Space Typhoon", 1), ("Trap Hole", 1)],
        extra=[("Flame Swordsman", 3), ("Gaia the Dragon Champion", 3), ("Blue-Eyes Ultimate Dragon", 3)] + LINKS),
    "pendule": dict(main=[
        ("Flash Knight", 3), ("Mandragon", 2), ("Fire Opal Head", 3), ("Gallugallud the Endless Enigma", 2),
        ("Dragong", 3), ("Lord of D.", 1), ("Mystic Tomato", 1), ("Bladefly", 1), ("Deepsea Macrotrema", 1),
        ("The Little Swordsman of Aile", 2), ("Command Knight", 2), ("Winged Rhynos", 2), ("Exiled Force", 1),
        ("Hinotama", 2), ("Banner of Courage", 1), ("Rush Recklessly", 2), ("Shrink", 1), ("Monster Reborn", 1),
        ("Pot of Greed", 1), ("Fissure", 1), ("Back-Up Rider", 1),
        ("Skill Drain", 1), ("Sakuretsu Armor", 2), ("Trap Hole", 1), ("Just Desserts", 1), ("Castle Walls", 1)],
        extra=LINKS + [("Gem-Knight Pearl", 3), ("Gagagigo the Risen", 3), ("Number 17: Leviathan Dragon", 3)]),
}


# "Proto-archetype" decks: built only with cards translated from the official scripts
# (translate/translate.py), around the archetypes best covered by the translator. They are not real tournament decks
# (no archetype is complete: 6 to 9 translated cards) but they have real synergies (searches, Summons from the
# hand / GY, discards, Xyz / Synchro materials), whereas DECKS measures basic play.
# No Equip Spells: the generator does not produce any (the bank would keep an equip width > 0, hence another engine
# compilation, for a mechanic the network has never seen).
# Separate bank: realbank_proto.npz (`python train/realcards.py --proto`), chosen by YGO_REAL=proto.
LINKS2 = [("LANphorhynchus", 1), ("Traffic Ghost", 1), ("Gaia Saber, the Lightning Shadow", 1)]
PROTO_DECKS = {
    "constellar": dict(main=[   # LIGHT, Rank 4 and 5 Xyz, Summons from the hand
        ("Constellar Algiedi", 3), ("Constellar Acubens", 2), ("Constellar Aldebaran", 2), ("Constellar Sheratan", 2),
        ("Constellar Zubeneschamali", 2), ("Constellar Virgo", 2), ("Constellar Alrescha", 1), ("Constellar Antares", 1),
        ("Shining Angel", 3), ("Senju of the Thousand Hands", 2),
        ("Pot of Greed", 1), ("Graceful Charity", 1), ("Monster Reborn", 1), ("Mystical Space Typhoon", 2),
        ("Twin Twisters", 1), ("Raigeki", 1), ("Dark Hole", 1), ("Foolish Burial", 1), ("Shrink", 2),
        ("Rush Recklessly", 1),
        ("Mirror Force", 1), ("Torrential Tribute", 1), ("Compulsory Evacuation Device", 1), ("Dimensional Prison", 2),
        ("Sakuretsu Armor", 1), ("Solar Ray", 1), ("Reinforcements", 1)],
        extra=[("Constellar Pleiades", 3), ("Adreus, Keeper of Armageddon", 1),
               ("Gem-Knight Pearl", 2), ("Fairy Cheer Girl", 2), ("Gagagigo the Risen", 2),
               ("Leviair the Sea Dragon", 2)] + LINKS2),
    "mermail": dict(main=[      # WATER, discard to search / Summon, recovery from the GY
        ("Mermail Abysspike", 3), ("Mermail Abyssturge", 2), ("Mermail Abysslinde", 3), ("Mermail Abyssgunde", 2),
        ("Mermail Abysshilde", 2), ("Abyss Soldier", 2), ("Deep Sea Diva", 2), ("Spined Gillman", 2),
        ("Mobius the Frost Monarch", 1),
        ("Salvage", 2), ("Moray of Avarice", 1), ("Water Hazard", 1), ("Pot of Greed", 1), ("Graceful Charity", 1),
        ("Monster Reborn", 1), ("Mystical Space Typhoon", 2), ("Twin Twisters", 1), ("Foolish Burial", 1),
        ("Raigeki", 1), ("Dark Hole", 1), ("Shrink", 1),
        ("Torrential Tribute", 1), ("Mirror Force", 1), ("Dimensional Prison", 2), ("Surface", 1),
        ("Compulsory Evacuation Device", 1), ("Sakuretsu Armor", 1)],
        extra=[("Leviair the Sea Dragon", 2), ("Submersible Carrier Aero Shark", 2), ("Black Ray Lancer", 2),
               ("Battlecruiser Dianthus", 1), ("Coral Dragon", 2), ("Gagagigo the Risen", 1), ("Gem-Knight Pearl", 2)] + LINKS2),
    "speedroid": dict(main=[    # WIND Machine, Tuners, Summons from the hand / the GY
        ("Speedroid Terrortop", 3), ("Speedroid Double Yoyo", 3), ("Speedroid Pachingo-Kart", 3),
        ("Speedroid Den-Den Daiko Duke", 3), ("Top Runner", 2),
        ("Gusto Gulldo", 2), ("Starship Spy Plane", 2), ("Winged Rhynos", 2),
        ("Speedroid Scratch", 2), ("Cards of Consonance", 1), ("Pot of Greed", 1), ("Graceful Charity", 1),
        ("Monster Reborn", 1), ("Mystical Space Typhoon", 2), ("Twin Twisters", 1), ("Raigeki", 1), ("Dark Hole", 1),
        ("Foolish Burial", 1), ("Rush Recklessly", 1), ("Shrink", 1),
        ("Mirror Force", 1), ("Torrential Tribute", 1), ("Dimensional Prison", 2), ("Compulsory Evacuation Device", 1),
        ("Sakuretsu Armor", 1)],
        extra=[("XX-Saber Hyunlei", 1), ("Orient Dragon", 1), ("Gaia Knight, the Force of Earth", 2),
               ("Scrap Archfiend", 3), ("Ally of Justice Light Gazer", 1),
               ("Gem-Knight Pearl", 1), ("Leviair the Sea Dragon", 1), ("Gagagigo the Risen", 2)]
              + LINKS2),
    "prophetie": dict(main=[    # Spellcasters, Spell-heavy deck (searches for Spellbooks)
        ("Spellbook Magician of Prophecy", 3), ("Charioteer of Prophecy", 1), ("Stoic of Prophecy", 1),
        ("Prophecy Destroyer", 1), ("Peten the Dark Clown", 2), ("Magician of Faith", 1), ("Old Vindictive Magician", 1),
        ("Witch of the Black Forest", 1), ("Mathematician", 1), ("Night's End Sorcerer", 1),
        ("Spellbook of Secrets", 3), ("Spellbook of Eternity", 2), ("Pot of Greed", 1),
        ("Graceful Charity", 1), ("Monster Reborn", 1), ("Mystical Space Typhoon", 2), ("Twin Twisters", 2),
        ("Raigeki", 1), ("Dark Hole", 1), ("Magical Stone Excavation", 2), ("Spell Reproduction", 1),
        ("Shrink", 1), ("Lightning Vortex", 2),
        ("Type Zero Magic Crusher", 1), ("Magic Jammer", 1), ("Mirror Force", 1), ("Torrential Tribute", 1),
        ("Dimensional Prison", 1), ("Sakuretsu Armor", 1), ("Compulsory Evacuation Device", 1)],
        extra=[("Cyberse Integrator", 2), ("Underworld Fighter Balmung", 2), ("Coral Dragon", 1),
               ("XX-Saber Hyunlei", 1), ("Gaia Knight, the Force of Earth", 1), ("Orient Dragon", 1),
               ("Leviair the Sea Dragon", 1), ("Gem-Knight Pearl", 2), ("Coral Dragon", 1)] + LINKS2),
}
PATH_PROTO = os.path.join(HERE, "realbank_proto.npz")


def translated(codes=None):
    """Usable translated cards (cache of translate/validate.py): {code: (effects, recipe or None, descriptions)}."""
    sys.path.insert(0, os.path.join(HERE, "..", "translate"))
    import validate
    import driver
    res = validate.translate_all(False)
    db = driver.CardDB()
    out = {}
    for c, v in res.items():
        if v[0] != "ok" or c not in db.rows or (codes is not None and c not in codes):
            continue
        t, proc = db.rows[c][3], v[3]
        if proc is None and t & validate.EXTRA_TYPES:
            continue
        if proc is not None and (t & validate.EXTRA_TYPES != validate.PROC_TYPE[proc[0]]
                                 or (proc[0] == "xyz" and proc[3] != db.rows[c][6] & 0xFF)):
            continue
        out[c] = (v[1], proc, v[4])
    return out


def proto_bank(db, codes, tr):
    """Bank of the code cards: hand-written version (engine/cards.py, checked by engine/regress.sh) for the
    classics that have one, translated version for the others (same recipes as translate/validate.py)."""
    import core
    import cards
    recipes = dict(fusion={}, synchro={}, xyz={}, link={})
    for c in codes:
        proc = tr[c][1]
        if proc is not None:
            k, f, *rest = proc
            recipes[k][c] = (cards.FusionRecipe(f, rest[0]) if k == "fusion"
                             else (f, rest[0], rest[1], 0) if k == "link" else (f, rest[0]))
    r = [db.rows[c] for c in codes]
    bank = core.make_bank(np.array(codes), [x[4] for x in r], [x[5] for x in r], [x[6] & 0xFF for x in r],
                          lsc=[(x[6] >> 24) & 0xFF for x in r], rsc=[(x[6] >> 16) & 0xFF for x in r],
                          ctype=[x[3] for x in r], race=[x[7] for x in r], attr=[x[8] for x in r],
                          setcode=[cards.setcodes(x[2]) for x in r],
                          **cards.assemble(codes, cards={c: cards.CARDS[c] if c in cards.CARDS else tr[c][0] for c in codes},
                                           ritual={}, **recipes))
    return bank, {c: i for i, c in enumerate(codes)}


def _expand(lst):
    return [name for name, k in lst for _ in range(k)]


def build(path=PATH, proto=False):
    """Bank of the cards of the decks (oracle database), padded to the procedural shapes, and decks -> npz.
    proto: PROTO_DECKS decks, translated cards (translate/translate.py) instead of the hand-written cards."""
    import driver
    import compare
    import cardgen
    db = driver.CardDB()
    decks = PROTO_DECKS if proto else DECKS
    by_name = {}
    for c, nm in db.names.items():
        if c in db.rows and db.rows[c][1] == 0:   # not the alternate versions (alias)
            by_name.setdefault(nm, c)
    missing = sorted({n for d in decks.values() for n, _ in d["main"] + d["extra"]} - set(by_name))
    assert not missing, f"cards not found: {missing}"
    for nm, d in decks.items():
        assert len(_expand(d["main"])) == 40, (nm, len(_expand(d["main"])))
        assert len(_expand(d["extra"])) == 15, (nm, len(_expand(d["extra"])))
    for nm, d in decks.items():
        cnt = {}
        for n in _expand(d["main"]) + _expand(d["extra"]):
            cnt[n] = cnt.get(n, 0) + 1
        assert max(cnt.values()) <= 3, (nm, [n for n, k in cnt.items() if k > 3])
    codes = sorted({by_name[n] for d in decks.values() for n in _expand(d["main"]) + _expand(d["extra"])})
    if proto:
        import cards
        tr = translated(set(codes))
        absent = sorted(db.names[c] for c in codes if c not in tr and c not in cards.CARDS)
        assert not absent, f"cards not translated or unusable: {absent}"
        bank, cmap = proto_bank(db, codes, tr)
    else:
        bank, cmap = compare.make_bank(db, codes)
    out = {}
    for k in bank._fields:
        x = np.asarray(getattr(bank, k))
        if k == "tprog":
            x = np.pad(x, ((0, 0), (0, 0), (0, 2 * cardgen.PT - x.shape[2]), (0, 0)))
        if k in ("einline", "erefund") and x.shape[1] == 0:   # same shapes as cardgen.to_bank
            x = np.zeros((x.shape[0], np.asarray(bank.ekind).shape[1]), x.dtype)
        assert x.shape[0] == len(codes), (k, x.shape)
        # padded with copies of the first card (never in a deck)
        out["bank_" + k] = np.concatenate([x, np.repeat(x[:1], C_PAD - len(codes), 0)])
    names = sorted(decks)
    out["deck_names"] = np.array(names)
    out["card_names"] = np.array([db.names[c] for c in codes] + [""] * (C_PAD - len(codes)))
    out["main"] = np.array([[cmap[by_name[n]] for n in _expand(decks[d]["main"])] for d in names], np.int32)
    out["extra"] = np.array([[cmap[by_name[n]] for n in _expand(decks[d]["extra"])] for d in names], np.int32)
    np.savez_compressed(path, **out)
    print(f"{path} : {len(codes)} cards (padded to {C_PAD}), decks {names}")


def default_path():
    """YGO_REAL=proto: bank of the proto-archetype decks; otherwise the 4 reference decks."""
    return PATH_PROTO if os.environ.get("YGO_REAL", "") == "proto" else PATH


def load(path=None):
    """-> (core.Bank, deck names, main [D, 40], extra [D, 15]); card names: card_names(path)."""
    import jax.numpy as jnp
    import core
    z = np.load(path or default_path())
    c = z["bank_code"].shape[0]
    # optional fields added after the file was created: width 0 (path eliminated at tracing)
    get = lambda k: z["bank_" + k] if "bank_" + k in z else np.zeros((c, 0), np.int32)

    def fit(k, x):  # file created with fewer effects per card (core.E): empty effects added
        if k == "econd" and x.shape[1] == 0:
            return np.zeros((c, core.E), np.int32)
        if k in E_FIELDS and 0 < x.shape[1] < core.E:
            x = np.pad(x, [(0, 0), (0, core.E - x.shape[1])] + [(0, 0)] * (x.ndim - 2))
        return x
    bank = core.Bank(**{k: jnp.asarray(fit(k, get(k))) for k in core.Bank._fields})
    return bank, list(z["deck_names"]), z["main"], z["extra"]


def card_names(path=None):
    """Name of each card of the bank ("" for the padding)."""
    return list(np.load(path or default_path())["card_names"])


if __name__ == "__main__":
    if "--proto" in sys.argv:
        build(PATH_PROTO, proto=True)
    else:
        build()
