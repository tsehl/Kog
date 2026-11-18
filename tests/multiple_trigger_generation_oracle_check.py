"""Procedural native/JAX regression runner focused on cards with >=2 trigger effects.

Banks 0-3, five games each, seed0 7400.
Usage: YGO_JAX_CACHE=/tmp/ygo-jax .venv/bin/python tests/multiple_trigger_generation_oracle_check.py
"""
import argparse
import collections
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "gen"
ENGINE = ROOT / "engine"
ORACLE = ROOT / "oracle"
sys.path[:0] = [str(GEN), str(ENGINE), str(ORACLE)]

import jax
import cardgen
import compare
import core
import driver
import tolua


def bank_range(value):
    lo, sep, hi = value.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def multi_trigger_indices(card):
    return tuple(i for i, effect in enumerate(card["effects"])
                 if effect.get("kind") == "trigger")


def decks(cset, rng, focus_half=True, main_size=40, extra_size=15):
    main = [c["code"] for c in cset if not cardgen.is_extra(c)]
    extra = [c["code"] for c in cset if cardgen.is_extra(c)]
    focused = [c["code"] for c in cset if not cardgen.is_extra(c)
               and len(multi_trigger_indices(c)) >= 2]
    if focus_half and not focused:
        raise RuntimeError("bank has no main-deck card with >=2 trigger effects")
    nfocus = main_size // 2 if focus_half else 0
    return ([[rng.choice(focused) for _ in range(nfocus)]
             + [rng.choice(main) for _ in range(main_size - nfocus)] for _ in range(2)],
            [[rng.choice(extra) for _ in range(extra_size)] for _ in range(2)])

def trigger_descs(cset):
    return {c["code"]: {(c["code"] << 20) | i for i in multi_trigger_indices(c)}
            for c in cset if len(multi_trigger_indices(c)) >= 2}


def watch_duel(state):
    base = driver.Duel
    class WatchedDuel(base):
        def _handle(self, msg):
            reader = driver.Reader(msg)
            typ = reader.take('B')
            if typ == driver.K['MSG_CHAINING']:
                code = reader.take('I')
                current_location = tuple(reader.take('BBII'))
                reader.take('BBI')
                desc, index = reader.take('QI')
                # A code alone does not identify a physical card when a deck
                # contains duplicate copies. Count matching source locations separately;
                # this is not a persistent identity if cards move during costs.
                identity = (code, *current_location)
                state['chain_links'][index] = (code, desc, identity)
                state['native_activations'][(code, desc)] += 1
            elif typ == driver.K['MSG_CHAIN_SOLVED']:
                index = reader.take('B')
                item = self.chains.get(index)
                if item is not None:
                    code, desc, negated = item
                    if not negated:
                        state['native_resolutions'][(code, desc)] += 1
                        if desc in state['multi_descs'].get(code, set()):
                            link = state['chain_links'].get(index)
                            if link is not None:
                                state['resolved_this_chain'].append((link[2], desc))
            elif typ == driver.K['MSG_CHAIN_END']:
                by_source = collections.defaultdict(set)
                for identity, desc in state['resolved_this_chain']:
                    by_source[identity].add(desc)
                for identity, descs in by_source.items():
                    if len(descs) >= 2:
                        state['same_location_multi_effect_chains'][identity] += 1
                state['resolved_this_chain'].clear()
                state['chain_links'].clear()
            return super()._handle(msg)
    return WatchedDuel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--banks', default='0-3')
    ap.add_argument('--games', type=int, default=5)
    ap.add_argument('--seed0', type=int, default=7400)
    ap.add_argument('--main', type=int, default=400)
    ap.add_argument('--extra', type=int, default=80)
    ap.add_argument('--out', default='/tmp/ygo-multi-trigger-gen')
    ap.add_argument('--no-focus', action='store_true', help='use fully random main decks')
    args = ap.parse_args()
    step, legal = jax.jit(core.step), jax.jit(core.legal)
    db = driver.CardDB()
    total = identical = 0
    aggregate = collections.Counter()
    for bank_id in bank_range(args.banks):
        cset = cardgen.Gen(bank_id).make_set(args.main, args.extra)
        eligible = {c['code']: multi_trigger_indices(c) for c in cset
                    if len(multi_trigger_indices(c)) >= 2 and not cardgen.is_extra(c)}
        if not eligible:
            raise RuntimeError(f'bank {bank_id} contains no eligible main-deck cards')
        bank = cardgen.to_bank(cset)
        cmap = {c['code']: i for i, c in enumerate(cset)}
        compare.EFFECT_KINDS.clear()
        compare.EFFECT_KINDS.update({c['code']: [cardgen.bricks(e)[0] for e in c['effects']]
                                     for c in cset})
        db.add(tolua.cdb_rows(cset))
        script_dir = tolua.write(cset, os.path.join(args.out, str(bank_id)))
        driver.SCRIPTS[:] = [script_dir] + [d for d in driver.SCRIPTS
                                            if not d.startswith(args.out)]
        multi_descs = trigger_descs(cset)
        print(f'bank {bank_id}: {len(eligible)} eligible codes; '
              f'{sum(len(v) for v in eligible.values())} trigger effects', flush=True)
        for game in range(args.seed0, args.seed0 + args.games):
            rng = random.Random(1000 * bank_id + game)
            given = decks(cset, rng, focus_half=not args.no_focus)
            p0_counts = collections.Counter(given[0][0])
            in_decks = collections.Counter(code for main in given[0] for code in main)
            present = {code for code in eligible if in_decks[code]}
            state = dict(multi_descs=multi_descs, chain_links={}, resolved_this_chain=[],
                         native_activations=collections.Counter(),
                         native_resolutions=collections.Counter(),
                         same_location_multi_effect_chains=collections.Counter())
            original_duel = driver.Duel
            driver.Duel = watch_duel(state)
            started = time.time()
            try:
                result = compare.play(db, bank, cmap, game, step, legal,
                                      mode='gen', given=given)
            except Exception as exc:
                result = {'ok': False, 'err': f'{type(exc).__name__}: {exc}'}
            finally:
                driver.Duel = original_duel
            total += 1
            identical += int(result.get('ok', False))
            resolved_effects = {(code, desc) for (code, desc), count
                                in state['native_resolutions'].items()
                                if count and code in eligible and desc in multi_descs[code]}
            resolved_cards = {code for code, _ in resolved_effects}
            same_chain_sources = set(state['same_location_multi_effect_chains'])
            aggregate.update({
                'eligible_codes_in_bank': len(eligible),
                'eligible_codes_present_in_p0_main': sum(bool(p0_counts[code]) for code in eligible),
                'eligible_codes_present_in_both_main_decks': len(present),
                'eligible_card_copies_in_both_main_decks': sum(in_decks[code] for code in present),
                'eligible_codes_with_trigger_resolved': len(resolved_cards),
                'eligible_trigger_effect_resolutions': len(resolved_effects),
                'same_location_two_trigger_effects_resolved_in_one_chain': len(same_chain_sources),
            })
            print(f'bank {bank_id} game {game}: {time.time()-started:.1f}s '
                  f'ok={result.get("ok")} err={result.get("err", "")} '
                  f'p0-present={sum(bool(p0_counts[c]) for c in eligible)} '
                  f'both-present={len(present)} copies={sum(in_decks[c] for c in present)} '
                  f'multi-trigger-resolved-codes={len(resolved_cards)} '
                  f'resolved-trigger-effects={len(resolved_effects)} '
                  f'same-location-two-effects-same-chain={len(same_chain_sources)}', flush=True)
    print(f'{identical}/{total} games identical')
    games = max(total, 1)
    print('means per game:')
    for key, value in aggregate.items():
        print(f'  {key}: {value / games:.3f}')
    return 0 if total and identical == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
