# Kog

**Kog** ("King of Games") is a Yu-Gi-Oh! rules engine written in pure JAX, plus a PPO self-play agent trained on it.
The engine is vectorised with `vmap`, so thousands of duels run in parallel on one GPU.

The agent never sees a real card during training:
- cards and decks are generated procedurally;
- every card effect is decomposed into atomic *bricks* (cost, target, filter, move, modify, trigger, ...), and the
  network reads these bricks directly.

The model therefore has to understand effect trees rather than memorise card identifiers. Real cards are only used as
a held-out benchmark.

**The goal is the complete game. The engine is not there yet.** The core rules and every summoning mechanic are in
place and checked against the reference engine. However, several card mechanics are still missing, and only 7.4% of
the official cards can be expressed fully today.

## Where it stands

### Rules engine

Implemented, and checked against the reference engine:
- turn structure and battle, including damage-step windows and attack replays;
- chains:
  - chain building and resolution;
  - SEGOC;
  - trigger effects and quick effects;
  - "once per turn" limits;
- continuous effects:
  - ATK/DEF layers (set, set final, set base, modifiers in effect order);
  - locks and restrictions;
  - protections and immunities;
  - negation;
- equip spells, costs, targeting;
- Extra Deck summoning procedures:
  - Fusion and Ritual;
  - Synchro and Xyz (with material attachment);
  - Link (with arrows and co-links, including Extra Link);
  - Pendulum.

Not implemented yet:
- field spells;
- card counters (spell counters and the like);
- tokens;
- change of control;
- position changes by card effects;
- Gemini, Union, Spirit and Toon monsters.

### Verification against the reference engine

Every rule is checked decision by decision against an oracle. The oracle is
[ygopro-core](https://github.com/edo9300/ygopro-core), the C++ core used by EDOPro, driven through its C API. In each
game, both engines must offer the same legal actions and reach the same state after every decision. This includes the
identity of each card copy.

| check | games | identical to the oracle |
|---|---:|---:|
| regression suite, 18 rule areas (`engine/regress.sh`) | 1,640 | 1,640 |
| procedural cards compiled to Lua, banks 0-9 (`gen/compare_gen.py`) | 100 | 100 |

The batched engine (`step_batch`) is also checked against the per-game engine decision by decision (`vmap_check.py`,
`sb_check.py`), on a fixed test bank and on a procedural bank.

The 18 rule areas of the regression suite:
- triggers, quick effects and chains;
- continuous effects, the damage step, locks and negation;
- Fusion, Ritual, Synchro, Xyz, Link, Extra Link and Pendulum;
- mixed and plain games.

### Card translator

`translate/` reads the official card scripts (Lua) symbolically and turns them into bricks. A card counts as
translated only when every one of its effects is.

| card type | scripts | fully translated | effects translated |
|---|---:|---:|---:|
| monster (Main Deck) | 6,389 | 512 (8.0%) | 20.3% |
| Extra Deck monster | 2,163 | 107 (4.9%) | 39.2% |
| spell | 2,844 | 283 (10.0%) | 27.4% |
| trap | 2,059 | 104 (5.1%) | 17.0% |
| other | 86 | 1 | 17.1% |
| **total** | **13,541** | **1,007 (7.4%)** | **24.6%** |

How these numbers are counted:
- The translator cannot start reading 2,353 of the scripts at all, so the effects of those cards are left out of the
  last column.
- Among the remaining unfinished cards, 4,789 have a single effect left untranslated.
- `translate/progress.py` regenerates this report and lists the refusal reasons, ranked by the number of cards they
  block.

Coverage by card family, translated / total:

| family | translated |
|---|---:|
| Fusion monsters | 61 / 571 |
| Xyz monsters | 22 / 595 |
| Synchro monsters | 20 / 534 |
| Link monsters | 4 / 463 |
| Pendulum monsters | 8 / 390 |
| equip spells | 38 / 282 |
| continuous traps | 12 / 563 |
| counter traps | 2 / 176 |
| field spells | 0 / 332 |

### Throughput

Measured on one NVIDIA L4 GPU in mid-November 2026, with a random policy and a 15-card Extra Deck
(`bench/engine_bench.py --extra 15`):

| games in parallel | policy decisions / s |
|---:|---:|
| 1,024 | ~10,000 |
| 4,096 | ~15,000 |

Full PPO training reaches ~3,700 decisions per second. That figure counts the rollout and the update, at 4,096 games
with 16-step rollouts.

These GPU figures were measured on the engine as it was before the TPU work below. The current version, rewritten for
TPU, has not been re-measured on GPU.

### TPU port (in progress)

The engine was first written and tuned for GPU. On one TPU v5e chip it ran 27 times slower than on an L4. We are
rewriting it for TPU. All figures below come from one v5e chip (Colab), with 1,024 games in parallel, a 15-card Extra
Deck and a random policy. One `step_batch` call advances every game to its next decision, about 1,000 decisions per
call.

| engine | ms per `step_batch` call | decisions / s |
|---|---:|---:|
| GPU version, unchanged | ~2,500 | 364 |
| dense histograms, per-slot card attributes, one-hot state reads | 323 | 3,082 |
| + dense `nonzero` | **212** | **4,702** |
| *for comparison: same engine before the TPU work, on one L4 GPU* | *~100* | *~10,000* |

The engine computes exactly the same games as before: every change kept passes the full oracle regression (1,640 /
1,640 games).

**Why it was slow.** A profile showed the TPU was 100% busy, with plenty of free memory:
- 1.3 s per call went into scatters and 1.0 s into gathers;
- ordinary computation took about 0.1 s.

A rules engine makes many small indexed accesses: "the parent card of this material", "rank of this card in its zone",
"attribute of card h". On a GPU, thousands of threads absorb them. On a TPU, XLA turns them into nearly serial loops.

**Method.** Every indexed access in the engine is over a small domain: 110 card slots per duel, 16–32 list entries, 4
registers. So each access can be rewritten as an exact dense computation. We compare the index to every position (a
one-hot mask), then select or reduce. This is more arithmetic, but of the kind the TPU's vector units run at full
speed. Concretely:
- histograms and ranks became pairwise comparisons;
- card attributes, which were read from the card bank 1,200 times per call, are now copied into the state once per
  game;
- `jnp.nonzero(size=k)`, which JAX implements as a scatter of every element, has a dense replacement.

**What did not work.**
- *One-hot reads inside the per-candidate test loop.* The loop that checks which effects can be activated runs every
  instruction for 24 candidates per game, and computes every branch of the opcode switch. There, a one-hot read of a
  single value over 110 slots costs more than the gather it replaces: 212 → 240 ms. That change was reverted.
- *More games per chip.* 4,096 games take 3.6 times as long as 1,024, so the throughput barely moves. The chip is
  limited by the amount of work per game, not by the number of operations.
- *Imposing a memory layout* (game axis last): no measurable effect.

**Next steps.** At 212 ms per call, the remaining time splits as follows:

| part | ms per call |
|---|---:|
| gathers | 107 |
| ordinary computation | 55 |
| data reformatting | 25 |
| small scatters | 23 |

The work planned next:
- **Activation tests.** The per-candidate activation test evaluates every opcode branch on [games × 24 × 110] arrays,
  at every instruction. We will move loop-invariant filters and reads out of the loop, and compile only the branches
  of opcodes present in the card bank. This test was already more than half of the GPU time, so the change should
  help on both GPU and TPU.
- **Remaining sorts and scalar writes, outside the candidate loops.** These rewrites are already validated against
  the oracle and still have to be measured on TPU.

The target is to at least match the L4 per chip, i.e. at most 100 ms per call. Each chip plays its own games, so the
throughput then adds up across the chips of a pod slice.

### Agent

**Architecture.**
- A small transformer encodes each card from its brick tokens: header, one token per effect, per instruction and per
  filter.
- A second transformer reads the visible game state: one token per card, with its zone, position, current stats and
  status. It outputs the policy and the value.

**Training.**
- PPO self-play on procedural cards and decks.
- A new card bank is generated every 10 iterations.
- 20% of the games are played against past policies and 20% against a scripted heuristic opponent.

**Evaluation.** Each deck pairing is played in both seats, on real-card decks that are never used in training:
- *reference*: four decks of hand-written real cards (control, Synchro, Ritual/Fusion, Pendulum);
- *archetype*: four synergy decks (Constellar, Mermail, Speedroid, Prophecy) built only from cards produced by the
  translator.

| player | vs random, reference | vs heuristic, reference | vs heuristic, archetype |
|---|---:|---:|---:|
| scripted heuristic | 87.9% | — | — |
| agent, 120 iterations | 98.4% | 73.0% | 66.2% |
| agent, 360 iterations (`train/ckpt/run5_it360_f16.pkl.gz`) | 100% | **83.9%** | **79.8%** |

On the archetype decks, the heuristic beats random 92.7% of the time.

## Layout

| path | content |
|---|---|
| `engine/` | rules engine (`core.py`), card bricks (`cards.py`), observation encoding, oracle comparison and consistency checks |
| `oracle/` | ctypes driver for ygopro-core, rule test scripts, build notes |
| `gen/` | procedural card and deck generator, Lua compiler for generated cards, fuzzing |
| `translate/` | translator from official Lua scripts to bricks, validation against the oracle |
| `train/` | network (`model.py`), PPO self-play (`ppo.py`), real-card benchmark banks, evaluation and behaviour probes |
| `bench/` | throughput and profiling tools |
| `cloud/` | launcher for GPU jobs on Modal |
| `tests/` | unit tests and oracle cross-checks |

## Setup

```bash
./setup.sh
```

`setup.sh` fetches and builds the external dependencies at pinned commits:
- ygopro-core (the oracle);
- [CardScripts](https://github.com/ProjectIgnis/CardScripts), the official card scripts;
- [BabelCDB](https://github.com/ProjectIgnis/BabelCDB), the card database.

These projects are under their own licences and are not redistributed here. Scripts are launched with
[uv](https://github.com/astral-sh/uv).

## Usage

```bash
# unit tests (CPU, ~6 min)
uv run --with jax --with luaparser --with optax python -m unittest discover -s tests

# full regression against the oracle, then the batching consistency checks (CPU, a few hours)
bash engine/regress.sh /tmp/regress

# engine throughput
uv run --with 'jax[cuda12]' python bench/engine_bench.py engine --extra 15 --b 1024,4096

# training with the configuration of the 360-iteration agent (one L4, about 2 hours)
uv run --with 'jax[cuda12]' --with optax python train/ppo.py --b 1024 --steps 64 --mb 8 --bf16 \
    --mix-heur 0.2 --mix-old 0.2 --iters 360 --eval-every 40 --old-lag 120 --save runs/ckpt

# evaluation on the reference and archetype decks
uv run --with 'jax[cuda12]' --with optax python train/real_compare.py train/ckpt/run5_it360_f16.pkl.gz --b 1024 --bf16

# translator coverage report
uv run --with luaparser --with jax python translate/progress.py --jobs 4
```

Two derived data files ship with the repository, so that training and evaluation do not need the external
dependencies:
- `gen/skeletons.pkl.gz` holds effect structures taken from the translated scripts, with all names and values
  stripped. Procedural cards can be drawn from these structures.
- `train/realbank*.npz` are the real-card benchmark banks.

Both can be rebuilt with `gen/skeleton.py` and `train/realcards.py`.

## Notes

Developed with the help of AI coding assistants.

Yu-Gi-Oh! is a trademark of Konami. This project is not affiliated with or endorsed by Konami.

## License

MIT, see `LICENSE`.
