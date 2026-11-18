"""Summary of a JAX profiler trace of step_batch calls (engine_bench.py --profile DIR).

Usage: python bench/trace_summary.py DIR_OR_TRACE_JSON_GZ [--top 20]
Prints, for the device ops of the trace:
- wall time of the step_batch modules and the busy fraction of the device;
- exclusive time per HLO category (custom fusions are the scatters and gathers on TPU);
- scatters and gathers by number of indices per game (a gather of 110 indices per game reads all N slots), and by
  source line of the engine when the trace carries it.
Written for TPU traces (where scatters and gathers appear as custom fusions); on GPU the categories differ.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--batch", default="1024,256,4096", help="batch sizes used to express index counts per game")
a = ap.parse_args()

path = a.path
if os.path.isdir(path):
    found = sorted(glob.glob(os.path.join(path, "**", "*.trace.json.gz"), recursive=True))
    assert found, f"no *.trace.json.gz under {path}"
    path = found[-1]
ev = json.load(gzip.open(path))["traceEvents"]
threads = {}
for e in ev:
    if e.get("ph") == "M" and e.get("name") == "thread_name":
        threads[(e["pid"], e["tid"])] = e["args"]["name"]
on = lambda e, name: e.get("ph") == "X" and threads.get((e["pid"], e["tid"])) == name
mods = [e for e in ev if on(e, "XLA Modules")]
ops = sorted([e for e in ev if on(e, "XLA Ops")], key=lambda e: (e["ts"], -e["dur"]))
assert ops, "no device ops in the trace"

# the step_batch module is the longest one
by_mod = collections.Counter()
for m in mods:
    by_mod[m["name"].split("(")[0]] += m["dur"]
main = by_mod.most_common(1)[0][0]
calls = [m for m in mods if m["name"].startswith(main)]
wall = sum(m["dur"] for m in calls)
nc = len(calls)
print(f"{path}\nmodule {main}: {nc} calls, {wall / 1000 / nc:.1f} ms per call")

# exclusive time (ops are nested: while / conditional contain their bodies)
excl, cat_t, cat_n = collections.Counter(), collections.Counter(), collections.Counter()
stack = []
for e in ops:
    s, t = e["ts"], e["ts"] + e["dur"]
    while stack and stack[-1][1] <= s:
        stack.pop()
    c = e.get("args", {}).get("hlo_category", "?")
    if stack:
        excl[stack[-1][0]] -= e["dur"]
        cat_t[stack[-1][2]] -= e["dur"]
    excl[id(e)] += e["dur"]
    cat_t[c] += e["dur"]
    cat_n[c] += 1
    stack.append((id(e), t, c))
busy = sum(cat_t.values())
print(f"device busy {100 * busy / max(sum(m['dur'] for m in mods), 1):.0f}% of the module time; "
      f"{len(ops) / nc:.0f} ops per call, {busy / len(ops):.2f} us per op on average\n")
print("exclusive time by HLO category (per call):")
for c, v in cat_t.most_common(12):
    print(f"  {c:26s} {v / 1000 / nc:9.1f} ms {cat_n[c] / nc:9.0f} ops {v / max(cat_n[c], 1):8.2f} us/op")


def elems(shape):
    p = 1
    for x in shape.split(","):
        if x:
            p *= int(x)
    return p


batches = [int(x) for x in a.batch.split(",")]


def per_game(n):
    for b in batches:
        if n % b == 0:
            return n // b
    return None


for kind in ("scatter", "gather"):
    sel = [e for e in ops if e["name"].startswith(kind)]
    if not sel:
        continue
    by_n, by_src, cnt_n = collections.Counter(), collections.Counter(), collections.Counter()
    for e in sel:
        ln = e["args"].get("long_name", "")
        body = ln.split("fusion(", 1)[1] if "fusion(" in ln else ""
        ins = [elems(x) for x in re.findall(r"\w+\[([\d,]*)\]\{", body)]
        # operand order: scatter_custom_fusion (indices, updates, init); scatter_fusion (operand, indices, updates);
        # gather_fusion (operand, indices)
        if kind == "scatter":
            idx = ins[0] if e["name"].startswith("scatter_custom") else (ins[1] if len(ins) > 1 else 0)
        else:
            idx = ins[1] if len(ins) > 1 else 0
        pg = per_game(idx)
        key = f"{pg} per game" if pg else f"{idx} in total"
        by_n[key] += e["dur"]
        cnt_n[key] += 1
        src = e["args"].get("source", "?")
        by_src[re.sub(r"^.*/(engine/[\w.]+:\d+)$", r"\1", src)] += e["dur"]
    tot = sum(by_n.values())
    print(f"\n{kind}s: {tot / 1000 / nc:.1f} ms per call, {len(sel) / nc:.0f} per call")
    for k, v in by_n.most_common(8):
        print(f"  indices {k:16s} {v / 1000 / nc:8.1f} ms {cnt_n[k] / nc:7.0f} ops")
    print("  by source line:")
    for k, v in by_src.most_common(a.top):
        print(f"    {v / 1000 / nc:8.1f} ms  {k}")
