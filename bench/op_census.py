"""Census of the ops executed by step_batch, per engine source line, from a CPU profiler trace.

Usage: python bench/op_census.py TRACE_DIR HLO_FILE [--calls 2] [--top 30]
TRACE_DIR: engine_bench.py --profile DIR (its B<b> subdirectory); HLO_FILE: engine_bench.py --hlo FILE (same run).
On TPU, the time floor of step_batch is the number of small ops executed per call (each fusion is a kernel launch):
the CPU trace records every executed op (thunk), so the count per call is measured locally, without a TPU, and
attributed to the engine line that produced it (metadata of the compiled HLO). The CPU fusions differ from the TPU
ones: the counts are a proxy, to compare two versions of the engine, not a TPU prediction.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("hlo")
ap.add_argument("--calls", type=int, default=2, help="number of step_batch calls in the trace (--profile-calls)")
ap.add_argument("--top", type=int, default=30)
a = ap.parse_args()

path = a.trace
if os.path.isdir(path):
    found = sorted(glob.glob(os.path.join(path, "**", "*.trace.json.gz"), recursive=True))
    found = [f for f in found if "perfetto" not in f]
    assert found, f"no *.trace.json.gz under {path}"
    path = found[-1]

# stack tables of the HLO text (FileNames, FunctionNames, FileLocations, StackFrames; parent_frame_id is the parent's
# id + 1, 0: none), then instruction name -> innermost engine frame and chain of engine functions
tables, cur = collections.defaultdict(dict), None
src = {}
pat = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*\S+\s+([\w\-]+)\(")
for line in open(a.hlo):
    if line.rstrip() in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
        cur = line.strip()
        continue
    if cur and re.match(r"^\d+ ", line):
        k, v = line.split(" ", 1)
        tables[cur][int(k)] = v.strip()
        continue
    cur = None
    m = pat.match(line)
    if m:
        f = re.search(r"stack_frame_id=(\d+)", line)
        src[m.group(1)] = int(f.group(1)) if f else 0
files = {k: v.strip('"') for k, v in tables["FileNames"].items()}
funcs = {k: v.strip('"') for k, v in tables["FunctionNames"].items()}
locs = {k: dict(re.findall(r"(\w+)=(\d+)", v)) for k, v in tables["FileLocations"].items()}
frames = {k: dict(re.findall(r"(\w+)=(\d+)", v)) for k, v in tables["StackFrames"].items()}


def chain(fid):  # engine frames, innermost first: (function, line)
    out = []
    while fid:
        fr = frames[fid]
        lo = locs[int(fr["file_location_id"])]
        if files[int(lo["file_name_id"])].endswith("engine/core.py"):
            out.append((funcs[int(lo["function_name_id"])], int(lo["line"])))
        fid = int(fr["parent_frame_id"]) - 1
    return out


chains = {}
for name, fid in src.items():
    if fid not in chains:
        chains[fid] = chain(fid)
    src[name] = chains[fid]

ev = json.load(gzip.open(path))["traceEvents"]
skip = re.compile(r"(\$|end: |ThreadpoolListener|Wait |ThunkExecutor|ProcessNode|PjRt|Thunk)")
ops = [e for e in ev if e.get("ph") == "X" and not skip.match(e["name"])]
known = [e for e in ops if e["name"] in src]
n = len(known)
print(f"{path}\n{n / a.calls:,.0f} ops per call ({len(ops) - n} trace events not in the HLO ignored), "
      f"{sum(e['dur'] for e in known) / 1000 / a.calls:.0f} ms of op time per call (all threads)")

line, incl, idur, kind = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
for e in known:
    ch = src[e["name"]]
    line[f"core.py:{ch[0][1]} {ch[0][0]}" if ch else "?"] += 1
    for fn in {f for f, _ in ch}:  # inclusive: every engine function on the stack
        incl[fn] += 1
        idur[fn] += e["dur"]
    kind[re.sub(r"[.\d]+$", "", e["name"])] += 1
c = a.calls
print("\nby op kind (count per call):")
for k, v in kind.most_common(a.top):
    print(f"  {v / c:9,.0f}  {k}")
print("\nby innermost engine line (count per call):")
for k, v in line.most_common(a.top):
    print(f"  {v / c:9,.0f}  {k}")
print("\nby engine function, inclusive (count per call, op time ms per call):")
for k, v in incl.most_common(2 * a.top):
    print(f"  {v / c:9,.0f}  {idur[k] / 1000 / c:7.1f} ms  {k}")
