"""Measurement run on a Kaggle TPU v5e-8 (free, counted in session hours of the weekly TPU quota).

Locally:   python cloud/kaggle_tpu.py push --commit SHA [--repo URL] [--timeout 9000] [--only parity,bench1]
           kaggle kernels status USER/kog-tpu ; kaggle kernels logs USER/kog-tpu ;
           kaggle kernels output USER/kog-tpu -p OUT_DIR       (logs of each step and summary.txt)
The commit must be pushed to the (public) repository: the Kaggle session clones it, installs JAX for TPU, then runs
the steps one after the other, each with its own time limit, and writes STEP.log and summary.txt to /kaggle/working.
Needs the Kaggle CLI, authenticated (~/.kaggle/access_token or kaggle.json), on an account allowed to use TPUs
(Kaggle asks for phone and identity verification first).

Steps (all on the same machine, so the numbers are comparable):
- parity: bench/tpu_parity.py, the TPU computes exactly like the CPU (where the oracle checks run);
- bench1: engine_bench.py on one chip, B=1024, Extra Deck 15 (same setting as the Colab v5e-1 measurements);
- bench8: the same on the 8 chips, 1024 games per chip (shard_map, no communication between chips);
- bench8p: the same as 8 independent processes, one per chip, measuring at the same time (barrier); fallback if the
  shard_map compilation fails (it segfaults in the XLA CPU compiler, untested on TPU);
- ppo: a few PPO iterations on one chip (the whole training loop runs on TPU; time per iteration).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

REPO = "https://github.com/tsehl/Kog.git"      # replaced by push
COMMIT = ""                                     # replaced by push
ONLY = ""                                       # replaced by push (empty: every step)
JAX = "0.11.2"                                  # version of the Colab TPU measurements
KERNEL = "kog-tpu"

STEPS = [
    ("parity", "python bench/tpu_parity.py engine --b 64 --calls 80", 2400),
    ("bench1", "python bench/engine_bench.py engine --extra 15 --b 1024 --calls 30", 2400),
    ("bench8p", None, 3600),
    ("bench8", "python bench/engine_bench.py engine --extra 15 --b 1024 --calls 30 --devices 8", 2400),
    ("ppo", "python train/ppo.py --b 1024 --steps 32 --iters 6 --precompile 3 --bf16 --eval-every 0 --bench-every 0", 4800),
]


def sh(cmd, log=None, timeout=None, cwd=None):
    """Runs cmd, copies its output to stdout (visible with `kaggle kernels logs`) and to log; returns (code, last lines)."""
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    f = open(log, "w") if log else None
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    tail, killed = [], []
    if timeout:  # watchdog: also kills a step that hangs without printing anything
        timer = threading.Timer(timeout, lambda: (killed.append(1), os.killpg(p.pid, 9)))
        timer.start()
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if f:
            f.write(line)
            f.flush()
        tail = (tail + [line.rstrip()])[-8:]
    code = p.wait()
    if timeout:
        timer.cancel()
    if killed:
        code = "timeout"
        tail.append(f"KILLED after {timeout} s")
    if f:
        f.close()
    print(f"[{code}] {time.time() - t0:.0f} s", flush=True)
    return code, tail


def multiproc(src, out, limit, n=8, calls=100, b=1024):
    """n engine_bench processes, one per chip (TPU_VISIBLE_CHIPS), measuring at the same time: total = sum."""
    import re
    bar = os.path.join(tempfile.gettempdir(), "barrier")
    shutil.rmtree(bar, ignore_errors=True)
    procs = []
    for i in range(n):
        env = dict(os.environ, TPU_VISIBLE_CHIPS=str(i), TPU_VISIBLE_DEVICES=str(i), TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1",
                   TPU_PROCESS_BOUNDS="1,1,1", TPU_PROCESS_PORT=str(8476 + i))
        log = open(os.path.join(out, f"bench8p.{i}.log"), "w")
        cmd = f"python bench/engine_bench.py engine --extra 15 --b {b} --calls {calls} --barrier {bar}:{n}"
        procs.append((subprocess.Popen(cmd, shell=True, cwd=src, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True), log))
    t0 = time.time()
    while any(p.poll() is None for p, _ in procs) and time.time() - t0 < limit:
        if any(p.poll() not in (None, 0) for p, _ in procs):  # one failed: the others would wait at the barrier
            break
        time.sleep(5)
    for p, log in procs:
        if p.poll() is None:
            os.killpg(p.pid, 9)
        p.wait()
        log.close()
    rate, win, lines = [], [], []
    for i in range(n):
        txt = open(os.path.join(out, f"bench8p.{i}.log")).read()
        m = re.search(r"([\d,]+) decisions/s ; ([\d.]+) ms/call.*measured from ([\d.]+) to ([\d.]+)", txt)
        head = re.search(r"jax .* visible", txt)
        lines.append(f"chip {i}: " + (head.group(0) if head else "?") + " | " + (m.group(0) if m else "no measurement"))
        if m:
            rate.append(float(m.group(1).replace(",", "")))
            win.append((float(m.group(3)), float(m.group(4))))
    if len(rate) == n:
        s, e = max(w[0] for w in win), min(w[1] for w in win)
        span = sum(w[1] - w[0] for w in win) / n
        lines.append(f"total {sum(rate):,.0f} decisions/s on {n} chips ({sum(rate) / n:,.0f} per chip); measurement "
                     f"windows overlap {100 * max(e - s, 0) / span:.0f}% (100% = all chips measured at the same time)")
    else:
        lines.append(f"only {len(rate)} / {n} processes measured")
    print("\n".join(lines), flush=True)
    return ("ok" if len(rate) == n else "failed"), lines


def on_kaggle():
    out = "/kaggle/working"
    src = os.path.join(tempfile.gettempdir(), "kog")
    summary = [f"repo {REPO} commit {COMMIT}"]
    code, _ = sh(f"git clone -q {REPO} {src} && cd {src} && git checkout -q {COMMIT} && git log --oneline -1")
    assert code == 0, "clone failed"
    sh(f'pip install -q "jax[tpu]=={JAX}" optax 2>&1 | tail -3')
    _, env = sh('python -c "import jax; print(jax.__version__, jax.devices())"; nproc; free -g | head -2')
    summary += env
    for name, cmd, limit in STEPS:
        if ONLY and name not in ONLY.split(","):
            continue
        t0 = time.time()
        if cmd is None:
            code, tail = multiproc(src, out, limit)
            cmd = "8 processes of engine_bench.py, one per chip"
        else:
            code, tail = sh(cmd, os.path.join(out, f"{name}.log"), limit, cwd=src)
        summary += ["", f"== {name} (exit {code}, {time.time() - t0:.0f} s): {cmd}"] + tail
        with open(os.path.join(out, "summary.txt"), "w") as f:  # rewritten after each step: kept if the session dies
            f.write("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


def push(a):
    cfg = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
    user = a.user or next((ln.split(":", 1)[1].strip() for ln in cfg.splitlines() if "username:" in ln), "")
    assert user, "Kaggle user name unknown: --user"
    d = tempfile.mkdtemp()
    code = open(__file__).read()
    code = code.replace('COMMIT = ""', f'COMMIT = "{a.commit}"', 1).replace(f'REPO = "{REPO}"', f'REPO = "{a.repo}"', 1)
    code = code.replace('ONLY = ""', f'ONLY = "{a.only}"', 1)
    open(os.path.join(d, "job.py"), "w").write(code)
    json.dump({"id": f"{user}/{KERNEL}", "title": KERNEL, "code_file": "job.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": True,
               "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(d, "kernel-metadata.json"), "w"))
    r = subprocess.run(["kaggle", "kernels", "push", "-p", d, "--accelerator", "TpuV5E8", "-t", str(a.timeout)])
    shutil.rmtree(d)
    sys.exit(r.returncode)


if __name__ == "__main__":
    if os.path.isdir("/kaggle/working") and len(sys.argv) == 1:
        on_kaggle()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["push"])
        ap.add_argument("--commit", required=True, help="commit to measure (pushed to --repo)")
        ap.add_argument("--repo", default=REPO)
        ap.add_argument("--user", default="", help="Kaggle user name (default: kaggle config view)")
        ap.add_argument("--timeout", type=int, default=9000, help="hard limit of the Kaggle session, in seconds")
        ap.add_argument("--only", default="", help="comma-separated steps (default: all)")
        push(ap.parse_args())
