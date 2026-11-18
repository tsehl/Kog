"""Run a GPU computation of the repository on Modal (GPUs rented on demand).

The container clones the public repository at the requested commit (it must be pushed), then runs the command in the
repository. Persistent volume `kog-data`, mounted on /data:
- /data/jax-cache: JAX compilation cache (YGO_JAX_CACHE), reused from one launch to the next;
- /data/runs/NAME: log (log.txt) and everything the command writes to $OUT (checkpoints: --save $OUT/run).
The volume is synchronized every 60 s during the computation, so an interrupted run keeps its latest checkpoints.

Usage (from the repository root, with the Modal client installed and authenticated):
  modal run --detach cloud/modal_run.py \\
      --name test --gpu L4 --commit HEAD --timeout 30 --cmd 'python bench/engine_bench.py engine --compile-only'
  modal volume get kog-data runs/test/log.txt -     (read the log)
  modal app list / modal app stop -y ID                (follow / stop)
By default 8 CPU cores and 32 GB: Modal gives only one core otherwise, and XLA compilation runs on the CPU.
--commit HEAD: the current local commit (must be on origin). --timeout in minutes (hard cutoff on the Modal side)."""
import os
import shlex
import subprocess
import threading
import time

import modal

REPO = "https://github.com/tsehl/Kog.git"
JAX = "0.11.1"                     # version of the published measurements (comparable cache and measurements)

app = modal.App("kog")
vol = modal.Volume.from_name("kog-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git")
         .pip_install(f"jax[cuda12]=={JAX}", "optax", "numpy"))


@app.function(image=image, volumes={"/data": vol}, gpu="L4", cpu=8, memory=32768, timeout=24 * 3600)
def job(commit: str, cmd: str, name: str):
    out = f"/data/runs/{name}"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", REPO, "/repo"], check=True)
    subprocess.run(["git", "-C", "/repo", "checkout", "--quiet", commit], check=True)
    env = dict(os.environ, YGO_JAX_CACHE="/data/jax-cache", OUT=out, PYTHONUNBUFFERED="1")
    stop = threading.Event()

    def sync():                    # make the log and checkpoints visible outside the container during the computation
        while not stop.wait(60):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("volume commit:", e, flush=True)

    th = threading.Thread(target=sync, daemon=True)
    th.start()
    t0 = time.time()
    with open(f"{out}/log.txt", "a") as log:
        log.write(f"== commit {commit} ; {cmd}\n")
        log.flush()
        p = subprocess.Popen(["bash", "-c", cmd], cwd="/repo", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        for line in p.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        rc = p.wait()
        log.write(f"== end: code {rc}, {time.time() - t0:.0f} s\n")
    stop.set()
    vol.commit()
    return rc


@app.local_entrypoint()
def main(cmd: str, name: str, commit: str = "HEAD", gpu: str = "L4", timeout: int = 60, cpu: int = 8, mem: int = 32):
    if commit == "HEAD":
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    on_origin = subprocess.run(["git", "branch", "-r", "--contains", commit], capture_output=True, text=True).stdout.strip()
    assert on_origin, f"commit {commit} is not on origin: push it first"
    print(f"{name}: {gpu}, {cpu} cores, {mem} GB, commit {commit[:9]}, {timeout} min max ; {shlex.quote(cmd)}", flush=True)
    rc = job.with_options(gpu=gpu, cpu=cpu, memory=mem * 1024, timeout=timeout * 60).remote(commit, cmd, name)
    print(f"{name}: code {rc}")
