"""Build logs/rerun_2026-10/manifest.json from processes.tsv, the job logs and the <label>.exit files.

Run from the worktree root with the experiment environment and WITHOUT thread variables set (as the runs):
    python logs/rerun_2026-10/make_manifest.py
"""
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import joblib
import numpy
import pandas
import scipy
import sklearn
import statsmodels
import threadpoolctl

L = Path("logs/rerun_2026-10")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def cpu_info():
    text = Path("/proc/cpuinfo").read_text()
    model = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
    cores = {(blk.get("physical id"), blk.get("core id"))
             for blk in (dict(re.findall(r"^(physical id|core id)\s*:\s*(\S+)", b, re.M)) for b in text.split("\n\n"))
             if blk}
    mem = re.search(r"^MemTotal:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
    return {"model": model.group(1) if model else platform.processor(), "physical_cores_seen_by_wsl": len(cores),
            "logical_cpus_seen_by_wsl": os.cpu_count(), "memory_total_gb_seen_by_wsl": round(int(mem.group(1)) / 2**20, 1),
            "note": ("values as seen inside the WSL2 VM; the host CPU (Intel Core i7-13650HX) has 14 cores "
                     "(6 performance + 8 efficiency) and 20 threads, of which WSL is configured to expose 16")}


def os_info():
    release = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', Path("/etc/os-release").read_text(), re.M))
    return {"distribution": release.get("PRETTY_NAME"), "kernel": platform.release(),
            "proc_version": Path("/proc/version").read_text().strip(), "platform": platform.platform()}


commits = [line.split("\t") for line in git("log", "--format=%H\t%cI").splitlines()]  # newest first


def blas_threads(cmd):
    """BLAS threads of a job, from its command line."""
    if "with_blas_threads.py 20" in cmd:
        main = "20 in the main process (with_blas_threads.py, threadpoolctl; March default)"
        return main + ("; 4 per DCV worker" if "OPENBLAS_NUM_THREADS=4" in cmd else "")
    if "OPENBLAS_NUM_THREADS=4" in cmd:
        return "OMP/OPENBLAS/MKL_NUM_THREADS=4 (4 per DCV worker, as in March; parent process also 4)"
    return "default (no thread variable set: 16 in the main process, 16 // 5 = 3 per DCV worker)"
jobs = []
for line in (L / "processes.tsv").read_text().splitlines():
    label, pid, start, log, cmd = line.split("\t")
    text = Path(log).read_text(errors="replace") if Path(log).exists() else ""
    m = re.search(r"Output:\s+(results/\S+)", text)
    exit_file = L / f"{label}.exit"
    status = dict(kv.split("=", 1) for kv in exit_file.read_text().split()) if exit_file.exists() else {}
    jobs.append({
        "label": label,
        "commit": next((h for h, d in commits if d <= start), None),
        "command": cmd,
        "cwd": str(Path.cwd()),
        "pid": int(pid),
        "log": log,
        "results_dir": m.group(1) if m else None,
        "start": start,
        "end": status.get("end"),
        "exit_code": int(status["exit"]) if "exit" in status else None,
        "blas_threads": blas_threads(cmd),
    })
# K-fold re-runs with 20 threads (<label>_t20) supersede the default-thread run once they finished without error;
# the 16-thread Algerian run, still in progress when its 20-thread run was launched, is kept for comparison only
# (stopped by decision on 27/10 09:40 and kept out of the tables). Jobs stopped to reduce CPU contention are
# relaunched as <label>_r<k> with the same command.
COMPARISON_ONLY = {"cv_alg_m2"}
for j in jobs:
    m = re.fullmatch(r"(.+)_t20(?:_r\d+)?", j["label"])
    if m:
        j["supersedes"] = m.group(1)
        old = next(o for o in jobs if o["label"] == j["supersedes"])
        if j["exit_code"] == 0 or old["label"] in COMPARISON_ONLY:
            old["superseded_by"] = j["label"]
        if old["label"] in COMPARISON_ONLY:
            old["role"] = "comparison only (16 threads), stopped by decision; the official run is " + j["label"]
for j in jobs:
    relaunch = [o["label"] for o in jobs if re.fullmatch(re.escape(j["label"]) + r"_r\d+", o["label"])]
    if j["exit_code"] not in (0, None) and j["label"].endswith("_t20"):
        j["role"] = (f"stopped by decision (exit {j['exit_code']}) to reduce CPU contention; " +
                     (f"relaunched as {relaunch[-1]}" if relaunch else "waiting in queue_t20.sh"))

# Runs made in the session scratchpad from `git archive` copies of the commit of the official run and copied here
# (folder name + "_<label>"): DCV of WILCAR-C with 4 BLAS threads per worker (as in March: 20 CPUs // 5 workers) and
# K-fold runs with 20 threads from the 20-thread verification; they supersede the default-thread runs, which stay
# in results/
rows = [line.split("\t") for line in (L / "promoted.tsv").read_text().splitlines()]
for p in (dict(zip(rows[0], r)) for r in rows[1:]):
    jobs.append({
        "label": p["label"],
        "commit": next(h for h, _ in commits if h.startswith(p["commit"])),
        "command": p["command"],
        "cwd": p["cwd"],
        "pid": None,
        "log": p["log"],
        "results_dir": p["results_dir"],
        "source_results_dir": p["source_results_dir"],
        "start": p["start"],
        "end": p["end"],
        "exit_code": int(p["exit"]),
        "blas_threads": blas_threads(p["command"]),
        "supersedes": p["supersedes"],
    })
    next(j for j in jobs if j["label"] == p["supersedes"])["superseded_by"] = p["label"]

# Re-runs of official jobs under other BLAS thread settings, compared bit for bit (results kept in the scratchpad)
rows = [line.split("\t") for line in (L / "verifications.tsv").read_text().splitlines()]
for v in (dict(zip(rows[0], r)) for r in rows[1:]):
    label = v.pop("verified_label")
    job = next(j for j in jobs if j["label"] == label)
    v["commit"] = next(h for h, _ in commits if h.startswith(v["commit"]))
    job.setdefault("verifications", []).append(v)

thread_vars = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
manifest = {
    "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "commit": git("rev-parse", "HEAD"),
    "commit_date": git("log", "-1", "--format=%ci"),
    "worktree_clean_for_code": git("status", "--porcelain", "--untracked-files=no", "--", "*.py", "src") == "",
    "environment": {
        "micromamba_env": "mestrado_ghabriel_pei",
        "python_executable": "/home/ghabr/micromamba/envs/mestrado_ghabriel_pei/bin/python",
        "versions": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__, "joblib": joblib.__version__,
                     "scikit-learn": sklearn.__version__, "pandas": pandas.__version__,
                     "threadpoolctl": threadpoolctl.__version__},
        "cpu": cpu_info(),
        "os": os_info(),
        "host": platform.node(),
        "thread_env_vars_at_launch": ("per job, see jobs[].blas_threads. Policy: K-fold with 20 threads in the main "
                                      "process (March default, via with_blas_threads.py); DCV of WILCAR-C with "
                                      "OMP/OPENBLAS/MKL_NUM_THREADS=4 (4 per worker, as in March); DCV of WILCAR "
                                      "with the default (identical to 4 per worker)"),
        "thread_env_vars_in_this_process": thread_vars,
        "blas_threadpool_info_default": threadpoolctl.threadpool_info(),
        "dcv_worker_threads": ("without thread variables, joblib/loky limits BLAS threads in each of the 5 DCV "
                               f"workers to cpu_count // n_jobs = {os.cpu_count()} // 5 = {os.cpu_count() // 5}. In "
                               "March WSL exposed the 20 logical CPUs of the host (no processors= in .wslconfig), "
                               "so the workers had 20 // 5 = 4 threads; the DCV M2 runs therefore set 4 explicitly. "
                               "K-fold processes use the OpenBLAS default (16 now, 20 in March)"),
    },
    "launcher": "logs/rerun_2026-10/launch.sh <label>... (each job: nohup bash -c '<command> > <log> 2>&1; "
                "echo exit=$? end=... > <label>.exit')",
    "jobs": jobs,
}
(L / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: v for k, v in manifest.items() if k != "jobs"}, indent=2))
print(f"{len(jobs)} jobs")
