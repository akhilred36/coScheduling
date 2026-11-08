#!/usr/bin/env python3
"""Generate socket-pinned isolated runs on Dane (8 nodes) to measure the socket-placement confound.

Every June 2026 isolated baseline ran on socket 0 (mpibind's default) with socket 1 idle. In
co-runs the socket each job lands on is a launch race. This campaign re-runs every isolated
application (and, optionally, the 9 panel inhibitors) pinned explicitly to socket 0 and to
socket 1, so socket effects can be separated from drift since June.

Design
------
* One sbatch per (app, replicate). It runs the socket-0 step and the socket-1 step back to back
  on the same 8 nodes. Even replicates run socket 0 first, odd replicates run socket 1 first, so
  order and node effects are balanced and both sockets share a submission window.
* The allocation and step shape are identical to generate_isolated_scripts_multinode.py
  (896 tasks / 112 per node / 240G; steps of 448 tasks / 56 per node / 119G, block:block).
* Pinning: mpibind is turned off for the measured steps (--whole --mpibind=off --cpu-bind=none). Each
  rank goes through bin/pin_to_socket.sh, which reads the topology from sysfs, picks the
  SLURM_LOCALID-th physical core of the requested socket (one PU per core, SMT sibling number
  SMT_INDEX), and execs `numactl --physcpubind=<pu> --localalloc` (falling back to taskset if
  numactl is missing). Both sockets use exactly the same mechanism.
  SMT_INDEX = 1 picks the same PUs June's mpibind used on socket 0 (112-167); socket 1 gets
  168-223.
* Verification: before each measured step, a probe step with the same srun options and the same
  wrapper prints each rank's actual Cpus_allowed_list, its package, NUMA node and memory policy.
  bin/check_placement.sh validates it: 448 ranks, 56 per node, one distinct PU each, all on the
  requested socket. If the check fails, the app step is skipped and a PLACEMENT_FAILED marker is
  written. Each job also records one mpibind reference probe (--mpibind=on,v), which shows what
  June's launcher would have chosen, plus lscpu and numactl --hardware.

Output layout (parsers work unchanged on each socket root; add a Socket column at parse time):
    <experiments_path>/<timestamp>/
        bin/                     pin_to_socket.sh, check_placement.sh
        slurm_scripts/           40 app jobs (submit these)
        slurm_scripts_inhib/     36 inhibitor jobs (generated only; lower priority)
        data/sock0/8_<app>_<r>/            mpip_profiles/, output.log, placement_*.{log,txt}
        data/sock1/8_<app>_<r>/
        data_inhib/sock{0,1}/8_inhib_<m>_<w>_d_<s>_<r>/
        job_logs/<job>/          slurm .out/.err, env.txt, mpibind_reference.log, steps.txt
        manifest.json

This script only writes files. It never submits jobs.
"""

import argparse
import json
import os
import shlex
import stat
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# User-modifiable variables
# =============================================================================
num_nodes = 8
redundant_runs = 4
app_walltime = "00:25:00"    # two ~400 s steps, plus probes and launch overhead
inhib_walltime = "00:20:00"  # two 360 s steps, plus probes
email = "aalasand1@unm.edu"
num_cpus = 56                # ranks per node per step (one socket)
mem = 240                    # GB per node, whole allocation
inhib_runtime = "360"        # seconds; same as the June isolated-inhibitor campaign
SMT_INDEX = 1                # 1 = June's PUs (112-167 on socket 0); 0 = PUs 0-55 / 56-111
SOCKETS = (0, 1)

# The 9 inhibitors of the full-data panel (from coScheduling-data-analyzer, 2026-11-07;
# also listed in scripts/dissertation_figs/output/batch2_panels.md). Strings are kept verbatim
# because the parser keys on them.
INHIB_PANEL = [
    ("2000000", "1000000", "0.3"),
    ("20000", "100000", "0.1"),
    ("200", "200000", "0.002237136"),
    ("2500000", "200000", "0.002237136"),
    ("2500000", "600000", "0.002237136"),
    ("400000", "1000000", "0.5"),
    ("40000", "1000000", "0.1"),
    ("40000", "1000000", "0.7"),
    ("500", "5000", "0.05"),
]

# =============================================================================
# Important variables, file paths, modules (same as the original isolated campaign)
# =============================================================================
modules = ["gcc/10", "openmpi", "cmake"]
spack_setup = "/g/g90/alasandagutt1/repos/spack/share/spack/setup-env.sh"
spack_env_dir = "/g/g90/alasandagutt1/spack_envs/beatnik/"
mpip_path = "/g/g90/alasandagutt1/spack_envs/beatnik/.spack-env/view/lib/libmpiP.so"
run_configs_path = "/g/g90/alasandagutt1/repos/coScheduling/run_configs/"
base_repo_path = "/g/g90/alasandagutt1/repos/coScheduling/"
inhib_exec_rel = "build/networkInhib"
experiments_path = "/p/lustre2/alasandagutt1/experiments_coscheduling_isolated_socket/"

STEP_TASKS = num_cpus * num_nodes
STEP_MEM = f"{mem // 2 - 1}G"
STEP_OPTS = (
    f"-n {STEP_TASKS} --nodes {num_nodes} --ntasks-per-node {num_cpus} --mem {STEP_MEM} "
    "--distribution=block:block"
)
# --whole: without it, Dane gives a 56-task/node step only socket 0's cores (pilot job 8518293:
# numactl rejected PUs 168-223 as outside the step cpuset). --whole hands the step every CPU of the
# job allocation on each node; the wrapper then picks the socket. Used for both sockets.
PINNED_OPTS = f"{STEP_OPTS} --whole --mpibind=off --cpu-bind=none"
# srun ignores --whole when -c is set, and `#SBATCH --cpus-per-task 1` exports it to every step
# (SLURM_CPUS_PER_TASK, plus SLURM_TRES_PER_TASK on Slurm 25.11; pilot job 2: "Ignoring --whole
# since -c/--cpus-per-task used"). Drop them for the pinned steps only; 1 CPU/task is the default.
PINNED_SRUN = "env -u SLURM_CPUS_PER_TASK -u SLURM_TRES_PER_TASK srun"
MPIBIND_REF_OPTS = f"{STEP_OPTS} --mpibind=on,v"

# =============================================================================
# Helper scripts written into the experiment directory
# =============================================================================
PIN_TO_SOCKET_SH = r"""#!/bin/bash
# Pin the calling Slurm task to one physical core of a socket, with local memory allocation.
#   pin_to_socket.sh <socket> <smt_index> probe
#   pin_to_socket.sh <socket> <smt_index> run <cmd> [args...]
# Core choice: the SLURM_LOCALID-th core (ascending PU number) of <socket>, using that core's
# <smt_index>-th hardware thread if the step cpuset allows it, else its first allowed sibling
# (logged as thread=). Topology comes from sysfs, so it doesn't depend on Slurm's CPU
# numbering. If PIN_LD_PRELOAD is set, it applies only to the final command (not to bash or
# numactl).
set -euo pipefail
sock=$1; smt=$2; mode=$3; shift 3
lid=${SLURM_LOCALID:?pin_to_socket.sh must run as a Slurm task}

expand_list() {  # "56,168" / "56-57" / "1-3,8" -> one PU per line
    local part a b i
    IFS=, read -ra parts <<<"$1"
    for part in "${parts[@]}"; do
        if [[ $part == *-* ]]; then a=${part%-*}; b=${part#*-}; for ((i=a; i<=b; i++)); do echo "$i"; done
        else echo "$part"; fi
    done
}

# PUs this task may use (the Slurm step cpuset). On Dane, an `srun --whole` step gets only
# SMT thread 0 of each core (0-111), while a default step got both threads of socket 0.
declare -A allowed=()
read -r _ cpus_allowed < <(grep '^Cpus_allowed_list' /proc/self/status)
while read -r c; do allowed[$c]=1; done < <(expand_list "$cpus_allowed")

# One entry per physical core on <socket>: "<first sibling> <chosen PU> <thread index>".
# The preferred thread is <smt_index>; if the cpuset doesn't contain it, take the core's first
# allowed sibling. Cores with no allowed sibling are skipped.
entries=()
for d in /sys/devices/system/cpu/cpu[0-9]*; do
    [[ -r $d/topology/physical_package_id ]] || continue
    cpu=${d##*/cpu}
    read -r pkg < "$d/topology/physical_package_id"
    [[ $pkg == "$sock" ]] || continue
    read -r sibs < "$d/topology/thread_siblings_list"
    mapfile -t sl < <(expand_list "$sibs")
    [[ ${sl[0]} == "$cpu" ]] || continue          # visit each core once
    choice=""; tidx=-1
    if (( smt < ${#sl[@]} )) && [[ -n ${allowed[${sl[$smt]}]:-} ]]; then
        choice=${sl[$smt]}; tidx=$smt
    else
        for i in "${!sl[@]}"; do [[ -n ${allowed[${sl[$i]}]:-} ]] && { choice=${sl[$i]}; tidx=$i; break; }; done
    fi
    [[ -n $choice ]] && entries+=("${sl[0]} $choice $tidx")
done
mapfile -t entries < <(printf '%s\n' "${entries[@]}" | sort -n -k1,1)
(( ${#entries[@]} > 0 && ${#entries[0]} > 0 )) || { echo "pin_to_socket: no allowed PUs on socket=$sock (cpuset $cpus_allowed) on $(hostname -s)" >&2; exit 97; }
(( lid < ${#entries[@]} )) || { echo "pin_to_socket: localid $lid >= ${#entries[@]} usable cores on socket $sock (cpuset $cpus_allowed)" >&2; exit 97; }
read -r _ pu thread <<<"${entries[$lid]}"
numa=-1
for f in /sys/devices/system/cpu/cpu"$pu"/node[0-9]*; do [[ -e $f ]] && numa=${f##*node}; done

if command -v numactl >/dev/null 2>&1; then
    bind=(numactl --physcpubind="$pu" --localalloc --); membind=localalloc
else
    bind=(taskset -c "$pu"); membind=kernel-default-local
fi
tag="PLACEMENT rank=${SLURM_PROCID} localid=$lid host=$(hostname -s) socket=$sock smt=$smt thread=$thread pu=$pu numa=$numa membind=$membind"

case $mode in
probe)
    exec "${bind[@]}" bash -c '
        a=$(awk "/^Cpus_allowed_list/{print \$2}" /proc/self/status)
        p=-1; f=/sys/devices/system/cpu/cpu$a/topology/physical_package_id
        [[ -r $f ]] && read -r p < "$f"
        if command -v numactl >/dev/null 2>&1; then pol=$(numactl --show | awk "/^policy/{print \$2}"); else pol=n/a; fi
        echo "$0 allowed=$a allowed_pkg=$p policy=$pol omp=${OMP_NUM_THREADS:-unset}"' "$tag" ;;
run)
    echo "$tag" >&2
    if [[ -n ${PIN_LD_PRELOAD:-} ]]; then exec "${bind[@]}" env LD_PRELOAD="$PIN_LD_PRELOAD" "$@"
    else exec "${bind[@]}" "$@"; fi ;;
*)
    echo "pin_to_socket: unknown mode $mode" >&2; exit 2 ;;
esac
"""

CHECK_PLACEMENT_SH = r"""#!/bin/bash
# Validate a probe log written by pin_to_socket.sh.
#   check_placement.sh <probe_log> <socket> <expected_ranks> <ranks_per_node>
# Exit 0 only if every rank is bound to exactly one PU, that PU is the intended one and lies on
# <socket>, no PU is shared within a node, and each node has <ranks_per_node> ranks.
set -euo pipefail
log=$1; sock=$2; want=$3; per_node=$4
awk -v sock="$sock" -v want="$want" -v per_node="$per_node" '
    /^PLACEMENT / {
        delete kv
        for (i = 2; i <= NF; i++) { split($i, p, "="); kv[p[1]] = p[2] }
        n++
        if (kv["socket"] != sock)          { bad++; print "wrong socket arg: " $0 }
        if (kv["allowed"] != kv["pu"])     { bad++; print "mask != intended PU: " $0 }
        if (kv["allowed_pkg"] != sock)     { bad++; print "PU not on socket " sock ": " $0 }
        key = kv["host"] ":" kv["allowed"]
        if (key in seen)                   { bad++; print "duplicate PU: " $0 }
        seen[key] = 1; host[kv["host"]]++
        if (min == "" || kv["pu"] + 0 < min) min = kv["pu"] + 0
        if (kv["pu"] + 0 > max) max = kv["pu"] + 0
        numa[kv["numa"]] = 1; pol[kv["policy"]] = 1; omp[kv["omp"]] = 1
    }
    END {
        if (n + 0 != want) { bad++; print "rank count " n + 0 " != " want }
        nh = 0; for (h in host) { nh++; if (host[h] != per_node) { bad++; print "node " h " has " host[h] " ranks" } }
        ns = ""; for (k in numa) ns = ns k " "
        ps = ""; for (k in pol) ps = ps k " "
        os = ""; for (k in omp) os = os k " "
        printf "ranks=%d nodes=%d socket=%s pu_range=%s-%s numa_nodes=[ %s] mem_policy=[ %s] omp=[ %s] problems=%d\n", n, nh, sock, min, max, ns, ps, os, bad + 0
        exit (bad > 0)
    }' "$log"
"""


# =============================================================================
# Generation
# =============================================================================
def format_args(args_dict):
    """Same argument formatting as the original isolated generator."""
    out = ""
    for key, value in args_dict.items():
        out += f" {value}" if key == "" else f" {key} {value}"
    return out


def socket_order(r):
    return (0, 1) if r % 2 == 0 else (1, 0)


def slurm_script(job_name, walltime, job_log_dir, bin_dir, steps):
    """steps: list of (socket, run_dir, command string) in execution order."""
    pin = bin_dir / "pin_to_socket.sh"
    check = bin_dir / "check_placement.sh"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name {job_name}",
        f"#SBATCH --mail-user {email}",
        "#SBATCH --mail-type FAIL,TIME_LIMIT",
        f"#SBATCH --output {job_log_dir}/{job_name}_%j.out",
        f"#SBATCH --error {job_log_dir}/{job_name}_%j.err",
        f"#SBATCH --ntasks {num_cpus * 2 * num_nodes}",
        f"#SBATCH --ntasks-per-node {num_cpus * 2}",
        "#SBATCH --cpus-per-task 1",
        f"#SBATCH --nodes {num_nodes}",
        f"#SBATCH --mem {mem}G",
        f"#SBATCH --time {walltime}",
        "#SBATCH --partition pbatch",
        "#SBATCH --distribution block:cyclic",
        "",
    ]
    lines += [f"module load {m}" for m in modules]
    lines += [
        f"source {spack_setup}",
        f"spack env activate {spack_env_dir}",
        "",
        f"JOBLOG={job_log_dir}",
        f"PIN={pin}",
        f"CHECK={check}",
        f"SMT={SMT_INDEX}",
        "status=0",
        "",
        "# Environment and topology record",
        "{ date -Is; echo \"job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST\"; scontrol show hostnames \"$SLURM_JOB_NODELIST\";",
        "  srun --version; grep Cpus_allowed_list /proc/self/status;",
        "  module list 2>&1; lscpu; numactl --hardware 2>&1 || echo 'numactl not found'; } > \"$JOBLOG/env.txt\" 2>&1",
        "",
        "# Reference: where June's launcher (mpibind) would put these ranks. Not a measurement.",
        f"srun {MPIBIND_REF_OPTS} bash -c 'echo \"MPIBIND_REF rank=$SLURM_PROCID host=$(hostname -s) "
        "allowed=$(awk \"/^Cpus_allowed_list/{print \\$2}\" /proc/self/status) "
        "omp=${OMP_NUM_THREADS:-unset} places=${OMP_PLACES:-unset}\"' > \"$JOBLOG/mpibind_reference.log\" 2>&1",
        "",
        "run_on_socket() {  # <socket> <run_dir> <cmd...>",
        "    local sock=$1 dir=$2; shift 2",
        f"    {PINNED_SRUN} {PINNED_OPTS} \"$PIN\" \"$sock\" \"$SMT\" probe > \"$dir/placement_probe.log\" 2>&1",
        f"    if ! \"$CHECK\" \"$dir/placement_probe.log\" \"$sock\" {STEP_TASKS} {num_cpus} > \"$dir/placement_check.txt\" 2>&1; then",
        "        echo \"placement check FAILED for socket $sock ($dir); skipping step\" >&2",
        "        touch \"$dir/PLACEMENT_FAILED\"; status=1; return",
        "    fi",
        "    echo \"socket=$sock start=$(date +%s.%N) dir=$dir\" >> \"$JOBLOG/steps.txt\"",
        f"    time {PINNED_SRUN} {PINNED_OPTS} env MPIP=\"-f $dir/mpip_profiles\" OMP_NUM_THREADS=1 OMP_PLACES=threads PIN_LD_PRELOAD=\"{mpip_path}\" \\",
        "        \"$PIN\" \"$sock\" \"$SMT\" run \"$@\" > \"$dir/output.log\" 2>&1 || status=1",
        "    echo \"socket=$sock end=$(date +%s.%N) rc=$status dir=$dir\" >> \"$JOBLOG/steps.txt\"",
        "}",
        "",
    ]
    for sock, run_dir, cmd in steps:
        lines.append(f"run_on_socket {sock} {run_dir} {cmd}")
    lines += ["", "exit $status", ""]
    return "\n".join(lines)


def write_exec(path, content):
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="Write the experiment here, not under experiments_path (for local dry runs)")
    ap.add_argument("--configs", default=run_configs_path, help="Directory holding <N>_nodes.json")
    ap.add_argument("--no-inhib", action="store_true", help="Skip the optional inhibitor scripts")
    opts = ap.parse_args()

    on_dane = Path(base_repo_path).is_dir()
    if not on_dane and not opts.root:
        sys.exit(f"{base_repo_path} not found; this doesn't look like Dane. Pass --root for a dry run.")

    with open(Path(opts.configs) / f"{num_nodes}_nodes.json") as f:
        apps = json.load(f)["apps"]

    root = Path(opts.root or experiments_path)
    ts_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    bin_dir = ts_dir / "bin"
    for d in (bin_dir, ts_dir / "slurm_scripts", ts_dir / "job_logs"):
        d.mkdir(parents=True)
    write_exec(bin_dir / "pin_to_socket.sh", PIN_TO_SOCKET_SH)
    write_exec(bin_dir / "check_placement.sh", CHECK_PLACEMENT_SH)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "socket-pinned isolated runs (socket-placement confound)",
        "smt_index": SMT_INDEX,
        "pinned_step_options": PINNED_OPTS,
        "mpibind_reference_options": MPIBIND_REF_OPTS,
        "jobs": [],
    }
    missing = []

    def add_job(kind, job_name, walltime, data_root, run_name, cmd, r):
        job_log_dir = ts_dir / "job_logs" / job_name
        job_log_dir.mkdir(parents=True)
        steps = []
        for sock in socket_order(r):
            run_dir = data_root / f"sock{sock}" / run_name
            (run_dir / "mpip_profiles").mkdir(parents=True)
            steps.append((sock, run_dir, cmd))
        scripts_dir = ts_dir / ("slurm_scripts" if kind == "app" else "slurm_scripts_inhib")
        scripts_dir.mkdir(exist_ok=True)
        (scripts_dir / f"{job_name}.slurm").write_text(
            slurm_script(job_name, walltime, job_log_dir, bin_dir, steps))
        manifest["jobs"].append({
            "kind": kind, "job": job_name, "run": run_name, "replicate": r,
            "socket_order": list(socket_order(r)), "command": cmd,
            "run_dirs": {f"sock{s}": str(d) for s, d, _ in steps},
        })

    # Applications: one job per (app, replicate)
    for app, info in apps.items():
        exe = os.path.join(base_repo_path, info["path"], info["exec"])
        if on_dane and not os.path.isfile(exe):
            missing.append(exe)
        cmd = shlex.quote(exe) + format_args(info["args"])
        for r in range(redundant_runs):
            add_job("app", f"{num_nodes}_{app}_{r}", app_walltime, ts_dir / "data",
                    f"{num_nodes}_{app}_{r}", cmd, r)

    # Optional: panel inhibitors (generated, not meant for the first submission)
    if not opts.no_inhib:
        inhib = os.path.join(base_repo_path, inhib_exec_rel)
        if on_dane and not os.path.isfile(inhib):
            missing.append(inhib)
        for m, w, s in INHIB_PANEL:
            name = f"{num_nodes}_inhib_{m}_{w}_d_{s}"
            cmd = f"{inhib} -m {m} -w {w} -c d -s {s} -r {inhib_runtime}"
            for r in range(redundant_runs):
                add_job("inhib", f"{name}_{r}", inhib_walltime, ts_dir / "data_inhib", f"{name}_{r}", cmd, r)

    (ts_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    n_app = sum(j["kind"] == "app" for j in manifest["jobs"])
    n_inh = len(manifest["jobs"]) - n_app
    print(f"Experiment directory: {ts_dir}")
    print(f"App jobs:   {n_app} ({2 * n_app} pinned runs)  -> {ts_dir / 'slurm_scripts'}")
    print(f"Inhib jobs: {n_inh} ({2 * n_inh} pinned runs)  -> {ts_dir / 'slurm_scripts_inhib'} (not for first submission)")
    if missing:
        print("WARNING: executables not found:\n  " + "\n  ".join(missing))
    print(f"Submit (apps): scripts/submit_slurm_scripts.sh {ts_dir / 'slurm_scripts'}")


if __name__ == "__main__":
    main()
