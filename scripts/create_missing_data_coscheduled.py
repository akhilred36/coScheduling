#!/usr/bin/env python3
"""Create explicit eight-node inhibitor co-scheduled Dane experiments.

This generator only creates experiment directories, manifests, and individual
Slurm scripts. It never submits jobs. Run one work package at a time on Dane:

    python3 /g/g90/alasandagutt1/repos/coScheduling/scripts/create_missing_data_coscheduled.py \
        --work-package first-wave
    python3 /g/g90/alasandagutt1/repos/coScheduling/scripts/create_missing_data_coscheduled.py \
        --work-package expanded

The first-wave package contains the eight profile/fanout configurations and the
five Barrier-only configurations. The expanded package contains the remaining
six profile/fanout configurations. A generated case is single-use; create a new
timestamped package rather than resubmitting a case after any failed attempt.
"""

import argparse
import hashlib
import json
import shlex
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path


REPOSITORY = Path("/g/g90/alasandagutt1/repos/coScheduling")
LUSTRE_ROOT = Path("/p/lustre2/alasandagutt1")
EXPERIMENTS_DIR = LUSTRE_ROOT / "missing_data_coscheduled"

APP_CONFIG_PATH = REPOSITORY / "run_configs/8_nodes.json"
INHIBITOR_SOURCE = REPOSITORY / "networkInhibitor/networkInhib.cpp"
INHIBITOR_EXECUTABLE = REPOSITORY / "build/networkInhib"
SPACK_SETUP = Path("/g/g90/alasandagutt1/repos/spack/share/spack/setup-env.sh")
SPACK_ENV = Path("/g/g90/alasandagutt1/spack_envs/beatnik")
MPIP_LIBRARY = SPACK_ENV / ".spack-env/view/lib/libmpiP.so"

APPLICATIONS = (
    "amg",
    "beatnik",
    "fiesta",
    "kripke",
    "laghos",
    "lammps",
    "minife",
    "minivite",
    "quicksilver",
    "tricount",
)

INHIBITOR_RUNTIME_SECONDS = {
    "amg": 917,
    "beatnik": 1055,
    "fiesta": 958,
    "kripke": 947,
    "laghos": 889,
    "lammps": 947,
    "minife": 1031,
    "minivite": 989,
    "quicksilver": 798,
    "tricount": 1005,
}

NUM_NODES = 8
ALLOCATION_TASKS = 896
ALLOCATION_TASKS_PER_NODE = 112
ALLOCATION_MEMORY = "240G"
WALLTIME = "00:25:00"
STEP_TASKS = 448
STEP_TASKS_PER_NODE = 56
STEP_MEMORY = "119G"
REPETITIONS = 4
MAIL_USER = "aalasand1@unm.edu"
MODULES = ("gcc/10", "openmpi", "cmake")

# After isolated calibration, change only final_wait_us when the one permitted
# correction changes a wait. initial_wait_us must remain the documented pilot.
FIRST_WAVE_PROFILE_CONFIGURATIONS = (
    {
        "priority": 1,
        "alias": "AMG-45",
        "message_bytes": 9472,
        "initial_wait_us": 62000,
        "final_wait_us": 62000,
        "peers_per_rank": 45,
        "sparsity": "0.100671141",
        "target": "AMG",
        "family": "profile_fanout",
    },
    {
        "priority": 2,
        "alias": "LAG-6",
        "message_bytes": 791,
        "initial_wait_us": 634,
        "final_wait_us": 634,
        "peers_per_rank": 6,
        "sparsity": "0.013422819",
        "target": "Laghos",
        "family": "profile_fanout",
    },
    {
        "priority": 3,
        "alias": "BEA-22",
        "message_bytes": 546432,
        "initial_wait_us": 139472,
        "final_wait_us": 139472,
        "peers_per_rank": 22,
        "sparsity": "0.049217002",
        "target": "Beatnik",
        "family": "profile_fanout",
    },
    {
        "priority": 4,
        "alias": "QS-1",
        "message_bytes": 210,
        "initial_wait_us": 172833,
        "final_wait_us": 172833,
        "peers_per_rank": 1,
        "sparsity": "0.002237136",
        "target": "QuickSilver",
        "family": "profile_fanout",
    },
    {
        "priority": 5,
        "alias": "TRI-1",
        "message_bytes": 1432391,
        "initial_wait_us": 16897,
        "final_wait_us": 16897,
        "peers_per_rank": 1,
        "sparsity": "0.002237136",
        "target": "TriCount",
        "family": "profile_fanout",
    },
    {
        "priority": 6,
        "alias": "KRI-1",
        "message_bytes": 1281840,
        "initial_wait_us": 518406,
        "final_wait_us": 518406,
        "peers_per_rank": 1,
        "sparsity": "0.002237136",
        "target": "Kripke",
        "family": "profile_fanout",
    },
    {
        "priority": 7,
        "alias": "LAG-22-C",
        "message_bytes": 791,
        "initial_wait_us": 3453,
        "final_wait_us": 3453,
        "peers_per_rank": 22,
        "sparsity": "0.049217002",
        "target": "Laghos control",
        "family": "profile_fanout",
    },
    {
        "priority": 8,
        "alias": "AMG-16-C",
        "message_bytes": 9472,
        "initial_wait_us": 21798,
        "final_wait_us": 21798,
        "peers_per_rank": 16,
        "sparsity": "0.035794183",
        "target": "AMG control",
        "family": "profile_fanout",
    },
)

BARRIER_CONFIGURATIONS = (
    {
        "priority": None,
        "alias": "BAR-0",
        "message_bytes": 1,
        "initial_wait_us": 0,
        "final_wait_us": 0,
        "peers_per_rank": 0,
        "sparsity": "0.0",
        "target": None,
        "family": "barrier_only",
    },
    {
        "priority": None,
        "alias": "BAR-100",
        "message_bytes": 1,
        "initial_wait_us": 100,
        "final_wait_us": 100,
        "peers_per_rank": 0,
        "sparsity": "0.0",
        "target": None,
        "family": "barrier_only",
    },
    {
        "priority": None,
        "alias": "BAR-500",
        "message_bytes": 1,
        "initial_wait_us": 500,
        "final_wait_us": 500,
        "peers_per_rank": 0,
        "sparsity": "0.0",
        "target": None,
        "family": "barrier_only",
    },
    {
        "priority": None,
        "alias": "BAR-1000",
        "message_bytes": 1,
        "initial_wait_us": 1000,
        "final_wait_us": 1000,
        "peers_per_rank": 0,
        "sparsity": "0.0",
        "target": None,
        "family": "barrier_only",
    },
    {
        "priority": None,
        "alias": "BAR-5000",
        "message_bytes": 1,
        "initial_wait_us": 5000,
        "final_wait_us": 5000,
        "peers_per_rank": 0,
        "sparsity": "0.0",
        "target": None,
        "family": "barrier_only",
    },
)

EXPANDED_CONFIGURATIONS = (
    {
        "priority": 9,
        "alias": "LAM-1",
        "message_bytes": 1465695,
        "initial_wait_us": 340423,
        "final_wait_us": 340423,
        "peers_per_rank": 1,
        "sparsity": "0.002237136",
        "target": "LAMMPS",
        "family": "profile_fanout",
    },
    {
        "priority": 10,
        "alias": "FIE-1",
        "message_bytes": 2542163,
        "initial_wait_us": 237894,
        "final_wait_us": 237894,
        "peers_per_rank": 1,
        "sparsity": "0.002237136",
        "target": "Fiesta",
        "family": "profile_fanout",
    },
    {
        "priority": 11,
        "alias": "MV-8",
        "message_bytes": 47103,
        "initial_wait_us": 52834,
        "final_wait_us": 52834,
        "peers_per_rank": 8,
        "sparsity": "0.017897092",
        "target": "MiniVite",
        "family": "profile_fanout",
    },
    {
        "priority": 12,
        "alias": "MF-8",
        "message_bytes": 56312,
        "initial_wait_us": 202304,
        "final_wait_us": 202304,
        "peers_per_rank": 8,
        "sparsity": "0.017897092",
        "target": "MiniFE",
        "family": "profile_fanout",
    },
    {
        "priority": 13,
        "alias": "BEA-45-C",
        "message_bytes": 546432,
        "initial_wait_us": 286019,
        "final_wait_us": 286019,
        "peers_per_rank": 45,
        "sparsity": "0.100671141",
        "target": "Beatnik control",
        "family": "profile_fanout",
    },
    {
        "priority": 14,
        "alias": "TRI-134-C",
        "message_bytes": 1432391,
        "initial_wait_us": 2198321,
        "final_wait_us": 2198321,
        "peers_per_rank": 134,
        "sparsity": "0.299776286",
        "target": "TriCount control",
        "family": "profile_fanout",
    },
)

WORK_PACKAGES = {
    "first-wave": FIRST_WAVE_PROFILE_CONFIGURATIONS + BARRIER_CONFIGURATIONS,
    "expanded": EXPANDED_CONFIGURATIONS,
}

EXPECTED_CONFIGURATION_COUNTS = {"first-wave": 13, "expanded": 6}
EXPECTED_CASE_COUNTS = {"first-wave": 520, "expanded": 240}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create explicit eight-node inhibitor co-scheduled Slurm scripts "
            "on Dane without submitting them."
        )
    )
    parser.add_argument(
        "--work-package",
        choices=tuple(WORK_PACKAGES),
        required=True,
        help="Generate either the 13-configuration first wave or six-configuration expanded wave.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_record(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def require_file(path, description):
    if not path.is_file():
        raise RuntimeError(f"Missing {description}: {path}")


def validate_dane_environment():
    if not LUSTRE_ROOT.is_dir():
        raise RuntimeError(
            f"Dane Lustre root is unavailable: {LUSTRE_ROOT}. "
            "Run this generator on Dane."
        )

    require_file(APP_CONFIG_PATH, "eight-node application configuration")
    require_file(INHIBITOR_SOURCE, "network inhibitor source")
    require_file(INHIBITOR_EXECUTABLE, "network inhibitor executable")
    require_file(SPACK_SETUP, "Spack setup script")
    require_file(MPIP_LIBRARY, "mpiP library")
    if not SPACK_ENV.is_dir():
        raise RuntimeError(f"Missing Spack environment: {SPACK_ENV}")


def load_applications():
    with APP_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        apps = json.load(stream)["apps"]

    if set(apps) != set(APPLICATIONS):
        missing = sorted(set(APPLICATIONS) - set(apps))
        unexpected = sorted(set(apps) - set(APPLICATIONS))
        raise ValueError(
            f"Unexpected application roster; missing={missing}, unexpected={unexpected}"
        )

    for app_name in APPLICATIONS:
        app = apps[app_name]
        executable = REPOSITORY / app["path"] / app["exec"]
        require_file(executable, f"{app_name} executable")
        app["executable_path"] = str(executable)

    return apps


def cpp_round_peer_count(sparsity):
    value = Decimal(sparsity) * Decimal(STEP_TASKS - 1)
    return int((value + Decimal("0.5")).to_integral_value(rounding=ROUND_FLOOR))


def application_argv(app):
    argv = [app["executable_path"]]
    for key, value in app["args"].items():
        if key:
            argv.extend(shlex.split(str(key)))
        if str(value):
            argv.extend(shlex.split(str(value)))
    return argv


def validate_configurations(work_package, configurations):
    expected_count = EXPECTED_CONFIGURATION_COUNTS[work_package]
    if len(configurations) != expected_count:
        raise AssertionError(
            f"{work_package} has {len(configurations)} configurations; expected {expected_count}"
        )

    aliases = set()
    inhibitor_ids = set()
    for configuration in configurations:
        alias = configuration["alias"]
        if alias in aliases:
            raise AssertionError(f"Duplicate candidate alias: {alias}")
        aliases.add(alias)

        if configuration["message_bytes"] <= 0:
            raise ValueError(f"Message size must be positive for {alias}")
        if configuration["final_wait_us"] < 0:
            raise ValueError(f"Wait must be nonnegative for {alias}")

        sparsity = configuration["sparsity"]
        if not Decimal("0") <= Decimal(sparsity) <= Decimal("1"):
            raise ValueError(f"Sparsity must be in [0, 1] for {alias}")

        actual_peers = cpp_round_peer_count(sparsity)
        expected_peers = configuration["peers_per_rank"]
        if actual_peers != expected_peers:
            raise AssertionError(
                f"{alias}: round({sparsity} * 447)={actual_peers}, expected {expected_peers}"
            )

        inhibitor_id = (
            f"8_{configuration['message_bytes']}_{configuration['final_wait_us']}_{sparsity}"
        )
        if inhibitor_id in inhibitor_ids:
            raise AssertionError(f"Duplicate inhibitor ID: {inhibitor_id}")
        inhibitor_ids.add(inhibitor_id)


def build_cases(work_package, configurations, apps, data_root, scripts_root):
    cases = []
    keys = set()

    for configuration in configurations:
        message = configuration["message_bytes"]
        wait = configuration["final_wait_us"]
        sparsity = configuration["sparsity"]
        inhibitor_id = f"8_{message}_{wait}_{sparsity}"

        for app_name in APPLICATIONS:
            app_argv = application_argv(apps[app_name])
            runtime = INHIBITOR_RUNTIME_SECONDS[app_name]

            for repetition in range(REPETITIONS):
                key = (app_name, message, wait, sparsity, repetition)
                if key in keys:
                    raise AssertionError(f"Duplicate generated case key: {key}")
                keys.add(key)

                experiment_name = (
                    f"8_{app_name}_inhib_{message}_{wait}_d_{sparsity}_{repetition}"
                )
                data_dir = data_root / experiment_name
                mpip_dir = data_dir / "mpip_profiles"
                slurm_script = scripts_root / f"{experiment_name}.slurm"
                slurm_checksum = scripts_root / f"{experiment_name}.slurm.sha256"
                inhibitor_arguments = {
                    "-m": str(message),
                    "-w": str(wait),
                    "-c": "d",
                    "-s": sparsity,
                    "-r": str(runtime),
                    "-o": str(data_dir / "inhib_stats.json"),
                }

                cases.append(
                    {
                        "experiment_name": experiment_name,
                        "work_package": work_package,
                        "family": configuration["family"],
                        "priority": configuration["priority"],
                        "candidate_alias": configuration["alias"],
                        "target": configuration["target"],
                        "application": app_name,
                        "application_argv": app_argv,
                        "application_command": shlex.join(app_argv),
                        "inhibitor_id": inhibitor_id,
                        "inhibitor_arguments": inhibitor_arguments,
                        "message_bytes": message,
                        "initial_wait_us": configuration["initial_wait_us"],
                        "final_wait_us": wait,
                        "communication_mode": "d",
                        "sparsity": sparsity,
                        "peers_per_rank": configuration["peers_per_rank"],
                        "inhibitor_runtime_seconds": runtime,
                        "repetition": repetition,
                        "data_directory": str(data_dir),
                        "mpip_profile_directory": str(mpip_dir),
                        "slurm_script": str(slurm_script),
                        "slurm_checksum": str(slurm_checksum),
                        "expected_outputs": [
                            str(data_dir / "app_output.log"),
                            str(data_dir / "inhib_output.log"),
                            str(data_dir / "inhib_stats.json"),
                            str(mpip_dir),
                            str(data_dir / "run_status.json"),
                        ],
                    }
                )

    expected_count = EXPECTED_CASE_COUNTS[work_package]
    if len(cases) != expected_count:
        raise AssertionError(
            f"{work_package} generated {len(cases)} cases; expected {expected_count}"
        )
    if len(keys) != len(cases):
        raise AssertionError("Generated case keys are not unique")

    return cases


def build_hashes(apps):
    generator_path = Path(__file__).resolve()
    source_hashes = {
        "generator": hash_record(generator_path),
        "network_inhibitor_source": hash_record(INHIBITOR_SOURCE),
        "spack_setup": hash_record(SPACK_SETUP),
    }
    config_hashes = {"eight_node_app_config": hash_record(APP_CONFIG_PATH)}
    executable_hashes = {
        "network_inhibitor": hash_record(INHIBITOR_EXECUTABLE),
        "mpip_library": hash_record(MPIP_LIBRARY),
        "applications": {
            app_name: hash_record(Path(apps[app_name]["executable_path"]))
            for app_name in APPLICATIONS
        },
    }
    return source_hashes, config_hashes, executable_hashes


def build_manifest(
    created_at,
    work_package,
    timestamp_dir,
    cases,
    source_hashes,
    config_hashes,
    executable_hashes,
):
    return {
        "schema_version": 1,
        "created_at_utc": created_at,
        "work_package": work_package,
        "experiment_directory": str(EXPERIMENTS_DIR),
        "timestamp_directory": str(timestamp_dir),
        "configuration_count": EXPECTED_CONFIGURATION_COUNTS[work_package],
        "application_count": len(APPLICATIONS),
        "repetitions": REPETITIONS,
        "planned_case_count": EXPECTED_CASE_COUNTS[work_package],
        "applications": list(APPLICATIONS),
        "resource_contract": {
            "nodes": NUM_NODES,
            "allocation_tasks": ALLOCATION_TASKS,
            "allocation_tasks_per_node": ALLOCATION_TASKS_PER_NODE,
            "allocation_memory": ALLOCATION_MEMORY,
            "walltime": WALLTIME,
            "step_tasks": STEP_TASKS,
            "step_tasks_per_node": STEP_TASKS_PER_NODE,
            "step_memory": STEP_MEMORY,
            "distribution": "block:block",
            "mpi_binding": "--mpibind=on,v",
        },
        "source_hashes": source_hashes,
        "config_hashes": config_hashes,
        "executable_hashes": executable_hashes,
        "cases": cases,
    }


def generate_slurm_script(
    case,
    manifest_path,
    manifest_sha256,
    source_hashes,
    config_hashes,
    executable_hashes,
):
    experiment_name = case["experiment_name"]
    data_dir = Path(case["data_directory"])
    mpip_dir = Path(case["mpip_profile_directory"])
    app_log = data_dir / "app_output.log"
    inhib_log = data_dir / "inhib_output.log"
    inhib_stats = data_dir / "inhib_stats.json"
    run_status = data_dir / "run_status.json"
    slurm_script = Path(case["slurm_script"])
    slurm_checksum = Path(case["slurm_checksum"])

    inhibitor_argv = [str(INHIBITOR_EXECUTABLE)]
    for key, value in case["inhibitor_arguments"].items():
        inhibitor_argv.extend((key, value))

    srun_prefix = (
        f"srun --exclusive -n {STEP_TASKS} --mem {STEP_MEMORY} "
        f"--nodes {NUM_NODES} --ntasks-per-node {STEP_TASKS_PER_NODE} "
        "--distribution=block:block --mpibind=on,v"
    )
    mpip_environment = (
        f"env LD_PRELOAD={shlex.quote(str(MPIP_LIBRARY))} "
        f"MPIP={shlex.quote(f'-f {mpip_dir}')}"
    )

    relevant_executable_hashes = {
        "network_inhibitor": executable_hashes["network_inhibitor"],
        "mpip_library": executable_hashes["mpip_library"],
        "application": executable_hashes["applications"][case["application"]],
    }

    artifact_checks = (
        (source_hashes["generator"], "generator"),
        (source_hashes["network_inhibitor_source"], "network inhibitor source"),
        (source_hashes["spack_setup"], "Spack setup"),
        (config_hashes["eight_node_app_config"], "eight-node app config"),
        (relevant_executable_hashes["network_inhibitor"], "network inhibitor"),
        (relevant_executable_hashes["mpip_library"], "mpiP library"),
        (relevant_executable_hashes["application"], "application executable"),
    )

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name {experiment_name}",
        f"#SBATCH --mail-user {MAIL_USER}",
        "#SBATCH --mail-type FAIL,TIME_LIMIT",
        f"#SBATCH --output {data_dir / (experiment_name + '_%j.out')}",
        f"#SBATCH --error {data_dir / (experiment_name + '_%j.err')}",
        f"#SBATCH --ntasks {ALLOCATION_TASKS}",
        f"#SBATCH --ntasks-per-node {ALLOCATION_TASKS_PER_NODE}",
        "#SBATCH --cpus-per-task 1",
        f"#SBATCH --nodes {NUM_NODES}",
        f"#SBATCH --mem {ALLOCATION_MEMORY}",
        f"#SBATCH --time {WALLTIME}",
        "#SBATCH --partition pbatch",
        "#SBATCH --distribution block:block",
        "",
        "set -euo pipefail",
        "umask 027",
        "",
        "if [[ -z \"${SLURM_JOB_ID:-}\" || -z \"${SLURM_JOB_NODELIST:-}\" ]]; then",
        "    echo \"Refusing to run outside a Slurm batch allocation.\" >&2",
        "    exit 2",
        "fi",
        (
            f"if [[ \"${{SLURM_JOB_NUM_NODES:-0}}\" -ne {NUM_NODES} || "
            f"\"${{SLURM_NTASKS:-0}}\" -ne {ALLOCATION_TASKS} || "
            f"\"${{SLURM_JOB_PARTITION:-}}\" != pbatch || "
            f"\"${{SLURM_JOB_NAME:-}}\" != {experiment_name} ]]; then"
        ),
        (
            "    echo \"Allocation does not match the required "
            f"{NUM_NODES}-node/{ALLOCATION_TASKS}-task contract.\" >&2"
        ),
        "    exit 2",
        "fi",
        "",
        f"module load {' '.join(MODULES)}",
        f"source {shlex.quote(str(SPACK_SETUP))}",
        f"spack env activate {shlex.quote(str(SPACK_ENV))}",
        "",
        f"DATA_DIR={shlex.quote(str(data_dir))}",
        f"MPIP_PROFILE_DIR={shlex.quote(str(mpip_dir))}",
        f"APP_LOG={shlex.quote(str(app_log))}",
        f"INHIB_LOG={shlex.quote(str(inhib_log))}",
        f"INHIB_STATS={shlex.quote(str(inhib_stats))}",
        f"RUN_STATUS_PATH={shlex.quote(str(run_status))}",
        f"SLURM_SCRIPT_PATH={shlex.quote(str(slurm_script))}",
        f"SLURM_CHECKSUM_PATH={shlex.quote(str(slurm_checksum))}",
        "ATTEMPT_LOCK=\"$DATA_DIR/.attempt_started\"",
        "mkdir -p \"$MPIP_PROFILE_DIR\"",
        "",
        "if ! mkdir \"$ATTEMPT_LOCK\"; then",
        "    echo \"This case already has an attempt lock; refusing duplicate execution.\" >&2",
        "    exit 3",
        "fi",
        "printf '%s\\n' \"$SLURM_JOB_ID\" > \"$ATTEMPT_LOCK/slurm_job_id\"",
        "",
        (
            "if [[ -e \"$APP_LOG\" || -e \"$INHIB_LOG\" || "
            "-e \"$INHIB_STATS\" || -e \"$RUN_STATUS_PATH\" ]] || "
            "compgen -G \"$MPIP_PROFILE_DIR/*\" >/dev/null; then"
        ),
        "    echo \"Refusing to overwrite outputs from an earlier attempt.\" >&2",
        "    exit 3",
        "fi",
        "",
        "timestamp_utc() {",
        "    date -u +\"%Y-%m-%dT%H:%M:%SZ\"",
        "}",
        "",
        "verify_sha256() {",
        "    local path=\"$1\"",
        "    local expected=\"$2\"",
        "    local description=\"$3\"",
        "    local actual",
        "    actual=\"$(sha256sum \"$path\" | cut -d ' ' -f 1)\"",
        "    if [[ \"$actual\" != \"$expected\" ]]; then",
        "        echo \"SHA-256 mismatch for $description: $path\" >&2",
        "        return 1",
        "    fi",
        "}",
        "",
        f"verify_sha256 {shlex.quote(str(manifest_path))} {manifest_sha256} 'case manifest'",
        *(
            f"verify_sha256 {shlex.quote(record['path'])} {record['sha256']} "
            f"{shlex.quote(description)}"
            for record, description in artifact_checks
        ),
        "",
        "HOSTNAME_LIST_JSON=\"[\"",
        "HOST_SEPARATOR=\"\"",
        "HOSTNAME_OUTPUT=\"$(scontrol show hostnames \"${SLURM_JOB_NODELIST}\")\"",
        "while IFS= read -r HOST; do",
        "    HOSTNAME_LIST_JSON=\"${HOSTNAME_LIST_JSON}${HOST_SEPARATOR}\\\"${HOST}\\\"\"",
        "    HOST_SEPARATOR=\",\"",
        "done <<< \"$HOSTNAME_OUTPUT\"",
        "HOSTNAME_LIST_JSON=\"${HOSTNAME_LIST_JSON}]\"",
        "EXECUTED_SLURM_SCRIPT_PATH=\"$(readlink -f \"$0\")\"",
        "EXPECTED_SLURM_SCRIPT_SHA256=\"$(cut -d ' ' -f 1 \"$SLURM_CHECKSUM_PATH\")\"",
        (
            "EXECUTED_SLURM_SCRIPT_SHA256=\"$(sha256sum "
            "\"$EXECUTED_SLURM_SCRIPT_PATH\" | cut -d ' ' -f 1)\""
        ),
        (
            "GENERATED_SLURM_SCRIPT_SHA256=\"$(sha256sum "
            "\"$SLURM_SCRIPT_PATH\" | cut -d ' ' -f 1)\""
        ),
        (
            "if [[ \"$GENERATED_SLURM_SCRIPT_SHA256\" != "
            "\"$EXPECTED_SLURM_SCRIPT_SHA256\" || "
            "\"$EXECUTED_SLURM_SCRIPT_SHA256\" != "
            "\"$EXPECTED_SLURM_SCRIPT_SHA256\" ]]; then"
        ),
        "    echo \"Generated or executed Slurm script failed its checksum.\" >&2",
        "    exit 4",
        "fi",
        "cd \"$DATA_DIR\"",
        "JOB_START_TIMESTAMP=\"$(timestamp_utc)\"",
        "INHIBITOR_START_TIMESTAMP=\"$(timestamp_utc)\"",
        "set +e",
        "",
        (
            f"time {srun_prefix} {mpip_environment} {shlex.join(inhibitor_argv)} "
            f"> {shlex.quote(str(inhib_log))} 2>&1 &"
        ),
        "INHIBITOR_PID=$!",
        "if kill -0 \"$INHIBITOR_PID\" 2>/dev/null; then",
        "    INHIBITOR_ALIVE_AT_APPLICATION_START=true",
        "else",
        "    INHIBITOR_ALIVE_AT_APPLICATION_START=false",
        "fi",
        "APPLICATION_START_TIMESTAMP=\"$(timestamp_utc)\"",
        "",
        "if [[ \"$INHIBITOR_ALIVE_AT_APPLICATION_START\" == true ]]; then",
        (
            f"    time {srun_prefix} {mpip_environment} {case['application_command']} "
            f"> {shlex.quote(str(app_log))} 2>&1"
        ),
        "    APPLICATION_EXIT_STATUS=$?",
        "else",
        "    APPLICATION_EXIT_STATUS=125",
        "fi",
        "APPLICATION_EXIT_TIMESTAMP=\"$(timestamp_utc)\"",
        "",
        "if kill -0 \"$INHIBITOR_PID\" 2>/dev/null; then",
        "    INHIBITOR_ALIVE_AT_APPLICATION_EXIT=true",
        "else",
        "    INHIBITOR_ALIVE_AT_APPLICATION_EXIT=false",
        "fi",
        "",
        "wait \"$INHIBITOR_PID\"",
        "INHIBITOR_EXIT_STATUS=$?",
        "INHIBITOR_EXIT_TIMESTAMP=\"$(timestamp_utc)\"",
        "set -e",
        "",
        "shopt -s nullglob",
        "MPIP_PROFILE_FILES=(\"$MPIP_PROFILE_DIR\"/*.mpiP)",
        "MPIP_PROFILE_COUNT=${#MPIP_PROFILE_FILES[@]}",
        "OUTPUTS_COMPLETE=true",
        (
            "if [[ ! -f \"$APP_LOG\" || ! -f \"$INHIB_LOG\" || "
            "! -s \"$INHIB_STATS\" || \"$MPIP_PROFILE_COUNT\" -ne 2 ]]; then"
        ),
        "    OUTPUTS_COMPLETE=false",
        "fi",
        "",
        "RUN_VALID=true",
        (
            "if [[ \"$APPLICATION_EXIT_STATUS\" -ne 0 || "
            "\"$INHIBITOR_EXIT_STATUS\" -ne 0 || "
            "\"$INHIBITOR_ALIVE_AT_APPLICATION_START\" != true || "
            "\"$INHIBITOR_ALIVE_AT_APPLICATION_EXIT\" != true || "
            "\"$OUTPUTS_COMPLETE\" != true ]]; then"
        ),
        "    RUN_VALID=false",
        "fi",
        "",
        "RUN_STATUS_TMP=\"${RUN_STATUS_PATH}.tmp.$$\"",
        "cat > \"$RUN_STATUS_TMP\" <<EOF",
        "{",
        f"  \"application\": {json.dumps(case['application'])},",
        f"  \"candidate_alias\": {json.dumps(case['candidate_alias'])},",
        f"  \"work_package\": {json.dumps(case['work_package'])},",
        f"  \"family\": {json.dumps(case['family'])},",
        f"  \"repetition\": {case['repetition']},",
        f"  \"inhibitor_id\": {json.dumps(case['inhibitor_id'])},",
        (
            "  \"inhibitor_arguments\": "
            f"{json.dumps(case['inhibitor_arguments'], sort_keys=True)},"
        ),
        f"  \"application_argv\": {json.dumps(case['application_argv'])},",
        "  \"slurm_job_id\": \"${SLURM_JOB_ID}\",",
        "  \"start_timestamps\": {",
        "    \"job\": \"${JOB_START_TIMESTAMP}\",",
        "    \"inhibitor\": \"${INHIBITOR_START_TIMESTAMP}\",",
        "    \"application\": \"${APPLICATION_START_TIMESTAMP}\"",
        "  },",
        "  \"application_exit_timestamp\": \"${APPLICATION_EXIT_TIMESTAMP}\",",
        "  \"application_exit_status\": ${APPLICATION_EXIT_STATUS},",
        "  \"inhibitor_exit_timestamp\": \"${INHIBITOR_EXIT_TIMESTAMP}\",",
        "  \"inhibitor_exit_status\": ${INHIBITOR_EXIT_STATUS},",
        (
            "  \"inhibitor_alive_at_application_start\": "
            "${INHIBITOR_ALIVE_AT_APPLICATION_START},"
        ),
        (
            "  \"inhibitor_alive_at_application_exit\": "
            "${INHIBITOR_ALIVE_AT_APPLICATION_EXIT},"
        ),
        "  \"hostname_list\": ${HOSTNAME_LIST_JSON},",
        f"  \"case_manifest\": {json.dumps(str(manifest_path))},",
        f"  \"case_manifest_sha256\": {json.dumps(manifest_sha256)},",
        "  \"source_hashes\": {",
        (
            "    \"generator\": "
            f"{json.dumps(source_hashes['generator'], sort_keys=True)},"
        ),
        (
            "    \"network_inhibitor_source\": "
            f"{json.dumps(source_hashes['network_inhibitor_source'], sort_keys=True)},"
        ),
        (
            "    \"spack_setup\": "
            f"{json.dumps(source_hashes['spack_setup'], sort_keys=True)},"
        ),
        "    \"slurm_script\": {",
        f"      \"generated_path\": {json.dumps(str(slurm_script))},",
        f"      \"checksum_path\": {json.dumps(str(slurm_checksum))},",
        "      \"expected_sha256\": \"${EXPECTED_SLURM_SCRIPT_SHA256}\",",
        "      \"generated_sha256\": \"${GENERATED_SLURM_SCRIPT_SHA256}\",",
        "      \"executed_path\": \"${EXECUTED_SLURM_SCRIPT_PATH}\",",
        "      \"executed_sha256\": \"${EXECUTED_SLURM_SCRIPT_SHA256}\"",
        "    }",
        "  },",
        f"  \"config_hashes\": {json.dumps(config_hashes, sort_keys=True)},",
        (
            "  \"executable_hashes\": "
            f"{json.dumps(relevant_executable_hashes, sort_keys=True)},"
        ),
        "  \"output_status\": {",
        "    \"mpip_profile_count\": ${MPIP_PROFILE_COUNT},",
        "    \"outputs_complete\": ${OUTPUTS_COMPLETE}",
        "  },",
        "  \"valid\": ${RUN_VALID}",
        "}",
        "EOF",
        "mv \"$RUN_STATUS_TMP\" \"$RUN_STATUS_PATH\"",
        "",
        "if [[ \"$RUN_VALID\" != true ]]; then",
        "    exit 1",
        "fi",
        "",
    ]
    return "\n".join(lines)


def create_output_tree(timestamp_dir, data_root, scripts_root):
    if EXPERIMENTS_DIR.exists():
        if not EXPERIMENTS_DIR.is_dir() or EXPERIMENTS_DIR.is_symlink():
            raise RuntimeError(f"Experiment path is not a real directory: {EXPERIMENTS_DIR}")
    else:
        EXPERIMENTS_DIR.mkdir()

    timestamp_dir.mkdir()
    data_root.mkdir()
    scripts_root.mkdir()


def write_experiments(
    cases,
    manifest,
    manifest_path,
    source_hashes,
    config_hashes,
    executable_hashes,
):
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()

    for case in cases:
        data_dir = Path(case["data_directory"])
        mpip_dir = Path(case["mpip_profile_directory"])
        data_dir.mkdir()
        mpip_dir.mkdir()

        slurm_path = Path(case["slurm_script"])
        slurm_content = generate_slurm_script(
            case,
            manifest_path,
            manifest_sha256,
            source_hashes,
            config_hashes,
            executable_hashes,
        )
        with slurm_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(slurm_content)
        slurm_path.chmod(0o750)

        slurm_sha256 = hashlib.sha256(slurm_content.encode("utf-8")).hexdigest()
        checksum_path = Path(case["slurm_checksum"])
        with checksum_path.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{slurm_sha256}  {slurm_path.name}\n")
        checksum_path.chmod(0o444)

    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(manifest_text)
    manifest_path.chmod(0o444)

    checksum_path = manifest_path.with_suffix(".sha256")
    with checksum_path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{manifest_sha256}  {manifest_path.name}\n")
    checksum_path.chmod(0o444)

    return manifest_sha256


def main():
    args = parse_arguments()
    validate_dane_environment()
    apps = load_applications()

    configurations = WORK_PACKAGES[args.work_package]
    validate_configurations(args.work_package, configurations)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    created_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    timestamp_dir = EXPERIMENTS_DIR / timestamp
    data_root = timestamp_dir / "data"
    scripts_root = timestamp_dir / "slurm_scripts"
    manifest_path = timestamp_dir / "case_manifest.json"

    cases = build_cases(
        args.work_package, configurations, apps, data_root, scripts_root
    )
    source_hashes, config_hashes, executable_hashes = build_hashes(apps)
    manifest = build_manifest(
        created_at,
        args.work_package,
        timestamp_dir,
        cases,
        source_hashes,
        config_hashes,
        executable_hashes,
    )

    create_output_tree(timestamp_dir, data_root, scripts_root)
    manifest_sha256 = write_experiments(
        cases,
        manifest,
        manifest_path,
        source_hashes,
        config_hashes,
        executable_hashes,
    )

    print(f"Generated {len(cases)} individual Slurm scripts.")
    print(f"Timestamp directory: {timestamp_dir}")
    print(f"Case manifest: {manifest_path}")
    print(f"Manifest SHA-256: {manifest_sha256}")
    print("No jobs were submitted.")


if __name__ == "__main__":
    main()
