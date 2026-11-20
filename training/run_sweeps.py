"""Run every sweep experiment (DQN, REINFORCE, A2C and PPO) as a queue shared by parallel worker processes.

Full sweep:  uv run python -m training.run_sweeps --workers 3
Subset:      uv run python -m training.run_sweeps --algo ppo,a2c --only ppo_03,a2c_06
Dry run:     uv run python -m training.run_sweeps --only reinforce_10,dqn_15 --total-frames 3000
More seeds:  uv run python -m training.run_sweeps --only dqn_03,ppo_05 --seeds 3

Each experiment runs in its own subprocess, so a crash only fails that experiment. Each worker
writes its summary row to its own file in logs/sweep/summary_parts/, and only this parent
process merges those rows into logs/summary.csv, so parallel workers never write the same file.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from training.common import append_row, finished_exp_ids, read_rows
from training.experiments import EXPERIMENTS, find_experiment, seed_copy

ALGO_ORDER = ("dqn", "reinforce", "a2c", "ppo")
TRAINING_MODULES = {
    "dqn": "training.dqn_training",
    "reinforce": "training.pg_training",
    "a2c": "training.pg_training",
    "ppo": "training.pg_training",
}
LOG_DIR = Path("logs")
SWEEP_DIR = LOG_DIR / "sweep"
PARTS_DIR = SWEEP_DIR / "summary_parts"
SUMMARY_PATH = LOG_DIR / "summary.csv"
DEFAULT_WORKERS = 3
POLL_SECONDS = 2.0
LOG_TAIL_CHARS = 300


def split_list(text):
    """Turn a comma separated option into a list of stripped names, or None when the option was not given."""
    if not text:
        return None
    return [name.strip() for name in text.split(",") if name.strip()]


def chosen_algos(algo_text):
    """Return the algorithms to sweep in the fixed interleaving order, rejecting unknown names."""
    wanted = split_list(algo_text) or list(ALGO_ORDER)
    unknown = [algo for algo in wanted if algo not in ALGO_ORDER]
    if unknown:
        raise SystemExit(f"Unknown algorithms: {unknown}, expected some of {ALGO_ORDER}")
    return [algo for algo in ALGO_ORDER if algo in wanted]


def check_only_ids(only, algos):
    """Stop with a message if any --only id is not a sweep experiment of the chosen algorithms."""
    unknown = [exp_id for exp_id in only if not any(find_experiment(exp_id, algo) for algo in algos)]
    if unknown:
        raise SystemExit(f"Unknown experiment ids for {algos}: {unknown}")


def interleave(lists):
    """Take one item from each list in turn until all are used up, so every algorithm starts early."""
    queue = []
    for position in range(max((len(items) for items in lists), default=0)):
        queue.extend(items[position] for items in lists if position < len(items))
    return queue


def build_queue(algos, only, done, seeds=1):
    """Return the jobs to run: the chosen experiments interleaved across algorithms, minus finished ones.

    With seeds above 1, each chosen experiment also gets copies with seeds 1 to seeds - 1, named
    <exp_id>_s<seed>; the original seed 0 run counts as the first seed.
    """
    per_algo = []
    for algo in algos:
        chosen = [experiment for experiment in EXPERIMENTS[algo] if only is None or experiment["exp_id"] in only]
        experiments = [copy for experiment in chosen for copy in with_seed_copies(experiment, seeds)]
        per_algo.append([{"algo": algo, **experiment} for experiment in experiments if experiment["exp_id"] not in done])
    return interleave(per_algo)


def with_seed_copies(experiment, seeds):
    """Return an experiment followed by its seed copies for seeds 1 to seeds - 1."""
    return [experiment] + [seed_copy(experiment, seed) for seed in range(1, seeds)]


def worker_command(job, total_frames):
    """Return the command line that trains one experiment and writes its summary row to its own part file."""
    command = [sys.executable, "-m", TRAINING_MODULES[job["algo"]]]
    if job["algo"] != "dqn":
        command += ["--algo", job["algo"]]
    command += ["--exp-id", job["exp_id"], "--summary-path", str(part_path(job["exp_id"]))]
    if total_frames is not None:
        command += ["--total-frames", str(total_frames)]
    return command


def part_path(exp_id):
    """Return the file one worker writes its summary row to."""
    return PARTS_DIR / f"{exp_id}.csv"


def log_path(exp_id):
    """Return the file that receives one experiment's stdout and stderr."""
    return SWEEP_DIR / f"{exp_id}.log"


def start_job(job, total_frames):
    """Launch one experiment in its own subprocess, sending its output to its log file."""
    part_path(job["exp_id"]).unlink(missing_ok=True)
    log_file = log_path(job["exp_id"]).open("w")
    process = subprocess.Popen(worker_command(job, total_frames), stdout=log_file, stderr=subprocess.STDOUT)
    return {"job": job, "process": process, "log_file": log_file, "start": time.time()}


def merge_part(path):
    """Append the rows of one summary part file to logs/summary.csv, delete the part, and return its rows."""
    rows, _ = read_rows(path)
    for row in rows:
        append_row(SUMMARY_PATH, row)
    path.unlink()
    return rows


def merge_leftover_parts():
    """Merge part files left by an earlier runner that stopped before merging them."""
    for path in sorted(PARTS_DIR.glob("*.csv")):
        rows = merge_part(path)
        print(f"merged leftover summary row for {path.stem} ({len(rows)} rows)")


def log_tail(exp_id):
    """Return the last characters of an experiment's log, to show why it failed."""
    text = log_path(exp_id).read_text(errors="replace").strip()
    return text[-LOG_TAIL_CHARS:].replace("\n", " | ")


def finish_job(running):
    """Close a finished job's log, merge its summary row, print its result line and return a failure or None."""
    running["log_file"].close()
    job = running["job"]
    wall_time = time.time() - running["start"]
    code = running["process"].returncode
    path = part_path(job["exp_id"])
    if code == 0 and path.exists():
        rows = merge_part(path)
        val_final = float(rows[-1]["val_final_return"])
        print(f"finish {job['exp_id']:<13} {wall_time:>7.1f}s  val final return {val_final:>8.2f}", flush=True)
        return None
    print(f"FAILED {job['exp_id']:<13} {wall_time:>7.1f}s  exit code {code}, see {log_path(job['exp_id'])}", flush=True)
    return {"exp_id": job["exp_id"], "algo": job["algo"], "exit_code": code, "tail": log_tail(job["exp_id"])}


def run_queue(queue, workers, total_frames):
    """Keep up to workers experiments running until the queue is empty, and return the failures."""
    pending = list(queue)
    running = []
    failures = []
    try:
        while pending or running:
            while pending and len(running) < workers:
                job = pending.pop(0)
                running.append(start_job(job, total_frames))
                print(f"start  {job['exp_id']:<13} ({len(running)} running, {len(pending)} queued)", flush=True)
            time.sleep(POLL_SECONDS)
            for item in [item for item in running if item["process"].poll() is not None]:
                running.remove(item)
                failure = finish_job(item)
                if failure:
                    failures.append(failure)
    finally:
        stop_running(running)
    return failures


def stop_running(running):
    """Terminate any experiments still running, for example after Ctrl+C."""
    for item in running:
        item["process"].terminate()
        item["process"].wait()
        item["log_file"].close()
        print(f"stopped {item['job']['exp_id']}")


def print_failures(failures):
    """Print a table of failed experiments, or say that none failed."""
    if not failures:
        print("\nno failures")
        return
    print(f"\n{len(failures)} failed experiments")
    print(f"{'exp_id':<13} {'algo':<10} {'exit':>5}  last log lines")
    for failure in failures:
        print(f"{failure['exp_id']:<13} {failure['algo']:<10} {str(failure['exit_code']):>5}  {failure['tail']}")


def parse_args():
    """Read the sweep runner options."""
    parser = argparse.ArgumentParser(description="Run the DQN, REINFORCE, A2C and PPO sweeps in parallel.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="experiments running at once")
    parser.add_argument("--algo", help="comma separated algorithms to include, default all four")
    parser.add_argument("--only", help="comma separated experiment ids to include")
    parser.add_argument("--total-frames", type=int, help="override total frames for every experiment, for dry runs")
    parser.add_argument("--seeds", type=int, default=1,
                        help="seeds per experiment: 3 adds copies with seed 1 and 2 named <exp_id>_s1 and _s2")
    return parser.parse_args()


def main():
    """Build the queue, skip finished experiments, run the rest in parallel and report failures."""
    args = parse_args()
    algos = chosen_algos(args.algo)
    only = split_list(args.only)
    if only:
        check_only_ids(only, algos)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    merge_leftover_parts()
    done = finished_exp_ids(log_dir=LOG_DIR)
    queue = build_queue(algos, only, done, args.seeds)
    skipped = [exp_id for exp_id in done if any(find_experiment(exp_id, algo) for algo in algos)
               and (only is None or exp_id in only or exp_id.rsplit("_s", 1)[0] in only)]
    print(f"{len(queue)} experiments queued, {len(skipped)} already in {SUMMARY_PATH}, {args.workers} workers")
    start = time.time()
    failures = run_queue(queue, args.workers, args.total_frames)
    print(f"\nsweep finished in {time.time() - start:.1f}s")
    print_failures(failures)


if __name__ == "__main__":
    main()
