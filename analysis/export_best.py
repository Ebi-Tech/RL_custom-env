"""Copy one typical trained agent per algorithm, and the overall best one, into models/best/ for the demo.

Run:  uv run python -m analysis.export_best

For each algorithm, the seed run whose test return is closest to the mean of its 3 seed runs is
exported (a typical run, not the luckiest). The overall best agent is the algorithm with the
highest mean test return across its seeds in the default market. All test numbers come from the
val-selected best checkpoints in logs/summary.csv.
"""
import shutil
from pathlib import Path

import numpy as np

from analysis.generalization import SEED_COPIES, best_checkpoint_path, run_id
from analysis.select_best import ALGOS, SELECTION_PATH, SUMMARY_PATH, winners
from training.common import read_rows

BEST_DIR = Path("models/best")
README_PATH = BEST_DIR / "README.md"


def seed_runs(summary_rows, algo, exp_id):
    """Return the summary rows of an algorithm's 3 seed runs of its selected config."""
    by_id = {row["exp_id"]: row for row in summary_rows if row["algo"] == algo}
    return [{"seed": seed, **by_id[run_id(exp_id, seed)]} for seed in SEED_COPIES]


def typical_run(runs):
    """Return the run whose test return is closest to the mean test return of all the runs."""
    returns = np.array([float(run["test_best_return_mean"]) for run in runs])
    return runs[int(np.argmin(np.abs(returns - returns.mean())))], float(returns.mean()), float(returns.std())


def export(algo, run):
    """Copy a run's best checkpoint to models/best/<algo>.pt and return the destination."""
    destination = BEST_DIR / f"{algo}.pt"
    shutil.copyfile(best_checkpoint_path(algo, run["exp_id"]), destination)
    return destination


def readme_text(exported, best_algo):
    """Return the README that lists every exported file with its source run, seed, test return and net worth."""
    lines = [
        "# Exported agents",
        "",
        "Each file is the val-selected best checkpoint of one training run, chosen as the typical run",
        "among its algorithm's 3 seeds (test return closest to the mean across seeds).",
        "Test numbers: greedy policy on test seeds 1000 to 1019, default market.",
        "",
        "| File | Algorithm | Source run | Seed | Test return | Test net worth | Mean +- std over 3 seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    for algo, item in exported.items():
        run = item["run"]
        lines.append(f"| {item['file'].name} | {algo} | {run['exp_id']} | {run['seed']} | "
                     f"{float(run['test_best_return_mean']):.2f} | {float(run['test_best_net_worth_mean']):,.0f} | "
                     f"{item['mean']:.2f} +- {item['std']:.2f} |")
    best = exported[best_algo]
    lines += ["", f"best_agent.pt is a copy of {best['file'].name} ({best_algo}, run {best['run']['exp_id']}): "
              "the algorithm with the highest mean test return across its 3 seeds.",
              "", "Watch it: uv run main.py (play.py loads best_agent.pt by default)."]
    return "\n".join(lines) + "\n"


def main():
    """Export the typical run of each algorithm and the overall best agent, then write the README."""
    summary_rows, _ = read_rows(SUMMARY_PATH)
    selected = winners(read_rows(SELECTION_PATH)[0])
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    exported = {}
    for algo in ALGOS:
        run, mean, std = typical_run(seed_runs(summary_rows, algo, selected[algo]))
        exported[algo] = {"run": run, "mean": mean, "std": std, "file": export(algo, run)}
        print(f"{algo:<10} {run['exp_id']:<16} seed {run['seed']}  test return {float(run['test_best_return_mean']):.2f}"
              f"  (mean {mean:.2f} +- {std:.2f})  -> {exported[algo]['file']}")
    best_algo = max(exported, key=lambda algo: exported[algo]["mean"])
    shutil.copyfile(exported[best_algo]["file"], BEST_DIR / "best_agent.pt")
    README_PATH.write_text(readme_text(exported, best_algo))
    print(f"best agent: {best_algo} -> {BEST_DIR / 'best_agent.pt'}\nwrote {README_PATH}")


if __name__ == "__main__":
    main()
