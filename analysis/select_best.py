"""Pick each algorithm's best sweep config from validation results only, and write logs/selection.csv.

Run:  uv run python -m analysis.select_best

The selection score is the mean greedy validation return over the last 5 eval points of a run,
which rewards configs that end training good and stable rather than one lucky eval. Test columns
are shown for information only; they never influence the choice.
"""
from pathlib import Path

from training.common import read_rows, write_rows
from training.experiments import EXPERIMENTS

LOG_DIR = Path("logs")
SUMMARY_PATH = LOG_DIR / "summary.csv"
SELECTION_PATH = LOG_DIR / "selection.csv"
LAST_EVALS = 5
ALGOS = ("dqn", "reinforce", "a2c", "ppo")
PG_ALGOS = ("reinforce", "a2c", "ppo")
SWEEP_IDS = {experiment["exp_id"] for experiments in EXPERIMENTS.values() for experiment in experiments}
COLUMNS = (
    "algo", "rank", "winner", "exp_id", "description", "selection_score", "val_final_return", "val_best_return",
    "test_best_return_mean", "test_best_net_worth_mean", "test_best_sample_return_mean",
)


def eval_path(algo, exp_id):
    """Return the eval CSV of one run."""
    return LOG_DIR / algo / f"{exp_id}_eval.csv"


def selection_score(algo, exp_id, last=LAST_EVALS):
    """Return the mean greedy validation return over a run's last eval points."""
    rows, _ = read_rows(eval_path(algo, exp_id))
    returns = [float(row["val_return_mean"]) for row in rows[-last:]]
    return sum(returns) / len(returns)


def number(row, key):
    """Read a float column from a summary row, or None when the column is empty or missing."""
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def ranked_runs(algo, summary_rows):
    """Return one algorithm's sweep runs with their scores, best selection score first."""
    runs = []
    for row in summary_rows:
        if row["algo"] != algo or row["exp_id"] not in SWEEP_IDS:
            continue
        runs.append({
            "algo": algo,
            "exp_id": row["exp_id"],
            "description": row["description"],
            "selection_score": selection_score(algo, row["exp_id"]),
            "val_final_return": number(row, "val_final_return"),
            "val_best_return": number(row, "val_best_return"),
            "test_best_return_mean": number(row, "test_best_return_mean"),
            "test_best_net_worth_mean": number(row, "test_best_net_worth_mean"),
            "test_best_sample_return_mean": number(row, "test_best_sample_return_mean") if algo in PG_ALGOS else None,
        })
    runs.sort(key=lambda run: run["selection_score"], reverse=True)
    for rank, run in enumerate(runs, start=1):
        run["rank"] = rank
        run["winner"] = "yes" if rank == 1 else ""
    return runs


def winners(selection_rows):
    """Return the winning exp_id of each algorithm from selection rows."""
    return {row["algo"]: row["exp_id"] for row in selection_rows if row["winner"] == "yes"}


def fmt(value, digits=2):
    """Format a number for the printed table, or a dash when it is missing."""
    if value is None:
        return "-"
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.{digits}f}"


def print_table(algo, runs, top=None):
    """Print one algorithm's ranked table, marking the winner."""
    print(f"\n{algo.upper()}  (score = mean val greedy return over the last {LAST_EVALS} evals)")
    header = f"  {'':2}{'exp_id':<14}{'description':<40}{'score':>8}{'val fin':>9}{'val best':>9}{'test best':>10}{'test nw':>12}"
    if algo in PG_ALGOS:
        header += f"{'test smp':>10}"
    print(header)
    for run in runs[:top]:
        mark = "* " if run["winner"] else "  "
        line = (f"  {mark}{run['exp_id']:<14}{run['description'][:38]:<40}{fmt(run['selection_score']):>8}"
                f"{fmt(run['val_final_return']):>9}{fmt(run['val_best_return']):>9}"
                f"{fmt(run['test_best_return_mean']):>10}{fmt(run['test_best_net_worth_mean']):>12}")
        if algo in PG_ALGOS:
            line += f"{fmt(run['test_best_sample_return_mean']):>10}"
        print(line)


def main():
    """Rank every algorithm's sweep runs on validation, print the tables and write logs/selection.csv."""
    summary_rows, _ = read_rows(SUMMARY_PATH)
    selection = []
    for algo in ALGOS:
        runs = ranked_runs(algo, summary_rows)
        print_table(algo, runs)
        selection.extend(runs)
    write_rows(SELECTION_PATH, list(COLUMNS), [{key: run[key] for key in COLUMNS} for run in selection])
    print(f"\nwinners: {winners(selection)}")
    print(f"wrote {SELECTION_PATH}")


if __name__ == "__main__":
    main()
