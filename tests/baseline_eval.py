"""Evaluate the three baseline policies on the held-out seeds and write logs/baselines.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.baselines import random_policy, rule_credit_policy, rule_policy
from training.common import EVAL_SEEDS, append_row, evaluate_policy, flatten_eval_result

OUTPUT_PATH = Path("logs/baselines.csv")
MARKET = "default"
POLICIES = {
    "random_policy": random_policy,
    "rule_policy": rule_policy,
    "rule_credit_policy": rule_credit_policy,
}


def print_result(name, result):
    """Print the headline metrics of one baseline as a short block."""
    print(f"\n{name} on {result['episodes']} held-out seeds")
    for metric in ("return", "final_cash", "lost_per_day", "trust", "debt_lent", "defaults"):
        print(f"  {metric:<14} mean {result[f'{metric}_mean']:>12.2f}  std {result[f'{metric}_std']:>12.2f}")
    print(f"  {'bankruptcies':<14} {result['bankruptcies']}")
    top_actions = sorted(result["action_freq"].items(), key=lambda item: item[1], reverse=True)[:4]
    print("  top actions    " + ", ".join(f"{action} {share:.2f}" for action, share in top_actions))


def main():
    """Evaluate every baseline and rewrite logs/baselines.csv with one row per policy."""
    OUTPUT_PATH.unlink(missing_ok=True)
    for name, policy in POLICIES.items():
        result = evaluate_policy(policy, market=MARKET, seeds=EVAL_SEEDS)
        print_result(name, result)
        row = {"policy": name, "market": MARKET, "seeds": f"{EVAL_SEEDS.start}-{EVAL_SEEDS.stop - 1}"}
        row.update(flatten_eval_result(result))
        append_row(OUTPUT_PATH, row)
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
