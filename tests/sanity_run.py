"""Run the baseline policies on 20 seeded episodes, print a summary, checks and two daily logs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from environment import ShopEnv
from environment.baselines import random_policy, rule_credit_policy, rule_policy
from environment.config import make_config

SEEDS = range(20)
TRUST_MARGIN = 0.1
LOG_SEED = 0
POLICIES = {
    "random_policy": random_policy,
    "rule_policy": rule_policy,
    "rule_credit_policy": rule_credit_policy,
}
SUMMARY_STATS = [
    ("return", "episode return"),
    ("final_cash", "final cash"),
    ("lost", "lost customers"),
    ("lost_per_day", "lost customers per day"),
    ("mean_trust", "mean trust"),
    ("lent", "debt lent"),
    ("defaults", "defaults (amount)"),
    ("haggle_rate", "haggle success rate"),
    ("haggles", "haggles per episode"),
    ("jara", "jara per episode"),
]


def run_episode(policy, seed, daily_log=None):
    """Play one full episode with a policy and return its totals, filling the daily log if given."""
    env = ShopEnv()
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    totals = {"return": 0.0, "lost": 0, "lent": 0.0, "defaults": 0.0, "haggles": 0,
              "haggle_successes": 0, "jara": 0, "trust_sum": 0.0, "days": 0}
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(policy(obs, env))
        add_day(totals, reward, info)
        if daily_log is not None:
            daily_log.append(format_day(info, env))
    finish_totals(totals, info)
    return totals


def add_day(totals, reward, info):
    """Add one day's reward and counters to the episode totals."""
    totals["return"] += reward
    totals["lost"] += info["lost_customers"]
    totals["lent"] += info["credit_given"]
    totals["defaults"] += info["defaults"]
    totals["haggles"] += info["haggles"]
    totals["haggle_successes"] += info["haggle_successes"]
    totals["jara"] += info["jara_count"]
    totals["trust_sum"] += info["trust"]
    totals["days"] += 1


def finish_totals(totals, info):
    """Turn the running sums into the per-episode stats the summary prints."""
    totals["final_cash"] = info["cash"]
    totals["bankrupt"] = info["bankrupt"]
    totals["haggle_rate"] = totals["haggle_successes"] / max(totals["haggles"], 1)
    totals["mean_trust"] = totals["trust_sum"] / totals["days"]
    totals["lost_per_day"] = totals["lost"] / totals["days"]


def format_day(info, env):
    """Format one day as a compact log line: day, action, sales, cash, debt, old debt, defaults and trust."""
    indomie, soap, bags, mudu = info["units_sold"]
    _, old_debt = env.debt_by_age()
    return (
        f"{info['day']:>3}  {info['action_name']:<22} "
        f"sold I{indomie:>4.0f} S{soap:>3.0f} R{bags:>2.0f} L{mudu:>3.0f}  "
        f"lost {info['lost_customers']:>2}  cash {info['cash']:>9.0f}  "
        f"debt {info['debt_outstanding']:>7.0f}  old {old_debt:>7.0f}  "
        f"default {info['defaults']:>6.0f}  trust {info['trust']:.3f}"
    )


def summarise(name, episodes):
    """Print the mean and standard deviation of each episode metric for one policy."""
    print(f"\n{name} over {len(episodes)} episodes")
    for key, label in SUMMARY_STATS:
        values = np.array([episode[key] for episode in episodes], dtype=np.float64)
        print(f"  {label:<24} mean {values.mean():>12.2f}  std {values.std():>12.2f}")
    print(f"  {'bankruptcies':<24} {sum(episode['bankrupt'] for episode in episodes)}")


def mean_of(episodes, key):
    """Return the mean of one metric across episodes."""
    return float(np.mean([episode[key] for episode in episodes]))


def run_checks(runs):
    """Print ok or FAILED for each sanity check and return whether all passed."""
    rule_runs = runs["rule_policy"]
    trust_floor = make_config()["trust"]["min"]
    checks = [
        ("rule_policy mean return is higher than random_policy's",
         mean_of(rule_runs, "return") > mean_of(runs["random_policy"], "return")),
        ("rule_policy mean return is positive", mean_of(rule_runs, "return") > 0),
        ("rule_policy has no bankruptcies", not any(episode["bankrupt"] for episode in rule_runs)),
        (f"rule_policy mean trust stays above the trust floor + {TRUST_MARGIN}",
         mean_of(rule_runs, "mean_trust") > trust_floor + TRUST_MARGIN),
        ("rule_policy lost customers per day is below 5", mean_of(rule_runs, "lost_per_day") < 5),
    ]
    for name, episodes in runs.items():
        checks.append((f"{name}: at least one haggle per episode on average", mean_of(episodes, "haggles") >= 1))
        checks.append((f"{name}: at least one jara per episode on average", mean_of(episodes, "jara") >= 1))
    print("\nChecks")
    for name, passed in checks:
        print(f"  {'ok' if passed else 'FAILED'}: {name}")
    return all(passed for _, passed in checks)


def print_daily_log(name):
    """Print the full daily log of one policy on the log seed."""
    daily_log = []
    run_episode(POLICIES[name], LOG_SEED, daily_log)
    print(f"\n{name} daily log, seed {LOG_SEED}")
    print("\n".join(daily_log))


def main():
    """Run every policy, print the summaries and checks, then print the two rule policy daily logs."""
    runs = {name: [run_episode(policy, seed) for seed in SEEDS] for name, policy in POLICIES.items()}
    for name, episodes in runs.items():
        summarise(name, episodes)
    if run_checks(runs):
        print("all checks passed")
    print_daily_log("rule_policy")
    print_daily_log("rule_credit_policy")


if __name__ == "__main__":
    main()
