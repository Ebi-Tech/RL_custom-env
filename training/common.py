"""Tools shared by every training algorithm: seeding, env building, evaluation, CSV logs and checkpoints."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torchrl.envs import GymWrapper, RewardSum, StepCounter, TransformedEnv

from environment import ACTION_NAMES, ShopEnv

EVAL_SEEDS = range(1000, 1020)
EVAL_METRICS = (
    "return", "final_cash", "final_net_worth", "lost_per_day", "trust", "debt_lent", "defaults",
    "invalid_rate", "hold_rate",
)
HOLD_ACTION = ACTION_NAMES.index("hold")
POLICY_KINDS = ("q", "logits")


def set_seed(seed):
    """Seed python, numpy and torch so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(market="default", seed=0):
    """Return the TorchRL version of ShopEnv with episode return and step count tracking, seeded."""
    env = TransformedEnv(GymWrapper(ShopEnv(market=market)))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.set_seed(seed)
    return env


def run_eval_episode(policy_fn, market, seed):
    """Play one full episode of the raw ShopEnv with a policy_fn and return its summary stats."""
    env = ShopEnv(market=market)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    stats = {"return": 0.0, "lost": 0, "trust_sum": 0.0, "debt_lent": 0.0, "defaults": 0.0,
             "invalid": 0, "days": 0}
    action_counts = np.zeros(len(ACTION_NAMES))
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(policy_fn(obs, env))
        obs, reward, terminated, truncated, info = env.step(action)
        action_counts[action] += 1
        add_eval_day(stats, reward, info)
    return {
        "return": stats["return"],
        "final_cash": info["cash"],
        "final_net_worth": env.net_worth(),
        "lost_per_day": stats["lost"] / stats["days"],
        "trust": stats["trust_sum"] / stats["days"],
        "debt_lent": stats["debt_lent"],
        "defaults": stats["defaults"],
        "invalid_rate": stats["invalid"] / stats["days"],
        "hold_rate": action_counts[HOLD_ACTION] / stats["days"],
        "bankrupt": bool(terminated),
        "action_counts": action_counts,
    }


def add_eval_day(stats, reward, info):
    """Add one day's reward and info counters to the running episode stats."""
    stats["return"] += reward
    stats["lost"] += info["lost_customers"]
    stats["trust_sum"] += info["trust"]
    stats["debt_lent"] += info["credit_given"]
    stats["defaults"] += info["defaults"]
    stats["invalid"] += int(info["invalid_action"])
    stats["days"] += 1


def evaluate_policy(policy_fn, market="default", seeds=EVAL_SEEDS):
    """Run one episode per held-out seed and return the mean and std of each metric.

    The result also holds the bankruptcy count and the share of days each action was chosen,
    so trained agents and the baselines can be compared on exactly the same footing.
    """
    episodes = [run_eval_episode(policy_fn, market, seed) for seed in seeds]
    result = {"episodes": len(episodes)}
    for metric in EVAL_METRICS:
        values = np.array([episode[metric] for episode in episodes], dtype=np.float64)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std())
    result["bankruptcies"] = sum(episode["bankrupt"] for episode in episodes)
    counts = np.sum([episode["action_counts"] for episode in episodes], axis=0)
    result["action_freq"] = {name: float(count / counts.sum()) for name, count in zip(ACTION_NAMES, counts)}
    return result


def flatten_eval_result(result):
    """Turn an evaluate_policy result into one flat dict, with one freq_<action> column per action."""
    flat = {key: value for key, value in result.items() if key != "action_freq"}
    for name, share in result["action_freq"].items():
        flat[f"freq_{name}"] = share
    return flat


def greedy_fn_from_module(module, kind):
    """Wrap a trained torch module into a policy_fn that always picks its highest scoring action.

    For kind "q" the module outputs Q-values and for kind "logits" it outputs action logits;
    in both cases the deterministic choice is the argmax.
    """
    if kind not in POLICY_KINDS:
        raise ValueError(f"Unknown kind {kind!r}, expected one of {POLICY_KINDS}")

    def policy_fn(obs, env):
        """Return the argmax action of the module for one observation."""
        with torch.no_grad():
            scores = module(torch.as_tensor(obs, dtype=torch.float32))
        return int(torch.argmax(scores).item())

    return policy_fn


def append_row(path, row):
    """Append one dict as a CSV row, creating the file and header if missing.

    If the row has columns the file does not have yet, the file is rewritten with the merged
    header, so logs/summary.csv can hold rows from algorithms with different hyperparameters.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, header = read_rows(path)
    new_columns = [key for key in row if key not in header]
    if path.exists() and not new_columns:
        with path.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=header).writerow(row)
        return
    write_rows(path, header + new_columns, rows + [row])


def read_rows(path):
    """Return the rows and header of a CSV file, or empty lists when it does not exist."""
    if not path.exists():
        return [], []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path, header, rows):
    """Write a full CSV file with the given header, leaving missing values empty."""
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, module, config, extra):
    """Save the module weights, the full config and extra info such as eval results in one .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": module.state_dict(), "config": config, "extra": extra}, path)


def load_checkpoint(path, build_fn):
    """Rebuild a module from a checkpoint using build_fn(config), and return it with the saved data."""
    data = torch.load(path)
    module = build_fn(data["config"])
    module.load_state_dict(data["state_dict"])
    module.eval()
    return module, data
