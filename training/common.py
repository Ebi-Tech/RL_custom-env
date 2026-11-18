"""Tools shared by every training algorithm: seeding, env building, evaluation, logs, checkpoints and summaries."""
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torchrl.envs import GymWrapper, RewardSum, StepCounter, TransformedEnv

from environment import ACTION_NAMES, ShopEnv

VAL_SEEDS = range(500, 520)
TEST_SEEDS = range(1000, 1020)
EVAL_METRICS = (
    "return", "final_cash", "final_net_worth", "lost_per_day", "trust", "debt_lent", "defaults",
    "invalid_rate", "hold_rate",
)
HOLD_ACTION = ACTION_NAMES.index("hold")
POLICY_KINDS = ("q", "logits")
SUMMARY_TEST_METRICS = ("net_worth", "invalid_rate", "hold_rate", "trust", "lost_per_day")
SUMMARY_METRIC_KEYS = {"net_worth": "final_net_worth"}


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
        "bankrupt": bool(info["bankrupt"]),
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


def evaluate_policy(policy_fn, market="default", seeds=TEST_SEEDS):
    """Run one episode per seed and return the mean and std of each metric.

    Training passes VAL_SEEDS for model selection; TEST_SEEDS are only used after training ends.

    The result also holds the bankruptcy count and the share of days each action was chosen,
    so trained agents and the baselines can be compared on exactly the same footing.
    """
    return evaluate_seeded_policy(lambda seed: policy_fn, market, seeds)


def evaluate_seeded_policy(make_policy_fn, market="default", seeds=TEST_SEEDS):
    """Like evaluate_policy, but builds a fresh policy_fn for each seed with make_policy_fn(seed).

    This lets a stochastic policy use its own random generator per seed, so its results repeat exactly.
    """
    episodes = [run_eval_episode(make_policy_fn(seed), market, seed) for seed in seeds]
    result = {"episodes": len(episodes)}
    for metric in EVAL_METRICS:
        values = np.array([episode[metric] for episode in episodes], dtype=np.float64)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std())
    result["bankruptcies"] = sum(episode["bankrupt"] for episode in episodes)
    counts = np.sum([episode["action_counts"] for episode in episodes], axis=0)
    result["action_freq"] = {name: float(count / counts.sum()) for name, count in zip(ACTION_NAMES, counts)}
    return result


def flatten_eval_result(result, prefix=""):
    """Turn an evaluate_policy result into one flat dict with prefixed keys and one freq column per action.

    The prefix names the seed set, for example "test_", so every column says where its number came from.
    """
    flat = {f"{prefix}{key}": value for key, value in result.items() if key != "action_freq"}
    for name, share in result["action_freq"].items():
        flat[f"{prefix}freq_{name}"] = share
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


def sampled_policy_maker(module):
    """Return make_policy_fn(seed) for a logits module that samples actions from its softmax distribution.

    Each seed gets its own torch generator seeded with that seed, so a sampled evaluation is
    reproducible and never touches the global torch random state used by training.
    """

    def make_policy_fn(seed):
        """Return a sampling policy_fn whose random draws come from a generator seeded with seed."""
        generator = torch.Generator().manual_seed(seed)

        def policy_fn(obs, env):
            """Draw one action from the module's action distribution for one observation."""
            with torch.no_grad():
                logits = module(torch.as_tensor(obs, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1, generator=generator).item())

        return policy_fn

    return make_policy_fn


def evaluate_sampled(module, market="default", seeds=TEST_SEEDS):
    """Evaluate a logits module with actions sampled from its distribution, one fixed generator per seed."""
    return evaluate_seeded_policy(sampled_policy_maker(module), market, seeds)


def append_row(path, row):
    """Append one dict as a CSV row, creating the file and header if missing.

    If the row has columns the file does not have yet, the file is rewritten with the merged
    header, so logs/summary.csv can hold rows from algorithms with different hyperparameters.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = read_header(path)
    new_columns = [key for key in row if key not in header]
    if header and not new_columns:
        with path.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=header, restval="").writerow(row)
        return
    rows, _ = read_rows(path)
    write_rows(path, header + new_columns, rows + [row])


def read_header(path):
    """Return only the header of a CSV file, or an empty list when it does not exist or is empty."""
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return next(csv.reader(file), [])


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


def experiment_paths(exp_id, log_subdir, model_subdir, log_dir="logs", model_dir="models"):
    """Return every file path one experiment writes, under logs/<log_subdir> and models/<model_subdir>."""
    log_folder = Path(log_dir) / log_subdir
    model_folder = Path(model_dir) / model_subdir
    return {
        "train": log_folder / f"{exp_id}_train.csv",
        "eval": log_folder / f"{exp_id}_eval.csv",
        "summary": Path(log_dir) / "summary.csv",
        "final": model_folder / f"{exp_id}_final.pt",
        "best": model_folder / f"{exp_id}_best.pt",
    }


def remove_old_logs(paths):
    """Delete this experiment's old train and eval CSVs so a rerun does not mix two runs."""
    for key in ("train", "eval"):
        paths[key].unlink(missing_ok=True)


def new_tracker(config):
    """Return the running state every training loop keeps: frames, episodes, timing and validation results."""
    return {
        "frames": 0,
        "episodes": 0,
        "best_val": None,
        "best_val_frames": None,
        "last_val": None,
        "last_val_frames": None,
        "start_time": time.time(),
        "next_eval": config["eval_every_frames"],
    }


def elapsed_seconds(tracker):
    """Return the seconds since the run started."""
    return time.time() - tracker["start_time"]


def checkpoint_extra(val_result, tracker, **more):
    """Return the extra info stored next to the weights: the validation result, frames, time and anything more."""
    extra = {"val_eval": val_result, "frames": tracker["frames"], "wall_time_s": elapsed_seconds(tracker)}
    extra.update(more)
    return extra


def val_row(frames, result):
    """Return one row of the eval CSV, every metric prefixed val_ because it comes from VAL_SEEDS."""
    return {
        "frames": frames,
        "val_return_mean": result["return_mean"],
        "val_return_std": result["return_std"],
        "val_final_cash_mean": result["final_cash_mean"],
        "val_net_worth_mean": result["final_net_worth_mean"],
        "val_lost_per_day_mean": result["lost_per_day_mean"],
        "val_trust_mean": result["trust_mean"],
        "val_defaults_mean": result["defaults_mean"],
        "val_invalid_rate_mean": result["invalid_rate_mean"],
        "val_hold_rate_mean": result["hold_rate_mean"],
    }


def val_sample_columns(result):
    """Return the eval CSV columns of a sampled validation, every name prefixed val_sample_."""
    return {
        "val_sample_return_mean": result["return_mean"],
        "val_sample_return_std": result["return_std"],
        "val_sample_net_worth_mean": result["final_net_worth_mean"],
        "val_sample_invalid_rate_mean": result["invalid_rate_mean"],
        "val_sample_hold_rate_mean": result["hold_rate_mean"],
    }


def record_validation(tracker, result, paths, module, config, extra, more_columns=None):
    """Log one validation result, remember it, and save the module as the best checkpoint when it is the best so far.

    The best checkpoint is always chosen on result, the greedy return; more_columns only adds to the CSV row.
    """
    row = val_row(tracker["frames"], result)
    row.update(more_columns or {})
    append_row(paths["eval"], row)
    tracker["last_val"] = result
    tracker["last_val_frames"] = tracker["frames"]
    if tracker["best_val"] is None or result["return_mean"] > tracker["best_val"]["return_mean"]:
        tracker["best_val"] = result
        tracker["best_val_frames"] = tracker["frames"]
        save_checkpoint(paths["best"], module, config, extra)


def evaluate_on_test(final_module, best_path, build_fn, kind, market, test_seeds, sampled=False):
    """Score the final module and the best checkpoint on the test seeds, which training never saw.

    With sampled=True the logits modules are also scored with sampled actions, under the keys
    final_sample and best_sample.
    """
    best_module, _ = load_checkpoint(best_path, build_fn)
    modules = {"final": final_module, "best": best_module}
    results = {name: evaluate_policy(greedy_fn_from_module(module, kind), market, test_seeds)
               for name, module in modules.items()}
    if sampled:
        for name, module in modules.items():
            results[f"{name}_sample"] = evaluate_sampled(module, market, test_seeds)
    return results


def summary_test_columns(prefix, result):
    """Return the summary columns of one test evaluation, every name starting with its prefix."""
    columns = {f"{prefix}return_mean": result["return_mean"], f"{prefix}return_std": result["return_std"]}
    for metric in SUMMARY_TEST_METRICS:
        columns[f"{prefix}{metric}_mean"] = result[f"{SUMMARY_METRIC_KEYS.get(metric, metric)}_mean"]
    return columns


def summary_row(algo, exp_id, description, config, tracker, test_results):
    """Build the one-line summary of a finished experiment for logs/summary.csv, the same for every algorithm."""
    row = {"algo": algo, "exp_id": exp_id, "description": description}
    row.update({key: str(value) if isinstance(value, list) else value for key, value in config.items()})
    row.update({
        "frames": tracker["frames"],
        "wall_time_s": round(elapsed_seconds(tracker), 1),
        "val_final_return": tracker["last_val"]["return_mean"],
        "val_best_return": tracker["best_val"]["return_mean"],
        "val_best_frames": tracker["best_val_frames"],
    })
    for name, result in test_results.items():
        row.update(summary_test_columns(f"test_{name}_", result))
    return row


def finished_exp_ids(algo, log_dir="logs"):
    """Return the exp_ids of one algorithm that already have a row in summary.csv."""
    rows, _ = read_rows(Path(log_dir) / "summary.csv")
    return {row["exp_id"] for row in rows if row.get("algo") == algo}


def finished_episode_returns(batch):
    """Return the returns of the episodes that ended inside a collected batch."""
    done = batch["next", "done"].squeeze(-1)
    return batch["next", "episode_reward"].squeeze(-1)[done].tolist()


def collector_frames(config):
    """Round total frames up to whole batches, which is what the collector would collect anyway."""
    batches = math.ceil(config["total_frames"] / config["frames_per_batch"])
    return batches * config["frames_per_batch"]
