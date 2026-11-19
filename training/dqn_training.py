"""DQN training for the Beyond Sale Shop environment with TorchRL 0.3.1.

Run one sweep entry:  uv run python -m training.dqn_training --exp-id dqn_03
Run the sweep:        uv run python -m training.dqn_training --sweep [--only dqn_02,dqn_05] [--skip-done]
Ad hoc run:           uv run python -m training.dqn_training --exp-id my_test --total-frames 20000
"""
import argparse
import copy

import torch
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from environment.custom_env import ACTION_NAMES, OBS_SIZE
from training.common import (
    TEST_SEEDS,
    VAL_SEEDS,
    append_row,
    checkpoint_extra,
    collector_frames,
    elapsed_seconds,
    evaluate_on_test,
    evaluate_policy,
    experiment_paths,
    finished_episode_returns,
    finished_exp_ids,
    greedy_fn_from_module,
    make_env,
    new_tracker,
    record_validation,
    remove_old_logs,
    save_checkpoint,
    set_seed,
    summary_row,
)
from training.experiments import find_experiment, select_experiments

ALGO = "dqn"
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}
TARGET_UPDATES = ("hard", "soft")

DEFAULT_DQN_CONFIG = {
    "market": "default",
    "total_frames": 200000,
    "frames_per_batch": 90,
    "init_random_frames": 2000,
    "buffer_size": 50000,
    "batch_size": 64,
    "lr": 1e-3,
    "gamma": 0.99,
    "hidden": [128, 128],
    "activation": "relu",
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_frames": 20000,
    "target_update": "hard",
    "hard_update_interval": 500,
    "soft_tau": 0.005,
    "double_dqn": False,
    "loss_function": "smooth_l1",
    "updates_per_batch": 10,
    "grad_clip": 10.0,
    "eval_every_frames": 5000,
    "seed": 0,
}


def merge_config(overrides):
    """Return a copy of the default config with overrides applied, rejecting unknown keys."""
    config = copy.deepcopy(DEFAULT_DQN_CONFIG)
    unknown = set(overrides or {}) - set(config)
    if unknown:
        raise ValueError(f"Unknown DQN config keys: {sorted(unknown)}")
    config.update(overrides or {})
    if config["target_update"] not in TARGET_UPDATES:
        raise ValueError(f"target_update must be one of {TARGET_UPDATES}")
    return config


def build_value_net(config):
    """Build the MLP that maps one observation to one Q-value per action."""
    return MLP(
        in_features=OBS_SIZE,
        out_features=len(ACTION_NAMES),
        num_cells=list(config["hidden"]),
        activation_class=ACTIVATIONS[config["activation"]],
    )


def build_agent(config, env):
    """Build the Q actor, the epsilon-greedy exploration policy, the DQN loss and the target updater."""
    value_net = build_value_net(config)
    actor = QValueActor(value_net, in_keys=["observation"], spec=env.action_spec)
    exploration = EGreedyModule(
        spec=env.action_spec,
        eps_init=float(config["eps_start"]),
        eps_end=float(config["eps_end"]),
        annealing_num_steps=config["eps_decay_frames"],
    )
    loss_module = DQNLoss(
        actor,
        loss_function=config["loss_function"],
        delay_value=True,
        double_dqn=config["double_dqn"],
    )
    loss_module.make_value_estimator(gamma=config["gamma"])
    return {
        "value_net": value_net,
        "actor": actor,
        "exploration": exploration,
        "policy": TensorDictSequential(actor, exploration),
        "loss_module": loss_module,
        "target_updater": build_target_updater(config, loss_module),
        "optimizer": torch.optim.Adam(loss_module.parameters(), lr=config["lr"]),
    }


def build_target_updater(config, loss_module):
    """Return a hard or soft target network updater as chosen in the config."""
    if config["target_update"] == "soft":
        return SoftUpdate(loss_module, tau=config["soft_tau"])
    return HardUpdate(loss_module, value_network_update_interval=config["hard_update_interval"])


def build_collector(config, env, policy):
    """Build the synchronous collector that gathers one batch of transitions at a time."""
    return SyncDataCollector(
        env,
        policy,
        frames_per_batch=config["frames_per_batch"],
        total_frames=collector_frames(config),
    )


def build_buffer(config):
    """Build the uniform replay buffer that stores transitions as tensordicts."""
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(config["buffer_size"]),
        batch_size=config["batch_size"],
    )


def gradient_step(agent, buffer, config):
    """Run one DQN update on a sampled batch and return the loss and the mean Q of the taken actions."""
    sample = buffer.sample()
    loss = agent["loss_module"](sample)["loss"]
    agent["optimizer"].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent["loss_module"].parameters(), config["grad_clip"])
    agent["optimizer"].step()
    agent["target_updater"].step()
    with torch.no_grad():
        q_values = agent["value_net"](sample["observation"])
        mean_q = (q_values * sample["action"]).sum(-1).mean()
    return loss.item(), mean_q.item()


def train_on_batch(agent, buffer, config, tracker):
    """Run the configured number of gradient steps and add their losses and Q-values to the tracker."""
    for _ in range(config["updates_per_batch"]):
        loss, mean_q = gradient_step(agent, buffer, config)
        tracker["losses"].append(loss)
        tracker["q_values"].append(mean_q)


def log_finished_episodes(batch, tracker, agent, paths):
    """Write one train CSV row per episode that ended in this batch."""
    for episode_return in finished_episode_returns(batch):
        tracker["episodes"] += 1
        append_row(paths["train"], {
            "frames": tracker["frames"],
            "episode": tracker["episodes"],
            "episode_return": episode_return,
            "mean_loss": mean_or_empty(tracker["losses"]),
            "epsilon": float(agent["exploration"].eps),
            "mean_q": mean_or_empty(tracker["q_values"]),
        })
        tracker["losses"].clear()
        tracker["q_values"].clear()


def mean_or_empty(values):
    """Return the mean of a list, or an empty string when no updates have happened yet."""
    return sum(values) / len(values) if values else ""


def run_validation(agent, config, tracker, paths, val_seeds):
    """Evaluate the greedy policy on the validation seeds, log it, keep the best model and print progress."""
    policy_fn = greedy_fn_from_module(agent["value_net"], "q")
    result = evaluate_policy(policy_fn, market=config["market"], seeds=val_seeds)
    record_validation(tracker, result, paths, agent["value_net"], config, checkpoint_extra(result, tracker))
    print(f"frames {tracker['frames']:>7}  val return {result['return_mean']:>8.2f}  "
          f"epsilon {float(agent['exploration'].eps):.3f}  {elapsed_seconds(tracker):>7.1f}s")


def run_experiment(exp_id, overrides=None, log_dir="logs", model_dir="models", val_seeds=VAL_SEEDS,
                   test_seeds=TEST_SEEDS, description="", summary_path=None):
    """Train one DQN agent with the defaults plus overrides, log everything and save the models.

    Validation seeds pick the best checkpoint during training; after training the final and the
    best networks are both scored on the test seeds. Returns the trained network, the results and
    the file paths, so a sweep can call it many times and tests can inspect what it wrote.
    summary_path, when given, receives the summary row instead of logs/summary.csv.
    """
    config = merge_config(overrides)
    set_seed(config["seed"])
    paths = experiment_paths(exp_id, ALGO, ALGO, log_dir, model_dir, summary_path)
    remove_old_logs(paths)
    env = make_env(config["market"], config["seed"])
    agent = build_agent(config, env)
    collector = build_collector(config, env, agent["policy"])
    buffer = build_buffer(config)
    tracker = new_tracker(config)
    tracker.update({"losses": [], "q_values": []})
    for batch in collector:
        tracker["frames"] += batch.numel()
        buffer.extend(batch.reshape(-1))
        if tracker["frames"] >= config["init_random_frames"]:
            train_on_batch(agent, buffer, config, tracker)
        agent["exploration"].step(batch.numel())
        log_finished_episodes(batch, tracker, agent, paths)
        if tracker["frames"] >= tracker["next_eval"]:
            run_validation(agent, config, tracker, paths, val_seeds)
            tracker["next_eval"] += config["eval_every_frames"]
    collector.shutdown()
    if tracker["last_val_frames"] != tracker["frames"]:
        run_validation(agent, config, tracker, paths, val_seeds)
    save_checkpoint(paths["final"], agent["value_net"], config, checkpoint_extra(tracker["last_val"], tracker))
    test_results = evaluate_on_test(agent["value_net"], paths["best"], build_value_net, "q",
                                    config["market"], test_seeds)
    append_row(paths["summary"], summary_row(ALGO, exp_id, description, config, tracker, test_results))
    print(f"test return: final {test_results['final']['return_mean']:.2f}, "
          f"best checkpoint {test_results['best']['return_mean']:.2f}")
    return {"value_net": agent["value_net"], "val_final": tracker["last_val"], "val_best": tracker["best_val"],
            "test": test_results, "paths": paths, "frames": tracker["frames"],
            "wall_time_s": elapsed_seconds(tracker)}


def parse_args():
    """Read the command line options for one run, a full sweep or a subset of the sweep."""
    parser = argparse.ArgumentParser(description="Train DQN on the Beyond Sale Shop environment.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--exp-id", help="a sweep id such as dqn_03, or any new id for an ad hoc run")
    target.add_argument("--sweep", action="store_true", help="run every experiment in DQN_EXPERIMENTS")
    parser.add_argument("--only", help="comma separated sweep ids to run, used with --sweep")
    parser.add_argument("--skip-done", action="store_true", help="skip ids that already have a summary row")
    parser.add_argument("--total-frames", type=int)
    parser.add_argument("--market", choices=["default", "harsh", "price_scale"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summary-path", help="write the summary row here instead of logs/summary.csv")
    return parser.parse_args()


def cli_overrides(args):
    """Return the config overrides given directly on the command line."""
    overrides = {"total_frames": args.total_frames, "market": args.market, "seed": args.seed}
    return {key: value for key, value in overrides.items() if value is not None}


def selected_experiments(args):
    """Return the experiments to run: one by id, the whole sweep, or the --only subset of it."""
    if args.exp_id:
        known = find_experiment(args.exp_id, ALGO)
        return [known or {"exp_id": args.exp_id, "description": "ad hoc run", "overrides": {}}]
    return select_experiments(ALGO, args.only)


def run_one(experiment, extra_overrides, summary_path=None):
    """Run one experiment entry with any command line overrides on top and print its result."""
    overrides = {**experiment["overrides"], **extra_overrides}
    print(f"\n=== {experiment['exp_id']}: {experiment['description']} ===")
    result = run_experiment(experiment["exp_id"], overrides, description=experiment["description"],
                            summary_path=summary_path)
    print(f"done: {result['frames']} frames in {result['wall_time_s']:.1f}s, "
          f"val return final {result['val_final']['return_mean']:.2f}, best {result['val_best']['return_mean']:.2f}")


def main():
    """Run the experiments chosen on the command line, skipping finished ones when asked."""
    args = parse_args()
    if args.only and not args.sweep:
        raise SystemExit("--only needs --sweep")
    done = finished_exp_ids(ALGO) if args.skip_done else set()
    for experiment in selected_experiments(args):
        if experiment["exp_id"] in done:
            print(f"skipping {experiment['exp_id']}, already in summary.csv")
            continue
        run_one(experiment, cli_overrides(args), args.summary_path)


if __name__ == "__main__":
    main()
