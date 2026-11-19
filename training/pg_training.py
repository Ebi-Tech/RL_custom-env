"""Policy-gradient training (REINFORCE, A2C and PPO) for the Beyond Sale Shop environment with TorchRL 0.3.1.

Run one sweep entry:  uv run python -m training.pg_training --algo ppo --exp-id ppo_03
Run a sweep:          uv run python -m training.pg_training --algo a2c --sweep [--only a2c_02,a2c_05] [--skip-done]
Ad hoc run:           uv run python -m training.pg_training --algo ppo --exp-id my_test --total-frames 20000
"""
import argparse
import copy

import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.utils import distance_loss
from torchrl.objectives.value import GAE

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
    evaluate_sampled,
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
    val_sample_columns,
)
from training.experiments import find_experiment, select_experiments

ALGOS = ("reinforce", "a2c", "ppo")
REINFORCE_BASELINES = ("none", "normalize", "value")
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}
ADVANTAGE_EPS = 1e-8
TRAIN_STATS_IN_EVAL = ("entropy", "approx_kl", "clip_fraction")
TOP_ACTIONS_SHOWN = 4

DEFAULT_REINFORCE_CONFIG = {
    "market": "default",
    "total_frames": 200000,
    "frames_per_batch": 450,
    "lr": 1e-3,
    "gamma": 0.99,
    "baseline": "normalize",
    "entropy_coef": 0.01,
    "critic_coef": 0.5,
    "loss_critic_type": "smooth_l1",
    "grad_clip": 1.0,
    "hidden": [128, 128],
    "activation": "relu",
    "eval_every_frames": 5000,
    "seed": 0,
}

DEFAULT_A2C_CONFIG = {
    "market": "default",
    "total_frames": 200000,
    "frames_per_batch": 90,
    "lr": 7e-4,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "entropy_coef": 0.01,
    "critic_coef": 0.5,
    "loss_critic_type": "smooth_l1",
    "normalize_advantage": False,
    "grad_clip": 0.5,
    "hidden": [128, 128],
    "activation": "relu",
    "eval_every_frames": 5000,
    "seed": 0,
}

DEFAULT_PPO_CONFIG = {
    "market": "default",
    "total_frames": 200000,
    "frames_per_batch": 900,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "ppo_epochs": 4,
    "minibatch_size": 128,
    "entropy_coef": 0.01,
    "critic_coef": 0.5,
    "loss_critic_type": "smooth_l1",
    "normalize_advantage": True,
    "grad_clip": 0.5,
    "hidden": [128, 128],
    "activation": "relu",
    "eval_every_frames": 5000,
    "seed": 0,
}

DEFAULT_CONFIGS = {"reinforce": DEFAULT_REINFORCE_CONFIG, "a2c": DEFAULT_A2C_CONFIG, "ppo": DEFAULT_PPO_CONFIG}


def merge_config(algo, overrides):
    """Return a copy of the algorithm's default config with overrides applied, rejecting unknown keys."""
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo {algo!r}, expected one of {ALGOS}")
    config = copy.deepcopy(DEFAULT_CONFIGS[algo])
    unknown = set(overrides or {}) - set(config)
    if unknown:
        raise ValueError(f"Unknown {algo} config keys: {sorted(unknown)}")
    config.update(overrides or {})
    if algo == "reinforce" and config["baseline"] not in REINFORCE_BASELINES:
        raise ValueError(f"baseline must be one of {REINFORCE_BASELINES}")
    return config


def uses_critic(algo, config):
    """Return whether this algorithm and config train a value network."""
    return algo in ("a2c", "ppo") or config["baseline"] == "value"


def build_mlp(config, out_features):
    """Build an MLP from the observation to out_features with the configured hidden layers."""
    return MLP(
        in_features=OBS_SIZE,
        out_features=out_features,
        num_cells=list(config["hidden"]),
        activation_class=ACTIVATIONS[config["activation"]],
    )


def build_logits_net(config):
    """Build the actor MLP that maps one observation to one logit per action."""
    return build_mlp(config, len(ACTION_NAMES))


def build_critic_net(config):
    """Build the critic MLP that maps one observation to one state value."""
    return build_mlp(config, 1)


def build_actor(logits_net, env):
    """Wrap the logits MLP in a ProbabilisticActor that samples one-hot actions and stores their log probs."""
    logits_module = TensorDictModule(logits_net, in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(
        logits_module,
        in_keys=["logits"],
        spec=env.action_spec,
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )


def build_agent(algo, config, env):
    """Build the actor, the critic where needed, the loss and advantage modules, and the optimiser."""
    logits_net = build_logits_net(config)
    agent = {"logits_net": logits_net, "actor": build_actor(logits_net, env), "critic_net": None,
             "critic": None, "loss_module": None, "advantage_module": None}
    if uses_critic(algo, config):
        agent["critic_net"] = build_critic_net(config)
        agent["critic"] = ValueOperator(agent["critic_net"], in_keys=["observation"])
    if algo in ("a2c", "ppo"):
        agent["loss_module"] = build_loss_module(algo, config, agent)
        agent["advantage_module"] = build_advantage_module(config, agent["critic"])
    agent["params"] = trainable_parameters(agent)
    agent["optimizer"] = torch.optim.Adam(agent["params"], lr=config["lr"])
    return agent


def build_advantage_module(config, critic):
    """Return the GAE module, using TorchRL's loop version when gae_lambda is 0.

    In TorchRL 0.3.1 the fast vectorised GAE crashes at lambda 0, because its (gamma * lambda)^k
    filter is cut to nothing; the loop version gives exactly the one step TD advantage there.
    """
    return GAE(gamma=config["gamma"], lmbda=config["gae_lambda"], value_network=critic,
               vectorized=config["gae_lambda"] > 0)


def build_loss_module(algo, config, agent):
    """Return the TorchRL A2C or clipped PPO loss for the actor and critic."""
    shared = {
        "entropy_coef": config["entropy_coef"],
        "critic_coef": config["critic_coef"],
        "loss_critic_type": config["loss_critic_type"],
    }
    if algo == "a2c":
        return A2CLoss(agent["actor"], agent["critic"], **shared)
    return ClipPPOLoss(agent["actor"], agent["critic"], clip_epsilon=config["clip_epsilon"],
                       normalize_advantage=config["normalize_advantage"], **shared)


def trainable_parameters(agent):
    """Return the parameters the optimiser updates: the loss module's, or the actor's and critic's directly."""
    if agent["loss_module"] is not None:
        return list(agent["loss_module"].parameters())
    params = list(agent["logits_net"].parameters())
    if agent["critic_net"] is not None:
        params += list(agent["critic_net"].parameters())
    return params


def build_collector(algo, config, env, actor):
    """Build the collector; REINFORCE restarts episodes each batch so every batch holds whole episodes."""
    return SyncDataCollector(
        env,
        actor,
        frames_per_batch=config["frames_per_batch"],
        total_frames=collector_frames(config),
        reset_at_each_iter=algo == "reinforce",
    )


def optimise(agent, loss, config):
    """Take one clipped gradient step on a scalar loss."""
    agent["optimizer"].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent["params"], config["grad_clip"])
    agent["optimizer"].step()


def returns_to_go(batch, gamma):
    """Return the discounted return-to-go of every step, and a mask of steps inside complete episodes.

    Steps after the last episode end in the batch belong to an unfinished episode, whose
    return is unknown, so they are masked out of the REINFORCE update.
    """
    rewards = batch["next", "reward"].squeeze(-1)
    dones = batch["next", "done"].squeeze(-1)
    returns = torch.zeros_like(rewards)
    running = torch.zeros(())
    for step in reversed(range(rewards.shape[0])):
        if dones[step]:
            running = torch.zeros(())
        running = rewards[step] + gamma * running
        returns[step] = running
    complete = torch.zeros_like(dones)
    ends = dones.nonzero().flatten()
    if ends.numel():
        complete[: int(ends[-1]) + 1] = True
    return returns, complete


def normalise(values):
    """Scale values to zero mean and unit standard deviation, a common variance reduction for advantages."""
    if values.numel() < 2:
        return values - values.mean()
    return (values - values.mean()) / (values.std() + ADVANTAGE_EPS)


def reinforce_advantage(agent, steps, returns, config):
    """Apply the configured REINFORCE baseline and return the advantage and the critic loss, if any."""
    if config["baseline"] == "none":
        return returns, None
    if config["baseline"] == "normalize":
        return normalise(returns), None
    values = agent["critic_net"](steps["observation"]).squeeze(-1)
    critic_loss = distance_loss(values, returns, config["loss_critic_type"]).mean()
    return returns - values.detach(), critic_loss


def reinforce_update(agent, batch, config):
    """Run one REINFORCE step: minus log prob times advantage, minus the entropy bonus, plus any critic loss."""
    returns, complete = returns_to_go(batch, config["gamma"])
    if not complete.any():
        return None
    steps = batch[complete]
    dist = agent["actor"].get_dist(steps.select("observation"))
    log_prob = dist.log_prob(steps["action"])
    entropy = dist.entropy().mean()
    advantage, critic_loss = reinforce_advantage(agent, steps, returns[complete], config)
    actor_loss = -(log_prob * advantage).mean()
    loss = actor_loss - config["entropy_coef"] * entropy
    if critic_loss is not None:
        loss = loss + config["critic_coef"] * critic_loss
    optimise(agent, loss, config)
    return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item() if critic_loss is not None else "",
            "entropy": entropy.item()}


def compute_advantages(agent, batch, config):
    """Fill in GAE advantages and value targets for a whole rollout, without gradients."""
    with torch.no_grad():
        agent["advantage_module"](batch)


def loss_step(agent, data, config):
    """Run the TorchRL loss on some data, take one gradient step, and return the logged loss parts."""
    out = agent["loss_module"](data)
    optimise(agent, out["loss_objective"] + out["loss_critic"] + out["loss_entropy"], config)
    return {"actor_loss": out["loss_objective"].item(), "critic_loss": out["loss_critic"].item(),
            "entropy": out["entropy"].item()}


def a2c_update(agent, batch, config):
    """Run one A2C step on the rollout, normalising the advantage first when configured."""
    compute_advantages(agent, batch, config)
    if config["normalize_advantage"]:
        batch.set("advantage", normalise(batch["advantage"]))
    return loss_step(agent, batch, config)


def ppo_diagnostics(agent, minibatch, config):
    """Return the approximate KL and the clip fraction of a minibatch before its gradient step.

    ClipPPOLoss in TorchRL 0.3.1 does not report them, so they are computed here from the new and
    old log probs, using the low variance KL estimate mean((ratio - 1) - log ratio).
    """
    with torch.no_grad():
        dist = agent["actor"].get_dist(minibatch.select("observation"))
        log_ratio = dist.log_prob(minibatch["action"]) - minibatch["sample_log_prob"]
        ratio = log_ratio.exp()
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > config["clip_epsilon"]).float().mean()
    return {"approx_kl": approx_kl.item(), "clip_fraction": clip_fraction.item()}


def ppo_update(agent, batch, config):
    """Compute GAE once for the rollout, then run the clipped PPO loss over shuffled minibatches for several epochs."""
    compute_advantages(agent, batch, config)
    flat = batch.reshape(-1)
    records = []
    for _ in range(config["ppo_epochs"]):
        order = torch.randperm(flat.shape[0])
        for start in range(0, flat.shape[0], config["minibatch_size"]):
            minibatch = flat[order[start:start + config["minibatch_size"]]]
            record = ppo_diagnostics(agent, minibatch, config)
            record.update(loss_step(agent, minibatch, config))
            records.append(record)
    return {key: sum(record[key] for record in records) / len(records) for key in records[0]}


UPDATE_FUNCTIONS = {"reinforce": reinforce_update, "a2c": a2c_update, "ppo": ppo_update}


def log_update(batch, stats, tracker, paths):
    """Write one train CSV row for this update, with the episodes that finished in its batch."""
    returns = finished_episode_returns(batch)
    tracker["episodes"] += len(returns)
    row = {
        "frames": tracker["frames"],
        "episodes": tracker["episodes"],
        "episodes_finished": len(returns),
        "mean_episode_return": sum(returns) / len(returns) if returns else "",
    }
    row.update(stats or {})
    if stats:
        tracker["last_stats"] = stats
    append_row(paths["train"], row)


def critic_extra(agent):
    """Return the critic weights to store in a checkpoint next to the actor, when there is a critic."""
    if agent["critic_net"] is None:
        return {}
    return {"critic_state_dict": agent["critic_net"].state_dict()}


def train_stat_columns(tracker):
    """Return the latest update's entropy, and PPO's approx_kl and clip_fraction, as train_ eval CSV columns."""
    stats = tracker["last_stats"]
    return {f"train_{key}": stats[key] for key in TRAIN_STATS_IN_EVAL if key in stats}


def run_validation(algo, agent, config, tracker, paths, val_seeds):
    """Evaluate the actor greedily and with sampled actions on the validation seeds, log both and print progress.

    The best checkpoint is chosen on the greedy return only.
    """
    greedy = evaluate_policy(greedy_fn_from_module(agent["logits_net"], "logits"), config["market"], val_seeds)
    sample = evaluate_sampled(agent["logits_net"], config["market"], val_seeds)
    extra = checkpoint_extra(greedy, tracker, algo=algo, val_sample_eval=sample, **critic_extra(agent))
    more_columns = val_sample_columns(sample)
    more_columns.update(train_stat_columns(tracker))
    record_validation(tracker, greedy, paths, agent["logits_net"], config, extra, more_columns)
    entropy = more_columns.get("train_entropy")
    entropy_text = "" if entropy is None else f"{entropy:.3f}"
    print(f"frames {tracker['frames']:>7}  val greedy {greedy['return_mean']:>8.2f}  "
          f"sample {sample['return_mean']:>8.2f}  entropy {entropy_text:>5}  {elapsed_seconds(tracker):>7.1f}s")


def top_actions(result, count=TOP_ACTIONS_SHOWN):
    """Return the most chosen actions of an evaluation as 'name share' text, most frequent first."""
    ranked = sorted(result["action_freq"].items(), key=lambda item: item[1], reverse=True)[:count]
    return ", ".join(f"{name} {share:.3f}" for name, share in ranked)


def print_test_results(test_results):
    """Print return, net worth, invalid rate, hold rate and top actions of every test evaluation."""
    for name, result in test_results.items():
        print(f"test {name:<12} return {result['return_mean']:>8.2f}  net worth {result['final_net_worth_mean']:>11.0f}  "
              f"invalid {result['invalid_rate_mean']:.3f}  hold {result['hold_rate_mean']:.3f}  "
              f"bankrupt {result['bankruptcies']}  top: {top_actions(result)}")


def load_pg_checkpoint(path):
    """Rebuild the actor, and the critic when one was saved, from a policy-gradient checkpoint."""
    data = torch.load(path)
    actor_net = build_logits_net(data["config"])
    actor_net.load_state_dict(data["state_dict"])
    critic_net = None
    if "critic_state_dict" in data["extra"]:
        critic_net = build_critic_net(data["config"])
        critic_net.load_state_dict(data["extra"]["critic_state_dict"])
    return actor_net, critic_net, data


def run_experiment(algo, exp_id, overrides=None, log_dir="logs", model_dir="models", val_seeds=VAL_SEEDS,
                   test_seeds=TEST_SEEDS, description="", summary_path=None):
    """Train one REINFORCE, A2C or PPO agent with the defaults plus overrides, log everything and save the models.

    Validation seeds pick the best checkpoint on the greedy return during training; after training
    the final and the best actors are scored on the test seeds both greedily and with sampled
    actions, and one row is added to logs/summary.csv, or to summary_path when given.
    """
    config = merge_config(algo, overrides)
    set_seed(config["seed"])
    paths = experiment_paths(exp_id, algo, f"pg/{algo}", log_dir, model_dir, summary_path)
    remove_old_logs(paths)
    env = make_env(config["market"], config["seed"])
    agent = build_agent(algo, config, env)
    collector = build_collector(algo, config, env, agent["actor"])
    tracker = new_tracker(config)
    tracker["last_stats"] = {}
    for batch in collector:
        tracker["frames"] += batch.numel()
        stats = UPDATE_FUNCTIONS[algo](agent, batch, config)
        log_update(batch, stats, tracker, paths)
        if tracker["frames"] >= tracker["next_eval"]:
            run_validation(algo, agent, config, tracker, paths, val_seeds)
            tracker["next_eval"] += config["eval_every_frames"]
    collector.shutdown()
    if tracker["last_val_frames"] != tracker["frames"]:
        run_validation(algo, agent, config, tracker, paths, val_seeds)
    final_extra = checkpoint_extra(tracker["last_val"], tracker, algo=algo, **critic_extra(agent))
    save_checkpoint(paths["final"], agent["logits_net"], config, final_extra)
    test_results = evaluate_on_test(agent["logits_net"], paths["best"], build_logits_net, "logits",
                                    config["market"], test_seeds, sampled=True)
    append_row(paths["summary"], summary_row(algo, exp_id, description, config, tracker, test_results))
    print_test_results(test_results)
    return {"logits_net": agent["logits_net"], "critic_net": agent["critic_net"], "val_final": tracker["last_val"],
            "val_best": tracker["best_val"], "test": test_results, "paths": paths, "frames": tracker["frames"],
            "wall_time_s": elapsed_seconds(tracker)}


def parse_args():
    """Read the command line options for one policy-gradient run."""
    parser = argparse.ArgumentParser(description="Train REINFORCE, A2C or PPO on the Beyond Sale Shop environment.")
    parser.add_argument("--algo", required=True, choices=ALGOS)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--exp-id", help="a sweep id such as ppo_03, or any new id for an ad hoc run")
    target.add_argument("--sweep", action="store_true", help="run every experiment in the algorithm's sweep list")
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
    """Return the experiments to run: one by id, the algorithm's whole sweep, or the --only subset of it."""
    if args.exp_id:
        known = find_experiment(args.exp_id, args.algo)
        return [known or {"exp_id": args.exp_id, "description": "ad hoc run", "overrides": {}}]
    return select_experiments(args.algo, args.only)


def run_one(algo, experiment, extra_overrides, summary_path=None):
    """Run one experiment entry with any command line overrides on top and print its result."""
    overrides = {**experiment["overrides"], **extra_overrides}
    print(f"\n=== {algo} {experiment['exp_id']}: {experiment['description']} ===")
    result = run_experiment(algo, experiment["exp_id"], overrides, description=experiment["description"],
                            summary_path=summary_path)
    print(f"done: {result['frames']} frames in {result['wall_time_s']:.1f}s, "
          f"val return final {result['val_final']['return_mean']:.2f}, best {result['val_best']['return_mean']:.2f}")


def main():
    """Run the experiments chosen on the command line, skipping finished ones when asked."""
    args = parse_args()
    if args.only and not args.sweep:
        raise SystemExit("--only needs --sweep")
    done = finished_exp_ids(args.algo) if args.skip_done else set()
    for experiment in selected_experiments(args):
        if experiment["exp_id"] in done:
            print(f"skipping {experiment['exp_id']}, already in summary.csv")
            continue
        run_one(args.algo, experiment, cli_overrides(args), args.summary_path)


if __name__ == "__main__":
    main()
