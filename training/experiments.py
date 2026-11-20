"""The hyperparameter sweeps: each experiment changes exactly one setting from its algorithm's default config."""
import re

SEED_COPY = re.compile(r"^(?P<base>.+)_s(?P<seed>\d+)$")

DQN_EXPERIMENTS = [
    {"exp_id": "dqn_01", "description": "baseline, default config", "overrides": {}},
    {"exp_id": "dqn_02", "description": "lower learning rate 3e-4", "overrides": {"lr": 3e-4}},
    {"exp_id": "dqn_03", "description": "higher learning rate 3e-3", "overrides": {"lr": 3e-3}},
    {"exp_id": "dqn_04", "description": "shorter horizon, gamma 0.95", "overrides": {"gamma": 0.95}},
    {"exp_id": "dqn_05", "description": "longer horizon, gamma 0.999", "overrides": {"gamma": 0.999}},
    {"exp_id": "dqn_06", "description": "smaller replay buffer 10000", "overrides": {"buffer_size": 10000}},
    {"exp_id": "dqn_07", "description": "larger batch size 128", "overrides": {"batch_size": 128}},
    {"exp_id": "dqn_08", "description": "slower epsilon decay over 60000 frames",
     "overrides": {"eps_decay_frames": 60000}},
    {"exp_id": "dqn_09", "description": "faster epsilon decay over 5000 frames",
     "overrides": {"eps_decay_frames": 5000}},
    {"exp_id": "dqn_10", "description": "soft target update, tau 0.005",
     "overrides": {"target_update": "soft", "soft_tau": 0.005}},
    {"exp_id": "dqn_11", "description": "hard target update every 100 updates",
     "overrides": {"target_update": "hard", "hard_update_interval": 100}},
    {"exp_id": "dqn_12", "description": "double DQN", "overrides": {"double_dqn": True}},
    {"exp_id": "dqn_13", "description": "L2 loss instead of smooth L1", "overrides": {"loss_function": "l2"}},
    {"exp_id": "dqn_14", "description": "smaller network [64, 64]", "overrides": {"hidden": [64, 64]}},
    {"exp_id": "dqn_15", "description": "fewer updates per batch, 2", "overrides": {"updates_per_batch": 2}},
]


REINFORCE_EXPERIMENTS = [
    {"exp_id": "reinforce_01", "description": "baseline, default config", "overrides": {}},
    {"exp_id": "reinforce_02", "description": "lower learning rate 3e-4", "overrides": {"lr": 3e-4}},
    {"exp_id": "reinforce_03", "description": "higher learning rate 3e-3", "overrides": {"lr": 3e-3}},
    {"exp_id": "reinforce_04", "description": "shorter horizon, gamma 0.95", "overrides": {"gamma": 0.95}},
    {"exp_id": "reinforce_05", "description": "longer horizon, gamma 0.999", "overrides": {"gamma": 0.999}},
    {"exp_id": "reinforce_06", "description": "no baseline", "overrides": {"baseline": "none"}},
    {"exp_id": "reinforce_07", "description": "learned value baseline", "overrides": {"baseline": "value"}},
    {"exp_id": "reinforce_08", "description": "no entropy bonus", "overrides": {"entropy_coef": 0.0}},
    {"exp_id": "reinforce_09", "description": "larger entropy bonus 0.05", "overrides": {"entropy_coef": 0.05}},
    {"exp_id": "reinforce_10", "description": "one episode per batch, 90 frames",
     "overrides": {"frames_per_batch": 90}},
    {"exp_id": "reinforce_11", "description": "twenty episodes per batch, 1800 frames",
     "overrides": {"frames_per_batch": 1800}},
    {"exp_id": "reinforce_12", "description": "smaller network [64, 64]", "overrides": {"hidden": [64, 64]}},
]

A2C_EXPERIMENTS = [
    {"exp_id": "a2c_01", "description": "baseline, default config", "overrides": {}},
    {"exp_id": "a2c_02", "description": "lower learning rate 2e-4", "overrides": {"lr": 2e-4}},
    {"exp_id": "a2c_03", "description": "higher learning rate 2e-3", "overrides": {"lr": 2e-3}},
    {"exp_id": "a2c_04", "description": "longer horizon, gamma 0.999", "overrides": {"gamma": 0.999}},
    {"exp_id": "a2c_05", "description": "GAE lambda 0.95", "overrides": {"gae_lambda": 0.95}},
    {"exp_id": "a2c_06", "description": "one step TD advantage, GAE lambda 0.0", "overrides": {"gae_lambda": 0.0}},
    {"exp_id": "a2c_07", "description": "no entropy bonus", "overrides": {"entropy_coef": 0.0}},
    {"exp_id": "a2c_08", "description": "larger entropy bonus 0.05", "overrides": {"entropy_coef": 0.05}},
    {"exp_id": "a2c_09", "description": "critic coefficient 1.0", "overrides": {"critic_coef": 1.0}},
    {"exp_id": "a2c_10", "description": "longer rollouts, 450 frames", "overrides": {"frames_per_batch": 450}},
    {"exp_id": "a2c_11", "description": "normalised advantages", "overrides": {"normalize_advantage": True}},
    {"exp_id": "a2c_12", "description": "L2 critic loss", "overrides": {"loss_critic_type": "l2"}},
]

PPO_EXPERIMENTS = [
    {"exp_id": "ppo_01", "description": "baseline, default config", "overrides": {}},
    {"exp_id": "ppo_02", "description": "lower learning rate 1e-4", "overrides": {"lr": 1e-4}},
    {"exp_id": "ppo_03", "description": "higher learning rate 1e-3", "overrides": {"lr": 1e-3}},
    {"exp_id": "ppo_04", "description": "longer horizon, gamma 0.999", "overrides": {"gamma": 0.999}},
    {"exp_id": "ppo_05", "description": "tighter clip 0.1", "overrides": {"clip_epsilon": 0.1}},
    {"exp_id": "ppo_06", "description": "looser clip 0.3", "overrides": {"clip_epsilon": 0.3}},
    {"exp_id": "ppo_07", "description": "no entropy bonus", "overrides": {"entropy_coef": 0.0}},
    {"exp_id": "ppo_08", "description": "larger entropy bonus 0.05", "overrides": {"entropy_coef": 0.05}},
    {"exp_id": "ppo_09", "description": "more epochs per rollout, 10", "overrides": {"ppo_epochs": 10}},
    {"exp_id": "ppo_10", "description": "longer rollouts, 2700 frames", "overrides": {"frames_per_batch": 2700}},
    {"exp_id": "ppo_11", "description": "GAE lambda 0.8", "overrides": {"gae_lambda": 0.8}},
    {"exp_id": "ppo_12", "description": "larger minibatch 512", "overrides": {"minibatch_size": 512}},
]

EXPERIMENTS = {
    "dqn": DQN_EXPERIMENTS,
    "reinforce": REINFORCE_EXPERIMENTS,
    "a2c": A2C_EXPERIMENTS,
    "ppo": PPO_EXPERIMENTS,
}


def find_experiment(exp_id, algo=None):
    """Return the sweep entry with this exp_id, from one algorithm's list or from any, or None for an ad hoc run.

    An id such as ppo_03_s1 is a seed copy: the same entry as ppo_03 with only the seed changed to 1.
    """
    lists = [EXPERIMENTS[algo]] if algo else EXPERIMENTS.values()
    for experiments in lists:
        for experiment in experiments:
            if experiment["exp_id"] == exp_id:
                return experiment
    match = SEED_COPY.match(exp_id)
    if match:
        base = find_experiment(match.group("base"), algo)
        if base:
            return seed_copy(base, int(match.group("seed")))
    return None


def seed_copy(experiment, seed):
    """Return a copy of a sweep entry that differs only in its seed, with the id suffixed _s<seed>."""
    return {
        "exp_id": f"{experiment['exp_id']}_s{seed}",
        "description": f"{experiment['description']}, seed {seed}",
        "overrides": {**experiment["overrides"], "seed": seed},
    }


def select_experiments(algo, only=None):
    """Return one algorithm's sweep list, or just the comma separated ids in only, rejecting unknown ids."""
    if not only:
        return list(EXPERIMENTS[algo])
    wanted = [exp_id.strip() for exp_id in only.split(",") if exp_id.strip()]
    unknown = [exp_id for exp_id in wanted if find_experiment(exp_id, algo) is None]
    if unknown:
        raise SystemExit(f"Unknown {algo} experiment ids: {unknown}")
    return [find_experiment(exp_id, algo) for exp_id in wanted]


def changed_settings(defaults, overrides):
    """Return the names of the settings an experiment really changes from the defaults."""
    return sorted(key for key, value in overrides.items() if defaults.get(key) != value)
