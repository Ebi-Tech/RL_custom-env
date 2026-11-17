"""The DQN hyperparameter sweep: each experiment changes exactly one setting from DEFAULT_DQN_CONFIG."""

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


def find_experiment(exp_id):
    """Return the sweep entry with this exp_id, or None for an ad hoc run."""
    for experiment in DQN_EXPERIMENTS:
        if experiment["exp_id"] == exp_id:
            return experiment
    return None
