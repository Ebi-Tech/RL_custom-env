"""Checks on the sweep lists and on the order the sweep runner queues them in."""
import pytest

from training import pg_training
from training.dqn_training import DEFAULT_DQN_CONFIG
from training.dqn_training import merge_config as merge_dqn_config
from training.experiments import EXPERIMENTS, changed_settings
from training.run_sweeps import ALGO_ORDER, build_queue, interleave

DEFAULTS = {
    "dqn": DEFAULT_DQN_CONFIG,
    "reinforce": pg_training.DEFAULT_REINFORCE_CONFIG,
    "a2c": pg_training.DEFAULT_A2C_CONFIG,
    "ppo": pg_training.DEFAULT_PPO_CONFIG,
}
EXPECTED_COUNTS = {"dqn": 15, "reinforce": 12, "a2c": 12, "ppo": 12}


def merge_for(algo, overrides):
    """Merge overrides with an algorithm's defaults, which rejects unknown keys."""
    if algo == "dqn":
        return merge_dqn_config(overrides)
    return pg_training.merge_config(algo, overrides)


@pytest.mark.parametrize("algo", ALGO_ORDER)
def test_each_experiment_changes_exactly_one_setting(algo):
    """The first experiment is the default config and every other one changes exactly one setting."""
    experiments = EXPERIMENTS[algo]
    assert len(experiments) == EXPECTED_COUNTS[algo]
    for number, experiment in enumerate(experiments, start=1):
        assert experiment["exp_id"] == f"{algo}_{number:02d}"
        merge_for(algo, experiment["overrides"])
        changed = changed_settings(DEFAULTS[algo], experiment["overrides"])
        assert len(changed) == (0 if number == 1 else 1), (experiment["exp_id"], changed)


def test_experiment_ids_are_unique():
    """No exp_id appears twice across the four sweeps."""
    ids = [experiment["exp_id"] for experiments in EXPERIMENTS.values() for experiment in experiments]
    assert len(ids) == len(set(ids)) == sum(EXPECTED_COUNTS.values())


def test_queue_interleaves_algorithms_and_skips_done():
    """The queue takes one experiment from each algorithm in turn and leaves out finished ones."""
    queue = build_queue(list(ALGO_ORDER), None, {"reinforce_01"})
    ids = [job["exp_id"] for job in queue]
    assert ids[:4] == ["dqn_01", "reinforce_02", "a2c_01", "ppo_01"]
    assert "reinforce_01" not in ids
    assert len(ids) == sum(EXPECTED_COUNTS.values()) - 1
    assert interleave([[1, 2, 3], [4]]) == [1, 4, 2, 3]
