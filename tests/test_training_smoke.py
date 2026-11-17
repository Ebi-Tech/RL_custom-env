"""Smoke test: a tiny DQN run finishes, writes its logs and checkpoints, and reloads to the same policy."""
import torch

from environment import ShopEnv
from training.common import greedy_fn_from_module, load_checkpoint
from training.dqn_training import build_value_net, run_experiment

SMOKE_OVERRIDES = {"total_frames": 1000, "init_random_frames": 200, "eval_every_frames": 500}
SMOKE_EVAL_SEEDS = range(1000, 1002)
SAMPLE_STEPS = 90


def sample_observations(steps=SAMPLE_STEPS):
    """Collect observations from one random episode to compare policies on."""
    env = ShopEnv()
    obs, _ = env.reset(seed=0)
    env.action_space.seed(0)
    observations = [obs]
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        observations.append(obs)
        if terminated or truncated:
            break
    return observations


def test_dqn_smoke_run(tmp_path):
    """A 1000 frame DQN run writes both CSVs and both checkpoints, and the reload acts identically."""
    result = run_experiment(
        "smoke",
        SMOKE_OVERRIDES,
        log_dir=tmp_path / "logs",
        model_dir=tmp_path / "models",
        eval_seeds=SMOKE_EVAL_SEEDS,
    )
    paths = result["paths"]
    for key in ("train", "eval", "summary", "final", "best"):
        assert paths[key].exists(), key
    reloaded, data = load_checkpoint(paths["final"], build_value_net)
    assert data["config"]["total_frames"] == SMOKE_OVERRIDES["total_frames"]
    trained_fn = greedy_fn_from_module(result["value_net"], "q")
    reloaded_fn = greedy_fn_from_module(reloaded, "q")
    for obs in sample_observations():
        assert trained_fn(obs, None) == reloaded_fn(obs, None)
    with torch.no_grad():
        obs = torch.as_tensor(sample_observations()[0])
        assert torch.allclose(result["value_net"](obs), reloaded(obs))
