"""Smoke tests: tiny DQN, REINFORCE, A2C and PPO runs finish, write their logs and checkpoints, and reload identically."""
import pytest
import torch

from environment import ShopEnv
from training import pg_training
from training.common import evaluate_sampled, greedy_fn_from_module, load_checkpoint, read_rows
from training.dqn_training import build_value_net, run_experiment

PG_SMOKE_OVERRIDES = {"total_frames": 1000, "eval_every_frames": 500}
PG_ALGOS_WITH_CRITIC = ("a2c", "ppo")

SMOKE_OVERRIDES = {"total_frames": 1000, "init_random_frames": 200, "eval_every_frames": 500}
SMOKE_VAL_SEEDS = range(500, 502)
SMOKE_TEST_SEEDS = range(1000, 1002)
SAMPLE_STEPS = 90
ENTROPY_MAX = 2.5650
EXPECTED_SUMMARY_COLUMNS = (
    "val_best_return",
    "test_final_return_mean",
    "test_final_return_std",
    "test_best_return_mean",
    "test_best_return_std",
    "test_final_net_worth_mean",
    "test_best_net_worth_mean",
    "test_final_invalid_rate_mean",
    "test_best_hold_rate_mean",
    "test_final_trust_mean",
    "test_best_lost_per_day_mean",
    "test_final_noop_rate_mean",
)
PG_EXTRA_SUMMARY_COLUMNS = (
    "test_final_sample_return_mean",
    "test_final_sample_net_worth_mean",
    "test_best_sample_return_mean",
    "test_best_sample_invalid_rate_mean",
    "test_best_sample_hold_rate_mean",
)
PG_EVAL_COLUMNS = (
    "val_return_mean", "val_noop_rate_mean", "val_sample_return_mean", "val_sample_noop_rate_mean", "train_entropy",
)


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
    """A 1000 frame DQN run writes its CSVs, checkpoints and test columns, and the reload acts identically."""
    result = run_experiment(
        "smoke",
        SMOKE_OVERRIDES,
        log_dir=tmp_path / "logs",
        model_dir=tmp_path / "models",
        val_seeds=SMOKE_VAL_SEEDS,
        test_seeds=SMOKE_TEST_SEEDS,
    )
    paths = result["paths"]
    for key in ("train", "eval", "summary", "final", "best"):
        assert paths[key].exists(), key
    rows, header = read_rows(paths["summary"])
    assert len(rows) == 1
    for column in EXPECTED_SUMMARY_COLUMNS:
        assert column in header, column
    reloaded, data = load_checkpoint(paths["final"], build_value_net)
    assert data["config"]["total_frames"] == SMOKE_OVERRIDES["total_frames"]
    trained_fn = greedy_fn_from_module(result["value_net"], "q")
    reloaded_fn = greedy_fn_from_module(reloaded, "q")
    for obs in sample_observations():
        assert trained_fn(obs, None) == reloaded_fn(obs, None)
    with torch.no_grad():
        obs = torch.as_tensor(sample_observations()[0])
        assert torch.allclose(result["value_net"](obs), reloaded(obs))


def assert_same_greedy_actions(first_net, second_net):
    """Check that two logits networks pick the same greedy action on every sampled observation."""
    first_fn = greedy_fn_from_module(first_net, "logits")
    second_fn = greedy_fn_from_module(second_net, "logits")
    for obs in sample_observations():
        assert first_fn(obs, None) == second_fn(obs, None)


@pytest.mark.parametrize("algo", pg_training.ALGOS)
def test_pg_smoke_run(algo, tmp_path):
    """A tiny run of each policy-gradient algorithm writes its files, and the reloaded actor acts identically."""
    result = pg_training.run_experiment(
        algo,
        f"{algo}_smoke",
        PG_SMOKE_OVERRIDES,
        log_dir=tmp_path / "logs",
        model_dir=tmp_path / "models",
        val_seeds=SMOKE_VAL_SEEDS,
        test_seeds=SMOKE_TEST_SEEDS,
    )
    paths = result["paths"]
    for key in ("train", "eval", "summary", "final", "best"):
        assert paths[key].exists(), key
    assert paths["final"].parent == tmp_path / "models" / "pg" / algo
    rows, header = read_rows(paths["summary"])
    assert rows[0]["algo"] == algo
    for column in EXPECTED_SUMMARY_COLUMNS + PG_EXTRA_SUMMARY_COLUMNS:
        assert column in header, column
    _, eval_header = read_rows(paths["eval"])
    for column in PG_EVAL_COLUMNS:
        assert column in eval_header, column
    actor_net, critic_net, data = pg_training.load_pg_checkpoint(paths["final"])
    assert data["extra"]["algo"] == algo
    assert (critic_net is not None) == (algo in PG_ALGOS_WITH_CRITIC)
    assert_same_greedy_actions(result["logits_net"], actor_net)


def test_sampled_evaluation_repeats_exactly():
    """Sampled evaluation of the same actor on the same seeds gives identical results twice, without touching torch's global RNG."""
    torch.manual_seed(0)
    actor_net = pg_training.build_logits_net(pg_training.DEFAULT_PPO_CONFIG)
    rng_before = torch.get_rng_state()
    first = evaluate_sampled(actor_net, seeds=SMOKE_TEST_SEEDS)
    second = evaluate_sampled(actor_net, seeds=SMOKE_TEST_SEEDS)
    assert first == second
    assert torch.equal(rng_before, torch.get_rng_state())


def test_a2c_runs_with_gae_lambda_zero(tmp_path):
    """A2C with gae_lambda 0, the one step TD advantage of sweep entry a2c_06, trains without crashing."""
    overrides = {**PG_SMOKE_OVERRIDES, "gae_lambda": 0.0}
    result = pg_training.run_experiment("a2c", "a2c_lambda0", overrides, log_dir=tmp_path / "logs",
                                        model_dir=tmp_path / "models", val_seeds=SMOKE_VAL_SEEDS,
                                        test_seeds=SMOKE_TEST_SEEDS)
    assert result["paths"]["final"].exists()


def test_a2c_runs_with_entropy_coef_zero(tmp_path):
    """A2C with entropy_coef 0, sweep entry a2c_07, trains without crashing and still logs a real entropy."""
    overrides = {**PG_SMOKE_OVERRIDES, "entropy_coef": 0.0}
    result = pg_training.run_experiment("a2c", "a2c_entropy0", overrides, log_dir=tmp_path / "logs",
                                        model_dir=tmp_path / "models", val_seeds=SMOKE_VAL_SEEDS,
                                        test_seeds=SMOKE_TEST_SEEDS)
    rows, _ = read_rows(result["paths"]["train"])
    entropies = [float(row["entropy"]) for row in rows if row["entropy"]]
    assert entropies and all(0 < value <= ENTROPY_MAX for value in entropies)
