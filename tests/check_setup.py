import sys

import gymnasium
import numpy
import tensordict
import torch
import torchrl
from torchrl.envs import GymEnv
from torchrl.modules import MLP, QValueActor
from torchrl.objectives import A2CLoss, ClipPPOLoss, DQNLoss, ReinforceLoss


def check(name, condition):
    """Print 'ok' or 'FAILED' for one named check and return whether it passed."""
    print(f"{'ok' if condition else 'FAILED'}: {name}")
    return condition


def main():
    """Print library versions, then confirm TorchRL can step a Gymnasium env and build a DQN loss.

    This catches install problems early, before any real project code depends on the stack.
    """
    print("python", sys.version.split()[0])
    print("torch", torch.__version__, "| torchrl", torchrl.__version__,
          "| tensordict", tensordict.__version__)
    print("gymnasium", gymnasium.__version__, "| numpy", numpy.__version__)

    results = []
    results.append(check("python is 3.11", sys.version_info[:2] == (3, 11)))
    results.append(check("numpy is below 2", int(numpy.__version__.split(".")[0]) < 2))
    results.append(check("torch tensor math works", torch.ones(3).sum().item() == 3))

    env = GymEnv("CartPole-v1")
    rollout = env.rollout(10)
    results.append(check("torchrl steps a gymnasium env", rollout.batch_size[0] > 0))

    net = MLP(out_features=2, num_cells=[32])
    actor = QValueActor(net, in_keys=["observation"], spec=env.action_spec)
    actor(rollout)
    loss = DQNLoss(actor, action_space=env.action_spec)
    results.append(check("DQN loss builds", loss is not None))
    results.append(check("PPO, A2C and REINFORCE losses import",
                         all([ClipPPOLoss, A2CLoss, ReinforceLoss])))

    print("all checks passed" if all(results) else "some checks FAILED")


if __name__ == "__main__":
    main()
