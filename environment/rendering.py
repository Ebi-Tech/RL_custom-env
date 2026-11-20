"""Turn ShopEnv episodes into JSON frames for the browser simulation, one frame per trading day.

The frames carry everything the 3D shop and its HUD show: the shop state, the day's sales,
haggles, credit and events, and the running result of rule_policy on the same seed, so the
agent can be compared with the rule shop side by side.
"""
import re
from pathlib import Path

import numpy as np

from environment.baselines import random_policy, rule_policy
from environment.custom_env import ACTION_NAMES, ShopEnv

DEFAULT_SEED = 1000
BUILTIN_POLICIES = {"rule": rule_policy, "random": random_policy}
PG_ALGOS = ("reinforce", "a2c", "ppo")
HAGGLE_EVENT = re.compile(r"^(?P<item>.+) haggle: (?:sold at (?P<price>\d+(?:\.\d+)?)|buyer walked away)$")
DISPLAY_WORDS = (("Indomie", "Noodle"), ("indomie", "noodles"))
PRODUCT_DISPLAY_NAMES = {
    "indomie": "Noodle packs",
    "soap": "Soap bars",
    "rice_bag": "Rice sacks",
    "loose_rice": "Loose rice (mudu)",
}


def display_text(text):
    """Replace brand words in env text with generic product words for everything the viewer sees."""
    for brand, generic in DISPLAY_WORDS:
        text = text.replace(brand, generic)
    return text


def action_label(action_name):
    """Return a readable, brand-free label for an action name, such as 'restock noodles small'."""
    return display_text(action_name).replace("_", " ")


def parse_haggles(events):
    """Return the day's haggles from the event list: item, outcome, and the price when a sale was made."""
    haggles = []
    for event in events:
        match = HAGGLE_EVENT.match(event)
        if not match:
            continue
        price = match.group("price")
        haggles.append({
            "item": display_text(match.group("item")),
            "outcome": "sold" if price else "walked_away",
            "price": float(price) if price else None,
        })
    return haggles


def product_sales(env, info):
    """Return the units sold today keyed by product."""
    keys = [product["key"] for product in env.unwrapped.products]
    return {key: float(units) for key, units in zip(keys, info["units_sold"])}


def frame_from_step(env, day_result):
    """Build one JSON-serialisable frame for a finished day.

    day_result holds the action, reward, cumulative return and info dict of that day.
    """
    info = day_result["info"]
    shop = env.unwrapped
    action_name = ACTION_NAMES[day_result["action"]]
    return {
        "day": int(info["day"]),
        "action": action_name,
        "action_label": action_label(action_name),
        "reward": float(day_result["reward"]),
        "cumulative_return": float(day_result["cumulative_return"]),
        "state": shop.get_state(),
        "net_worth": float(shop.net_worth()),
        "events": [display_text(event) for event in info["events"]],
        "customers": int(info["customers"]),
        "sales": product_sales(env, info),
        "revenue": float(info["revenue"]),
        "lost_customers": int(info["lost_customers"]),
        "price_walkaways": int(info["price_walkaways"]),
        "refused_credit": int(info["refused_credit"]),
        "haggles": parse_haggles(info["events"]),
        "haggle_count": int(info["haggles"]),
        "haggle_successes": int(info["haggle_successes"]),
        "jara_count": int(info["jara_count"]),
        "jara_given": float(info["jara_given"]),
        "credit_given": float(info["credit_given"]),
        "repayments": float(info["repayments"]),
        "defaults": float(info["defaults"]),
        "invalid_action": bool(info["invalid_action"]),
        "bankrupt": bool(info["bankrupt"]),
        "end_reason": info["end_reason"],
    }


def initial_frame(env):
    """Return the shop as it opens on day 0, before any action, so the scene can start full."""
    shop = env.unwrapped
    return {"day": 0, "state": shop.get_state(), "net_worth": float(shop.net_worth())}


class EpisodeRunner:
    """Play one episode of the raw ShopEnv day by day with a policy, next to rule_policy on the same seed.

    Iterating over frames() yields one frame per day; each frame also carries the rule shop's
    cumulative return and net worth so far. summary() describes both shops once the episode ends.
    """

    def __init__(self, policy_fn, seed=DEFAULT_SEED, market="default"):
        """Reset the agent env and the rule twin env on the same seed."""
        self.policy_fn = policy_fn
        self.seed = seed
        self.market = market
        self.env, self.obs = self._new_env()
        self.rule_env, self.rule_obs = self._new_env()
        self.total = 0.0
        self.rule_total = 0.0
        self.rule_done = False
        self.last_frame = None

    def _new_env(self):
        """Return a ShopEnv reset on this runner's seed, with its action space seeded too."""
        env = ShopEnv(market=self.market)
        obs, _ = env.reset(seed=self.seed)
        env.action_space.seed(self.seed)
        return env, obs

    def initial_frame(self):
        """Return the day 0 frame of the agent shop."""
        frame = initial_frame(self.env)
        frame["rule"] = {"cumulative_return": 0.0, "net_worth": float(self.rule_env.net_worth())}
        return frame

    def _step_rule(self):
        """Advance the rule shop by one day unless its episode already ended, and return its action name."""
        if self.rule_done:
            return None
        action = int(rule_policy(self.rule_obs, self.rule_env))
        self.rule_obs, reward, terminated, truncated, _ = self.rule_env.step(action)
        self.rule_total += reward
        self.rule_done = terminated or truncated
        return ACTION_NAMES[action]

    def frames(self):
        """Yield one frame per day until the agent's episode ends."""
        done = False
        while not done:
            action = int(self.policy_fn(self.obs, self.env))
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            self.total += reward
            done = terminated or truncated
            day_result = {"action": action, "reward": reward, "cumulative_return": self.total, "info": info}
            frame = frame_from_step(self.env, day_result)
            rule_action = self._step_rule()
            frame["rule"] = {
                "cumulative_return": float(self.rule_total),
                "net_worth": float(self.rule_env.net_worth()),
                "action_label": action_label(rule_action) if rule_action else None,
            }
            self.last_frame = frame
            yield frame

    def summary(self):
        """Return the end of episode comparison of the agent shop and the rule shop."""
        frame = self.last_frame or self.initial_frame()
        return {
            "seed": self.seed,
            "market": self.market,
            "days": int(frame["day"]),
            "end_reason": frame.get("end_reason"),
            "agent": {
                "return": float(self.total),
                "net_worth": float(self.env.net_worth()),
                "cash": float(self.env.cash),
                "trust": float(self.env.trust),
                "debt": float(self.env.total_debt()),
            },
            "rule": {
                "return": float(self.rule_total),
                "net_worth": float(self.rule_env.net_worth()),
                "cash": float(self.rule_env.cash),
                "trust": float(self.rule_env.trust),
                "debt": float(self.rule_env.total_debt()),
            },
        }


def checkpoint_kind(data):
    """Return "logits" for a policy-gradient checkpoint and "q" for a DQN one, from its saved config."""
    if data.get("extra", {}).get("algo") in PG_ALGOS or "eps_start" not in data["config"]:
        return "logits"
    return "q"


def load_checkpoint_policy(path):
    """Load a DQN or policy-gradient checkpoint and return a greedy policy_fn for it."""
    import torch

    from training.common import greedy_fn_from_module, load_checkpoint

    data = torch.load(path)
    kind = checkpoint_kind(data)
    if kind == "logits":
        from training.pg_training import build_logits_net as build_fn
    else:
        from training.dqn_training import build_value_net as build_fn
    module, _ = load_checkpoint(path, build_fn)
    return greedy_fn_from_module(module, kind)


def load_policy(spec):
    """Return (name, policy_fn) for "rule", "random" or a checkpoint path."""
    if spec in BUILTIN_POLICIES:
        return spec, BUILTIN_POLICIES[spec]
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"No policy named {spec!r} and no checkpoint at {path}")
    return path.stem, load_checkpoint_policy(path)


def product_meta(env):
    """Return each product's key, display name and storage cap for the browser."""
    return [
        {"key": product["key"], "name": PRODUCT_DISPLAY_NAMES.get(product["key"], display_text(product["label"])),
         "cap": float(product["cap"])}
        for product in env.unwrapped.products
    ]


def episode_meta(policy_name, seed, market):
    """Return the static facts the browser needs: actions, products, caps, episode length, policy and seed."""
    env = ShopEnv(market=market)
    return {
        "actions": ACTION_NAMES,
        "action_labels": [action_label(name) for name in ACTION_NAMES],
        "products": product_meta(env),
        "episode_days": int(env.config["calendar"]["episode_days"]),
        "policy": policy_name,
        "seed": seed,
        "market": market,
        "seconds_per_day_at_1x": 2.0,
        "start_cash": float(env.config["start_cash"]),
    }


def to_jsonable(value):
    """Convert numpy scalars and arrays inside a frame to plain Python values."""
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
