"""Turn ShopEnv episodes into JSON frames for the 3D market street, one frame per trading day.

Five shops trade side by side in lockstep on the same seed and market: the four trained agents
(DQN, REINFORCE, A2C, PPO) and Mama Ngozi, a typical owner who follows rule_policy. Each frame
carries every shop's state and day, the agents' top 3 action scores, and a net worth leaderboard.
"""
import re
from pathlib import Path

import numpy as np

from environment.baselines import rule_policy
from environment.custom_env import ACTION_NAMES, ShopEnv

DEFAULT_SEED = 1000
DEFAULT_MODEL_DIR = Path("models/best")
MARKETS = ("default", "harsh")
TOP_SCORES = 3
PG_ALGOS = ("reinforce", "a2c", "ppo")
TYPICAL_OWNER = "mama"
SCORE_LABELS = {"logits": "probability", "q": "relative preference"}
SHOP_SPECS = (
    {"id": "dqn", "name": "DQN Stores", "checkpoint": "dqn.pt", "color": "#e4572e"},
    {"id": "reinforce", "name": "REINFORCE & Sons", "checkpoint": "reinforce.pt", "color": "#f2a541"},
    {"id": "a2c", "name": "A2C Enterprises", "checkpoint": "a2c.pt", "color": "#17a398"},
    {"id": "ppo", "name": "PPO Provisions", "checkpoint": "ppo.pt", "color": "#8e5fd6"},
    {"id": TYPICAL_OWNER, "name": "Mama Ngozi Provisions", "checkpoint": None, "color": "#2e86de",
     "subtitle": "typical owner"},
)
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
    """Build one JSON-serialisable shop frame for a finished day.

    day_result holds the action, reward, cumulative return and info dict of that day.
    """
    info = day_result["info"]
    shop = env.unwrapped
    action_name = ACTION_NAMES[day_result["action"]]
    state = shop.get_state()
    return {
        "day": int(info["day"]),
        "action": action_name,
        "action_label": action_label(action_name),
        "reward": float(day_result["reward"]),
        "cumulative_return": float(day_result["cumulative_return"]),
        "state": state,
        "net_worth": float(shop.net_worth()),
        "trust": state["trust"],
        "price_band": state["price_band"],
        "credit_policy": state["credit_policy"],
        "events": [display_text(event) for event in info["events"]],
        "customers": int(info["customers"]),
        "sales": product_sales(env, info),
        "revenue": float(info["revenue"]),
        "lost_customers": int(info["lost_customers"]),
        "price_walkaways": int(info["price_walkaways"]),
        "refused_credit": int(info["refused_credit"]),
        "turned_away": int(info["lost_customers"] + info["price_walkaways"] + info["refused_credit"]),
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


def checkpoint_kind(data):
    """Return "logits" for a policy-gradient checkpoint and "q" for a DQN one, from its saved config."""
    if data.get("extra", {}).get("algo") in PG_ALGOS or "eps_start" not in data["config"]:
        return "logits"
    return "q"


def load_agent_module(path):
    """Load a DQN or policy-gradient checkpoint and return its network and its output kind."""
    import torch

    from training.common import load_checkpoint

    data = torch.load(path)
    kind = checkpoint_kind(data)
    if kind == "logits":
        from training.pg_training import build_logits_net as build_fn
    else:
        from training.dqn_training import build_value_net as build_fn
    module, _ = load_checkpoint(path, build_fn)
    return module, kind


def load_shops(model_dir=DEFAULT_MODEL_DIR):
    """Return the five shop specs with their loaded networks; a missing checkpoint becomes a rule placeholder."""
    shops = []
    for spec in SHOP_SPECS:
        shop = {**spec, "module": None, "kind": "rule", "placeholder": False, "score_label": None}
        if spec["checkpoint"]:
            path = Path(model_dir) / spec["checkpoint"]
            if path.exists():
                shop["module"], shop["kind"] = load_agent_module(path)
                shop["score_label"] = SCORE_LABELS[shop["kind"]]
            else:
                shop["placeholder"] = True
                shop["name"] = f"{spec['name']} (placeholder)"
        shops.append(shop)
    return shops


def action_distribution(module, obs):
    """Return the softmax over a network's 13 outputs for one observation, as a numpy array summing to 1."""
    import torch

    with torch.no_grad():
        output = module(torch.as_tensor(obs, dtype=torch.float32))
    return torch.softmax(output, dim=-1).numpy().astype(np.float64)


def top_scores(distribution, count=TOP_SCORES):
    """Return the highest scoring actions with their labels and scores, best first."""
    order = np.argsort(-distribution)[:count]
    return [{"action": ACTION_NAMES[index], "label": action_label(ACTION_NAMES[index]),
             "score": float(distribution[index])} for index in order]


def choose_action(shop, obs, env):
    """Return a shop's action for today and its top scores: the argmax for agents, the rule for the owner."""
    if shop["module"] is None:
        return int(rule_policy(obs, env)), None
    distribution = action_distribution(shop["module"], obs)
    return int(np.argmax(distribution)), top_scores(distribution)


def shop_header(shop):
    """Return the fields that identify a shop in every frame."""
    return {"id": shop["id"], "name": shop["name"], "color": shop["color"], "placeholder": shop["placeholder"]}


class MarketRunner:
    """Run the five shops in lockstep, one day at a time, on the same seed and market.

    Iterating over frames() yields one frame per day holding every shop; a shop that goes
    bankrupt stays closed for the remaining days. summary() gives the final leaderboard.
    """

    def __init__(self, shops, seed=DEFAULT_SEED, market="default"):
        """Reset one ShopEnv per shop on the same seed and market."""
        self.shops = shops
        self.seed = seed
        self.market = market
        self.lanes = [self._new_lane(shop) for shop in shops]
        self.episode_days = int(self.lanes[0]["env"].config["calendar"]["episode_days"])
        self.day = 0

    def _new_lane(self, shop):
        """Return the running state of one shop: its env, observation and totals."""
        env = ShopEnv(market=self.market)
        obs, _ = env.reset(seed=self.seed)
        env.action_space.seed(self.seed)
        return {"shop": shop, "env": env, "obs": obs, "total": 0.0, "done": False, "frame": None,
                "turned_away": 0, "bad_debt": 0.0}

    def _opening_frame(self, lane):
        """Return a shop as it opens on day 0, before any action."""
        env = lane["env"]
        state = env.get_state()
        return {**shop_header(lane["shop"]), "day": 0, "state": state, "net_worth": float(env.net_worth()),
                "cumulative_return": 0.0, "trust": state["trust"], "turned_away_total": 0, "bad_debt_total": 0.0}

    def initial_frame(self):
        """Return the day 0 frame of the whole market."""
        shops = [self._opening_frame(lane) for lane in self.lanes]
        return {"day": 0, "shops": shops, "leaderboard": leaderboard(shops)}

    def _step_lane(self, lane):
        """Play one day for one shop and return its frame, or repeat its last frame if it has closed."""
        if lane["done"]:
            return {**lane["frame"], "closed": True, "action": "closed", "action_label": "closed",
                    "reward": 0.0, "events": [], "customers": 0, "turned_away": 0, "haggles": [], "jara_count": 0,
                    "scores": None, "day": self.day}
        action, scores = choose_action(lane["shop"], lane["obs"], lane["env"])
        lane["obs"], reward, terminated, truncated, info = lane["env"].step(action)
        lane["total"] += reward
        lane["done"] = terminated or truncated
        day_result = {"action": action, "reward": reward, "cumulative_return": lane["total"], "info": info}
        frame = {**shop_header(lane["shop"]), **frame_from_step(lane["env"], day_result)}
        lane["turned_away"] += frame["turned_away"]
        lane["bad_debt"] += frame["defaults"]
        frame.update({"scores": scores, "score_label": lane["shop"]["score_label"], "closed": False,
                      "turned_away_total": lane["turned_away"], "bad_debt_total": lane["bad_debt"]})
        lane["frame"] = frame
        return frame

    def frames(self):
        """Yield one market frame per day until every shop has finished its episode."""
        while not all(lane["done"] for lane in self.lanes):
            self.day += 1
            shops = [self._step_lane(lane) for lane in self.lanes]
            yield {"day": self.day, "shops": shops, "leaderboard": leaderboard(shops),
                   "total_customers": sum(shop["customers"] for shop in shops)}

    def summary(self):
        """Return the final leaderboard with returns, net worth, turned away customers and bad debt."""
        shops = [lane["frame"] or self._opening_frame(lane) for lane in self.lanes]
        return {"seed": self.seed, "market": self.market, "days": self.day, "leaderboard": leaderboard(shops),
                "end_reasons": {lane["shop"]["id"]: (lane["frame"] or {}).get("end_reason") for lane in self.lanes}}


def leaderboard(shops):
    """Rank the shops by net worth, with each shop's difference from the typical owner."""
    owner = next(shop for shop in shops if shop["id"] == TYPICAL_OWNER)
    rows = [{
        "id": shop["id"], "name": shop["name"], "color": shop["color"],
        "net_worth": float(shop["net_worth"]), "cumulative_return": float(shop["cumulative_return"]),
        "net_worth_vs_owner": float(shop["net_worth"] - owner["net_worth"]),
        "return_vs_owner": float(shop["cumulative_return"] - owner["cumulative_return"]),
        "turned_away_total": int(shop.get("turned_away_total", 0)),
        "bad_debt_total": float(shop.get("bad_debt_total", 0.0)),
    } for shop in shops]
    rows.sort(key=lambda row: row["net_worth"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def product_meta(env):
    """Return each product's key, display name and storage cap for the browser."""
    return [
        {"key": product["key"], "name": PRODUCT_DISPLAY_NAMES.get(product["key"], display_text(product["label"])),
         "cap": float(product["cap"])}
        for product in env.unwrapped.products
    ]


def market_meta(shops, seed, market):
    """Return the static facts the browser needs: shops, actions, products, caps, episode length, seed and market."""
    env = ShopEnv(market=market)
    return {
        "shops": [{**shop_header(shop), "subtitle": shop.get("subtitle", ""), "kind": shop["kind"],
                   "score_label": shop["score_label"]} for shop in shops],
        "actions": ACTION_NAMES,
        "action_labels": [action_label(name) for name in ACTION_NAMES],
        "products": product_meta(env),
        "episode_days": int(env.config["calendar"]["episode_days"]),
        "seed": seed,
        "market": market,
        "markets": list(MARKETS),
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
