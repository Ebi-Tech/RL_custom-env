"""All tunable numbers for the Beyond Sale Shop environment, plus the market variants.

Money values are placeholder simulation values, to be calibrated later with real shop data.
"""
import copy

MARKETS = ("default", "harsh", "price_scale")

HARSH_DEFAULT_PROB_MULTIPLIER = 2.0
HARSH_ARRIVAL_MULTIPLIER = 0.8
DEFAULT_PRICE_FACTOR = 0.6

BASE_CONFIG = {
    "market": "default",
    "money_scale": 1.0,
    "start_cash": 200000,
    "fixed_cost": 3000,
    "start_stock_jitter": (0.8, 1.2),
    "mudu_per_bag": 36,
    "calendar": {
        "episode_days": 90,
        "month_length": 30,
    },
    "products": [
        {
            "key": "indomie",
            "name": "Indomie",
            "label": "Indomie packs",
            "cost": 350,
            "price": 450,
            "start_stock": 300,
            "cap": 800,
            "sales_norm": 150,
            "small_batch": 120,
            "large_batch": 400,
        },
        {
            "key": "soap",
            "name": "Soap",
            "label": "soap bars",
            "cost": 400,
            "price": 550,
            "start_stock": 80,
            "cap": 300,
            "sales_norm": 60,
            "small_batch": 48,
            "large_batch": 144,
        },
        {
            "key": "rice_bag",
            "name": "Rice bag",
            "label": "rice bags",
            "cost": 72000,
            "price": 80000,
            "start_stock": 3,
            "cap": 8,
            "sales_norm": 3,
            "small_batch": 1,
            "large_batch": 3,
        },
        {
            "key": "loose_rice",
            "name": "Loose rice",
            "label": "mudu of loose rice",
            "cost": None,
            "price": 2600,
            "start_stock": 36,
            "cap": 72,
            "sales_norm": 60,
            "small_batch": None,
            "large_batch": None,
        },
    ],
    "purchases": [
        {
            "key": "indomie_single",
            "label": "Indomie single",
            "slot": 0,
            "share": 0.47,
            "min_qty": 1,
            "max_qty": 5,
            "units_per_item": 1,
            "item_price_factor": 1.0,
            "haggle": False,
            "jara": False,
        },
        {
            "key": "indomie_carton",
            "label": "Indomie carton",
            "slot": 0,
            "share": 0.02,
            "min_qty": 1,
            "max_qty": 1,
            "units_per_item": 40,
            "item_price_factor": 0.95,
            "haggle": True,
            "jara": False,
        },
        {
            "key": "soap",
            "label": "Soap",
            "slot": 1,
            "share": 0.28,
            "min_qty": 1,
            "max_qty": 3,
            "units_per_item": 1,
            "item_price_factor": 1.0,
            "haggle": False,
            "jara": False,
        },
        {
            "key": "rice_bag",
            "label": "Rice bag",
            "slot": 2,
            "share": 0.02,
            "min_qty": 1,
            "max_qty": 1,
            "units_per_item": 1,
            "item_price_factor": 1.0,
            "haggle": True,
            "jara": False,
        },
        {
            "key": "loose_rice",
            "label": "Loose rice",
            "slot": 3,
            "share": 0.21,
            "min_qty": 1,
            "max_qty": 3,
            "units_per_item": 1,
            "item_price_factor": 1.0,
            "haggle": False,
            "jara": True,
        },
    ],
    "customers": {
        "base_mean": 30,
        "trust_offset": 0.5,
        "weekday_factors": (1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 0.6),
        "payday_factor": 1.3,
        "payday_last_days": 3,
        "payday_first_days": 2,
        "arrival_factor": 1.0,
    },
    "pricing": {
        "band_multipliers": (0.92, 1.0, 1.08),
        "walkaway_probs": (0.02, 0.05, 0.20),
    },
    "haggle": {
        "buyer_max_range": (0.88, 1.05),
        "floor_ratio": 0.92,
    },
    "credit": {
        "ask_prob": 0.2,
        "open_trust_gain": 0.002,
        "strict_pay_cash_prob": 0.5,
        "refused_trust_loss": 0.005,
    },
    "jara": {
        "ask_prob": 0.6,
        "amount": 0.1,
        "trust_gain": 0.001,
    },
    "debt": {
        "repay_prob": 0.06,
        "chase_repay_prob": 0.15,
        "chase_min_age": 7,
        "chase_trust_loss": 0.02,
        "old_debt_age": 21,
        "default_prob": 0.05,
    },
    "trust": {
        "start": 0.7,
        "target": 0.7,
        "recovery_rate": 0.05,
        "min": 0.2,
        "max": 1.0,
        "stockout_loss": 0.003,
    },
    "reward": {
        "money_unit": 10000,
        "lost_customer_penalty": 0.02,
        "invalid_action_penalty": 0.2,
        "final_debt_weight": 0.5,
        "bankruptcy_penalty": 10.0,
    },
    "observation": {
        "cash_norm": 500000,
        "debt_norm": 100000,
        "lost_customers_norm": 10,
        "history_days": 7,
        "default_window_days": 30,
        "no_haggle_rate": 0.5,
    },
    "rule_policy": {
        "open_bag_below_mudu": 8,
        "reorder_points": (150, 40, 2),
        "large_batch_min_cash_after": 50000,
        "chase_old_debt_above": 20000,
        "credit_chase_old_debt_above": 10000,
    },
}


def make_config(market="default", price_factor=None):
    """Return a fresh config dict for a market variant, so each env gets its own copy to change."""
    if market not in MARKETS:
        raise ValueError(f"Unknown market {market!r}, expected one of {MARKETS}")
    config = copy.deepcopy(BASE_CONFIG)
    config["market"] = market
    if market == "harsh":
        apply_harsh_market(config)
    if market == "price_scale":
        factor = DEFAULT_PRICE_FACTOR if price_factor is None else price_factor
        apply_price_scale(config, factor)
    set_loose_rice_cost(config)
    return config


def apply_harsh_market(config):
    """Double the debt default probability and cut customer arrivals, for a tougher market."""
    config["debt"]["default_prob"] *= HARSH_DEFAULT_PROB_MULTIPLIER
    config["customers"]["arrival_factor"] *= HARSH_ARRIVAL_MULTIPLIER


def apply_price_scale(config, factor):
    """Scale every money amount by one factor, so rewards and observations stay comparable."""
    for product in config["products"]:
        if product["cost"] is not None:
            product["cost"] *= factor
        product["price"] *= factor
    config["start_cash"] *= factor
    config["fixed_cost"] *= factor
    config["money_scale"] *= factor


def set_loose_rice_cost(config):
    """Derive the cost of one mudu from the rice bag cost, so opening a bag keeps stock value equal."""
    rice_bag, loose_rice = config["products"][2], config["products"][3]
    loose_rice["cost"] = rice_bag["cost"] / config["mudu_per_bag"]
