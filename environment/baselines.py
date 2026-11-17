"""Baseline policies for the Beyond Sale Shop environment, used as yardsticks for trained agents."""
from environment.custom_env import (
    CHASE_DEBTS,
    CREDIT_OPEN,
    CREDIT_STRICT,
    HOLD,
    LOOSE_RICE,
    LOWER_BAND,
    NORMAL_BAND,
    OPEN_BAG,
    RAISE_BAND,
    RESTOCK_ACTIONS,
    RICE_BAG,
)

RESTOCK_ACTION_FOR = {slot_and_batch: action for action, slot_and_batch in RESTOCK_ACTIONS.items()}


def random_policy(obs, env):
    """Pick a uniformly random action, the floor any learned agent must beat."""
    return int(env.action_space.sample())


def rule_policy(obs, env):
    """Follow a sensible shopkeeper rule of thumb with strict credit, reading state from env.unwrapped."""
    shop = env.unwrapped
    chase_above = shop.config["rule_policy"]["chase_old_debt_above"]
    return shopkeeper_action(shop, credit_open=False, chase_above=chase_above)


def rule_credit_policy(obs, env):
    """Follow the same rule of thumb but keep credit open and chase debts sooner, to test the credit trade-off."""
    shop = env.unwrapped
    chase_above = shop.config["rule_policy"]["credit_chase_old_debt_above"]
    return shopkeeper_action(shop, credit_open=True, chase_above=chase_above)


def shopkeeper_action(shop, credit_open, chase_above):
    """Return the shopkeeper's action in priority order: policy settings, open bag, restock, chase, hold.

    Credit is kept at the wanted policy and the band at normal before anything else, so the
    rest of the rule always runs under the same trading conditions.
    """
    rules = shop.config["rule_policy"]
    if shop.credit_open != credit_open:
        return CREDIT_OPEN if credit_open else CREDIT_STRICT
    if shop.price_band > NORMAL_BAND:
        return LOWER_BAND
    if shop.price_band < NORMAL_BAND:
        return RAISE_BAND
    if shop.stock[LOOSE_RICE] < rules["open_bag_below_mudu"] and shop.stock[RICE_BAG] >= 1:
        return OPEN_BAG
    restock = choose_restock(shop, rules)
    if restock is not None:
        return restock
    _, old_debt = shop.debt_by_age()
    if old_debt / shop.config["money_scale"] > chase_above:
        return CHASE_DEBTS
    return HOLD


def choose_restock(shop, rules):
    """Return a restock action for the first product below its reorder point that cash can cover.

    The large batch is used when cash after paying stays above a safety buffer, otherwise the
    small batch when it is affordable, otherwise the next product is checked.
    """
    min_cash_after = rules["large_batch_min_cash_after"] * shop.config["money_scale"]
    for slot, reorder_point in enumerate(rules["reorder_points"]):
        if shop.stock[slot] + shop.pending[slot] >= reorder_point:
            continue
        product = shop.products[slot]
        if shop.cash - product["large_batch"] * product["cost"] >= min_cash_after:
            return RESTOCK_ACTION_FOR[(slot, "large_batch")]
        if shop.cash >= product["small_batch"] * product["cost"]:
            return RESTOCK_ACTION_FOR[(slot, "small_batch")]
    return None
