"""Beyond Sale Shop: a Gymnasium environment that simulates one small Nigerian provision shop.

One step is one trading day and one episode is 90 days. The agent suggests the owner's daily
decision and must protect the shop's cash while handling haggling, customer credit, debt
defaults and stock.
"""
import math
from collections import deque

import gymnasium
import numpy as np
from gymnasium import spaces

from environment.config import make_config

INDOMIE, SOAP, RICE_BAG, LOOSE_RICE = 0, 1, 2, 3
NUM_SLOTS = 4
RESTOCKABLE_SLOTS = 3
OBS_SIZE = 23
DAYS_IN_WEEK = 7

HOLD = 0
OPEN_BAG = 7
RAISE_BAND = 8
LOWER_BAND = 9
CREDIT_OPEN = 10
CREDIT_STRICT = 11
CHASE_DEBTS = 12

ACTION_NAMES = [
    "hold",
    "restock_indomie_small",
    "restock_indomie_large",
    "restock_soap_small",
    "restock_soap_large",
    "restock_rice_small",
    "restock_rice_large",
    "open_rice_bag",
    "raise_price_band",
    "lower_price_band",
    "credit_open",
    "credit_strict",
    "chase_debts",
]

RESTOCK_ACTIONS = {
    1: (INDOMIE, "small_batch"),
    2: (INDOMIE, "large_batch"),
    3: (SOAP, "small_batch"),
    4: (SOAP, "large_batch"),
    5: (RICE_BAG, "small_batch"),
    6: (RICE_BAG, "large_batch"),
}

LOW_BAND, NORMAL_BAND, HIGH_BAND = 0, 1, 2
BAND_NAMES = ("low", "normal", "high")


class ShopEnv(gymnasium.Env):
    """A 90 day simulation of one informal provision shop, with 13 daily actions and 23 observations.

    Money is kept in raw simulation units inside the env and in the info dict, and divided by
    the config's money scale wherever it enters the reward or the observation.
    """

    metadata = {"render_modes": []}

    def __init__(self, market="default", price_factor=None, config=None):
        """Build the shop from a market variant name, or from a full config dict when one is given."""
        self.config = config if config is not None else make_config(market, price_factor)
        self.render_mode = None
        self.products = self.config["products"]
        self.purchase_types = self.config["purchases"]
        shares = np.array([purchase["share"] for purchase in self.purchase_types], dtype=np.float64)
        purchase_cdf = (shares / shares.sum()).cumsum()
        self.purchase_cdf = purchase_cdf / purchase_cdf[-1]
        self.unit_costs = np.array([product["cost"] for product in self.products], dtype=np.float64)
        self.stock_caps = np.array([product["cap"] for product in self.products], dtype=np.float64)
        self.sales_norms = np.array([product["sales_norm"] for product in self.products], dtype=np.float64)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Start a new episode with seeded random stock levels and a random calendar position."""
        super().reset(seed=seed)
        month_length = self.config["calendar"]["month_length"]
        self.day = 0
        self.weekday = int(self.np_random.integers(0, DAYS_IN_WEEK))
        self.day_of_month = int(self.np_random.integers(1, month_length + 1))
        self.cash = float(self.config["start_cash"])
        self.stock = self._random_start_stock()
        self.pending = np.zeros(NUM_SLOTS)
        self.debts = []
        self.trust = float(self.config["trust"]["start"])
        self.price_band = NORMAL_BAND
        self.credit_open = False
        self.lost_today = 0
        self._reset_history()
        return self._observation(), self._base_info()

    def step(self, action):
        """Play one trading day in the fixed order of events and return the Gymnasium step tuple."""
        action = int(action)
        today = self._new_day_record(action)
        wealth_before = self._wealth()
        order = self._apply_action(action, today)
        self._deliver(order, today)
        self._serve_customers(today)
        self._waste_loose_rice(today)
        self._settle_debts(today)
        self._pay_holding_cost(today)
        self.cash -= self.config["fixed_cost"]
        self._drift_trust()
        self._close_day(today)
        reward, terminated, truncated = self._score_day(today, wealth_before)
        return self._observation(), reward, terminated, truncated, self._step_info(today)

    def total_debt(self):
        """Return the full amount customers still owe the shop."""
        return float(sum(entry["amount"] for entry in self.debts))

    def net_worth(self):
        """Return cash plus stock and pending stock at cost plus the expected recovery of outstanding debt."""
        debt_weight = self.config["reward"]["final_debt_weight"]
        return self._wealth() + debt_weight * self.total_debt()

    def debt_by_age(self):
        """Split outstanding debt into recent and old amounts, since old debt is the risky part."""
        old_age = self.config["debt"]["old_debt_age"]
        old = sum(entry["amount"] for entry in self.debts if entry["age"] > old_age)
        return self.total_debt() - old, float(old)

    def get_state(self):
        """Return a JSON-serialisable snapshot of the full shop state for the future web renderer."""
        recent_debt, old_debt = self.debt_by_age()
        return {
            "market": self.config["market"],
            "money_scale": float(self.config["money_scale"]),
            "day": int(self.day),
            "weekday": int(self.weekday),
            "day_of_month": int(self.day_of_month),
            "cash": float(self.cash),
            "stock": self._per_product(self.stock),
            "pending": self._per_product(self.pending),
            "price_band": BAND_NAMES[self.price_band],
            "credit_policy": "open" if self.credit_open else "strict",
            "trust": float(self.trust),
            "debts": [{"amount": float(entry["amount"]), "age": int(entry["age"])} for entry in self.debts],
            "recent_debt": float(recent_debt),
            "old_debt": float(old_debt),
            "lost_customers_today": int(self.lost_today),
        }

    def _per_product(self, values):
        """Map a per-slot array to a dict keyed by product, for readable state snapshots."""
        return {product["key"]: float(value) for product, value in zip(self.products, values)}

    def _random_start_stock(self):
        """Draw each starting stock level as the base amount times a seeded random factor."""
        low, high = self.config["start_stock_jitter"]
        levels = [round(product["start_stock"] * self.np_random.uniform(low, high)) for product in self.products]
        return np.array(levels, dtype=np.float64)

    def _reset_history(self):
        """Clear the rolling windows used for the 7 day and 30 day observation features."""
        obs_config = self.config["observation"]
        self.sales_history = deque(maxlen=obs_config["history_days"])
        self.haggle_history = deque(maxlen=obs_config["history_days"])
        self.lending_history = deque(maxlen=obs_config["default_window_days"])

    def _new_day_record(self, action):
        """Create the counters and event list that one trading day fills in."""
        return {
            "action_name": ACTION_NAMES[action],
            "invalid_action": False,
            "noop": False,
            "chase": False,
            "customers": 0,
            "units_sold": np.zeros(NUM_SLOTS),
            "revenue": 0.0,
            "cash_sales": 0.0,
            "lost_customers": 0,
            "refused_credit": 0,
            "price_walkaways": 0,
            "haggles": 0,
            "haggle_successes": 0,
            "jara_requests": 0,
            "jara_count": 0,
            "jara_given": 0.0,
            "credit_given": 0.0,
            "repayments": 0.0,
            "defaults": 0.0,
            "waste": 0.0,
            "holding_cost": 0.0,
            "profit_today": 0.0,
            "bankrupt": False,
            "end_reason": None,
            "events": [],
        }

    def _wealth(self):
        """Return cash plus all stock and pending stock valued at cost, the base of the reward."""
        return self.cash + float(np.dot(self.stock + self.pending, self.unit_costs))

    def _apply_action(self, action, today):
        """Carry out the owner's decision for the day and return any restock order placed."""
        today["noop"] = self._is_noop(action)
        order = np.zeros(NUM_SLOTS)
        if action in RESTOCK_ACTIONS:
            self._restock(action, order, today)
        elif action == OPEN_BAG:
            self._open_bag(today)
        elif action in (RAISE_BAND, LOWER_BAND):
            self._move_band(action, today)
        elif action in (CREDIT_OPEN, CREDIT_STRICT):
            self._set_credit(action == CREDIT_OPEN, today)
        elif action == CHASE_DEBTS:
            self._start_chase(today)
        return order

    def _is_noop(self, action):
        """Return whether a valid action would change nothing about the shop today.

        That is holding, raising the band at high, lowering it at low, choosing the credit policy
        already in place, or chasing when no debt is old enough to be chased. A no-op chase still
        costs its trust loss; the flag only records that it reaches no debt.
        """
        if action == HOLD:
            return True
        if action == RAISE_BAND:
            return self.price_band == HIGH_BAND
        if action == LOWER_BAND:
            return self.price_band == LOW_BAND
        if action in (CREDIT_OPEN, CREDIT_STRICT):
            return self.credit_open == (action == CREDIT_OPEN)
        if action == CHASE_DEBTS:
            return not self._has_chaseable_debt()
        return False

    def _has_chaseable_debt(self):
        """Return whether any debt is older than the minimum age that chasing affects."""
        min_age = self.config["debt"]["chase_min_age"]
        return any(entry["age"] > min_age for entry in self.debts)

    def _restock(self, action, order, today):
        """Pay cash now for a batch that arrives tomorrow, or mark the action invalid if cash is short."""
        slot, batch_key = RESTOCK_ACTIONS[action]
        product = self.products[slot]
        quantity = product[batch_key]
        cost = quantity * product["cost"]
        if self.stock[slot] + self.pending[slot] + quantity > self.stock_caps[slot]:
            self._mark_invalid(today, f"Not enough space for {quantity} more {product['label']}")
            return
        if self.cash < cost:
            self._mark_invalid(today, f"cannot afford {quantity} {product['label']} ({cost:.0f})")
            return
        self.cash -= cost
        order[slot] += quantity
        today["events"].append(f"Ordered {quantity} {product['label']} for {cost:.0f}, arrives tomorrow")

    def _open_bag(self, today):
        """Move one rice bag into loose rice for measured sale, or mark invalid if no bag is in stock."""
        if self.stock[RICE_BAG] < 1:
            self._mark_invalid(today, "no rice bag to open")
            return
        mudu = self.config["mudu_per_bag"]
        if self.stock[LOOSE_RICE] + mudu > self.stock_caps[LOOSE_RICE]:
            self._mark_invalid(today, f"Not enough space for {mudu} more mudu of loose rice")
            return
        self.stock[RICE_BAG] -= 1
        self.stock[LOOSE_RICE] += mudu
        today["events"].append(f"Opened a rice bag into {mudu} mudu")

    def _move_band(self, action, today):
        """Raise or lower the price band by one level, staying within low and high."""
        change = 1 if action == RAISE_BAND else -1
        self.price_band = int(np.clip(self.price_band + change, LOW_BAND, HIGH_BAND))
        today["events"].append(f"Price band is {BAND_NAMES[self.price_band]}")

    def _set_credit(self, is_open, today):
        """Switch the credit policy, which persists until changed again."""
        self.credit_open = is_open
        today["events"].append(f"Credit policy is {'open' if is_open else 'strict'}")

    def _start_chase(self, today):
        """Mark today as a debt chasing day, which costs some trust once."""
        today["chase"] = True
        self._change_trust(-self.config["debt"]["chase_trust_loss"])
        today["events"].append("Chased old debts")

    def _mark_invalid(self, today, reason):
        """Record that the chosen action could not be carried out, so it gets the invalid penalty."""
        today["invalid_action"] = True
        today["events"].append(f"Invalid action: {reason}")

    def _change_trust(self, delta):
        """Add a change to trust and keep it inside the configured range."""
        trust_config = self.config["trust"]
        self.trust = float(min(max(self.trust + delta, trust_config["min"]), trust_config["max"]))

    def _deliver(self, order, today):
        """Put yesterday's order on the shelf and hold today's order until tomorrow."""
        if self.pending.sum() > 0:
            self.stock += self.pending
            today["events"].append(f"Delivery arrived: {self._describe_units(self.pending)}")
        self.pending = order

    def _describe_units(self, units):
        """Turn a per-slot unit array into short text such as '40 Indomie packs'."""
        parts = [f"{amount:g} {product['label']}" for product, amount in zip(self.products, units) if amount > 0]
        return ", ".join(parts)

    def _customer_mean(self):
        """Return today's expected customer count from trust, weekday, payday and market factors."""
        customers = self.config["customers"]
        weekday_factor = customers["weekday_factors"][self.weekday]
        payday_factor = customers["payday_factor"] if self._is_payday() else 1.0
        trust_factor = customers["trust_offset"] + self.trust
        return customers["base_mean"] * trust_factor * weekday_factor * payday_factor * customers["arrival_factor"]

    def _is_payday(self):
        """Return whether today falls in the payday window at the end or start of the month."""
        customers = self.config["customers"]
        month_length = self.config["calendar"]["month_length"]
        near_end = self.day_of_month > month_length - customers["payday_last_days"]
        near_start = self.day_of_month <= customers["payday_first_days"]
        return near_end or near_start

    def _serve_customers(self, today):
        """Draw today's customers and serve each one in turn."""
        count = int(self.np_random.poisson(self._customer_mean()))
        today["customers"] = count
        for _ in range(count):
            self._serve_customer(today)
        self._log_sales(today)

    def _serve_customer(self, today):
        """Handle one customer from choosing a purchase through stock, credit, price and jara."""
        purchase = self.purchase_types[self._draw_purchase_index()]
        quantity = int(self.np_random.integers(purchase["min_qty"], purchase["max_qty"] + 1))
        slot = purchase["slot"]
        units = quantity * purchase["units_per_item"]
        if self.stock[slot] < units:
            self._lose_customer(today)
            return
        on_credit = self._credit_choice(today)
        if on_credit is None:
            return
        if purchase["haggle"]:
            price = self._haggle(purchase, quantity, today)
        else:
            price = self._fixed_price(purchase, quantity, today)
        if price is None:
            return
        self._complete_sale(slot, units, price, on_credit, today)
        if purchase["jara"]:
            self._handle_jara(today)

    def _draw_purchase_index(self):
        """Pick a purchase type by its share, using the same single uniform draw as Generator.choice.

        Generator.choice with probabilities does exactly this cumulative-share search, but it
        revalidates the probabilities on every call, which made it the slowest part of a step.
        """
        return int(self.purchase_cdf.searchsorted(self.np_random.random(), side="right"))

    def _lose_customer(self, today):
        """Count a customer lost to an empty shelf and lower trust."""
        today["lost_customers"] += 1
        self._change_trust(-self.config["trust"]["stockout_loss"])

    def _credit_choice(self, today):
        """Return True for a credit sale, False for a cash sale, or None when the buyer leaves."""
        credit = self.config["credit"]
        if self.np_random.random() >= credit["ask_prob"]:
            return False
        if self.credit_open:
            return True
        if self.np_random.random() < credit["strict_pay_cash_prob"]:
            return False
        today["refused_credit"] += 1
        self._change_trust(-credit["refused_trust_loss"])
        return None

    def _item_price(self, purchase):
        """Return the normal price of one item of a purchase type before the price band."""
        product = self.products[purchase["slot"]]
        return product["price"] * purchase["units_per_item"] * purchase["item_price_factor"]

    def _band_price(self, normal_price):
        """Apply the current price band multiplier to a normal price."""
        return normal_price * self.config["pricing"]["band_multipliers"][self.price_band]

    def _fixed_price(self, purchase, quantity, today):
        """Sell at the band price unless the buyer walks away over price, which is likelier at high band.

        A walk away at the high band also costs trust, because the shop gets a name for being expensive.
        """
        walk_prob = self.config["pricing"]["walkaway_probs"][self.price_band]
        if self.np_random.random() < walk_prob:
            today["price_walkaways"] += 1
            if self.price_band == HIGH_BAND:
                self._change_trust(-self.config["pricing"]["price_walkaway_trust_loss"])
            return None
        return quantity * self._band_price(self._item_price(purchase))

    def _haggle(self, purchase, quantity, today):
        """Bargain against the buyer's hidden maximum and return the agreed total, or None on walk away."""
        haggle = self.config["haggle"]
        normal_price = self._item_price(purchase)
        ask = self._band_price(normal_price)
        buyer_max = normal_price * self.np_random.uniform(*haggle["buyer_max_range"])
        today["haggles"] += 1
        price = settle_haggle(ask, buyer_max, haggle["floor_ratio"])
        if price is None:
            today["events"].append(f"{purchase['label']} haggle: buyer walked away")
            return None
        today["haggle_successes"] += 1
        today["events"].append(f"{purchase['label']} haggle: sold at {price:.0f}")
        return price * quantity

    def _complete_sale(self, slot, units, price, on_credit, today):
        """Take the goods off the shelf and record the money as cash or as a new debt entry."""
        self.stock[slot] -= units
        today["units_sold"][slot] += units
        today["revenue"] += price
        if on_credit:
            self.debts.append({"amount": float(price), "age": 0})
            today["credit_given"] += price
            self._change_trust(self.config["credit"]["open_trust_gain"])
            return
        self.cash += price
        today["cash_sales"] += price

    def _handle_jara(self, today):
        """Give a small free extra of loose rice when the buyer asks and stock allows."""
        jara = self.config["jara"]
        if self.np_random.random() >= jara["ask_prob"]:
            return
        today["jara_requests"] += 1
        if self.stock[LOOSE_RICE] < jara["amount"]:
            return
        self.stock[LOOSE_RICE] = round(self.stock[LOOSE_RICE] - jara["amount"], 6)
        today["jara_count"] += 1
        today["jara_given"] += jara["amount"]
        self._change_trust(jara["trust_gain"])

    def _log_sales(self, today):
        """Add short human-readable lines that summarise the day's sales and lost business."""
        events = today["events"]
        for product, units in zip(self.products, today["units_sold"]):
            if units > 0:
                events.append(f"Sold {units:g} {product['label']}")
        if today["lost_customers"]:
            events.append(f"Lost {today['lost_customers']} customers to empty shelves")
        if today["price_walkaways"] and self.price_band == HIGH_BAND:
            events.append(f"Lost {today['price_walkaways']} buyers to high prices")
        if today["refused_credit"]:
            events.append(f"Refused credit to {today['refused_credit']} buyers")
        if today["jara_count"]:
            events.append(f"Gave {today['jara_given']:.1f} mudu jara to {today['jara_count']} buyers")

    def _settle_debts(self, today):
        """Let each debt entry repay, default or age by one day."""
        kept = []
        for entry in self.debts:
            outcome = self._debt_outcome(entry["age"], today["chase"])
            if outcome == "repaid":
                self.cash += entry["amount"]
                today["repayments"] += entry["amount"]
            elif outcome == "defaulted":
                today["defaults"] += entry["amount"]
            else:
                entry["age"] += 1
                kept.append(entry)
        self.debts = kept
        if today["repayments"]:
            today["events"].append(f"Debts repaid: {today['repayments']:.0f}")
        if today["defaults"]:
            today["events"].append(f"Debts defaulted: {today['defaults']:.0f}")

    def _debt_outcome(self, age, chasing):
        """Decide whether one debt entry is repaid, defaults or stays open today."""
        debt = self.config["debt"]
        chased = chasing and age > debt["chase_min_age"]
        repay_prob = debt["chase_repay_prob"] if chased else debt["repay_prob"]
        if self.np_random.random() < repay_prob:
            return "repaid"
        if age > debt["old_debt_age"] and self.np_random.random() < debt["default_prob"]:
            return "defaulted"
        return "open"

    def _waste_loose_rice(self, today):
        """Lose a small share of open loose rice each day, rounded down to the waste step, as spoilage."""
        step = self.config["loose_rice_waste_step"]
        raw_waste = self.config["loose_rice_waste_rate"] * self.stock[LOOSE_RICE]
        waste = round(math.floor(round(raw_waste / step, 6)) * step, 6)
        if waste <= 0:
            return
        self.stock[LOOSE_RICE] = round(self.stock[LOOSE_RICE] - waste, 6)
        today["waste"] = waste
        today["events"].append(f"Lost {waste:.1f} mudu of loose rice to waste")

    def _pay_holding_cost(self, today):
        """Pay a daily cost proportional to the stock on the shelf, so hoarding stock is not free."""
        holding_cost = self.config["holding_cost_rate"] * float(np.dot(self.stock, self.unit_costs))
        self.cash -= holding_cost
        today["holding_cost"] = holding_cost

    def _drift_trust(self):
        """Move trust a fixed share of the gap toward its long run level, so shocks fade over time."""
        trust_config = self.config["trust"]
        gap = trust_config["target"] - self.trust
        self._change_trust(trust_config["recovery_rate"] * gap)

    def _close_day(self, today):
        """Store the day in the rolling windows and move the calendar to the next day."""
        self.sales_history.append(today["units_sold"].copy())
        self.haggle_history.append((today["haggles"], today["haggle_successes"]))
        self.lending_history.append((today["credit_given"], today["defaults"]))
        self.lost_today = today["lost_customers"]
        month_length = self.config["calendar"]["month_length"]
        self.day += 1
        self.weekday = (self.weekday + 1) % DAYS_IN_WEEK
        self.day_of_month = self.day_of_month % month_length + 1

    def _score_day(self, today, wealth_before):
        """Compute the day's reward and whether the episode ended, by bankruptcy or on the final day.

        Both endings are terminal: day 90 is the true end of the task, since the day is observed
        and the debt settlement happens on it, so nothing is truncated. The info flags bankrupt
        and end_reason tell the two endings apart, with bankruptcy taking precedence.
        """
        rewards = self.config["reward"]
        money_unit = rewards["money_unit"] * self.config["money_scale"]
        profit = self._wealth() - wealth_before
        today["profit_today"] = profit
        reward = profit / money_unit
        reward -= rewards["lost_customer_penalty"] * today["lost_customers"]
        reward -= rewards["invalid_action_penalty"] * float(today["invalid_action"])
        final_day = self.day >= self.config["calendar"]["episode_days"]
        if final_day:
            reward += rewards["final_debt_weight"] * self.total_debt() / money_unit
        bankrupt = self.cash < 0
        if bankrupt:
            reward -= rewards["bankruptcy_penalty"]
            today["events"].append("Bankrupt: cash fell below zero")
        today["bankrupt"] = bool(bankrupt)
        today["end_reason"] = end_reason(bankrupt, final_day)
        return float(reward), bool(bankrupt or final_day), False

    def _observation(self):
        """Build the 23 value observation in its fixed order, with every value clipped to [0, 1]."""
        obs_config = self.config["observation"]
        money_scale = self.config["money_scale"]
        month_length = self.config["calendar"]["month_length"]
        recent_debt, old_debt = self.debt_by_age()
        debt_norm = obs_config["debt_norm"] * money_scale
        values = np.concatenate([
            self.stock / self.stock_caps,
            self.pending[:RESTOCKABLE_SLOTS] / self.stock_caps[:RESTOCKABLE_SLOTS],
            [
                self.cash / (obs_config["cash_norm"] * money_scale),
                recent_debt / debt_norm,
                old_debt / debt_norm,
                self.trust,
                self.price_band / HIGH_BAND,
                float(self.credit_open),
            ],
            self._average_sales() / self.sales_norms,
            [
                self._haggle_success_rate(),
                self._default_share(),
                self.weekday / (DAYS_IN_WEEK - 1),
                (month_length - self.day_of_month) / month_length,
                self.day / self.config["calendar"]["episode_days"],
                self.lost_today / obs_config["lost_customers_norm"],
            ],
        ])
        return np.clip(values, 0.0, 1.0).astype(np.float32)

    def _average_sales(self):
        """Return average units sold per slot over the recent window, or zeros before any sales."""
        if not self.sales_history:
            return np.zeros(NUM_SLOTS)
        return np.mean(np.array(self.sales_history), axis=0)

    def _haggle_success_rate(self):
        """Return the share of recent haggles that ended in a sale, or a neutral value when none."""
        haggles = sum(count for count, _ in self.haggle_history)
        if haggles == 0:
            return self.config["observation"]["no_haggle_rate"]
        return sum(success for _, success in self.haggle_history) / haggles

    def _default_share(self):
        """Return the amount defaulted over the amount lent in the recent window, or zero when none lent."""
        lent = sum(amount for amount, _ in self.lending_history)
        if lent == 0:
            return 0.0
        return sum(defaulted for _, defaulted in self.lending_history) / lent

    def _base_info(self):
        """Return the info fields that describe the shop at any moment."""
        return {
            "day": int(self.day),
            "cash": float(self.cash),
            "stock": self.stock.tolist(),
            "pending": self.pending.tolist(),
            "trust": float(self.trust),
            "debt_outstanding": self.total_debt(),
        }

    def _step_info(self, today):
        """Combine the shop snapshot with the day's counters and events for logs and rendering."""
        info = self._base_info()
        info.update(today)
        info["units_sold"] = today["units_sold"].tolist()
        return info


def end_reason(bankrupt, final_day):
    """Return why the episode ended today: "bankrupt", "day_90", or None while it continues."""
    if bankrupt:
        return "bankrupt"
    if final_day:
        return "day_90"
    return None


def settle_haggle(ask, buyer_max, floor_ratio):
    """Return the sale price of one haggle: the ask, the buyer's maximum above the floor, or None."""
    if buyer_max >= ask:
        return ask
    if buyer_max >= ask * floor_ratio:
        return buyer_max
    return None
