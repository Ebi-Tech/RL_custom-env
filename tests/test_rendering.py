"""Checks on the market street simulation: frames, scores, placeholders, credited assets and the server."""
import json
import threading
import urllib.request

import numpy as np
import pytest

import play
from environment.rendering import (
    SHOP_SPECS,
    MarketRunner,
    action_distribution,
    load_shops,
    to_jsonable,
    top_scores,
)

SEED = 1000
EPISODE_DAYS = 90
SHOP_COUNT = 5
FAST_SPEED = 100000
SHOP_KEYS = {
    "id", "name", "action", "action_label", "reward", "cumulative_return", "state", "net_worth", "events",
    "sales", "lost_customers", "price_walkaways", "haggles", "jara_count", "credit_given", "repayments",
    "defaults", "trust", "price_band", "credit_policy", "scores",
}


@pytest.fixture(scope="module")
def shops():
    """Load the five shops once from models/best/."""
    return load_shops()


@pytest.fixture
def server_url():
    """Run a play.py server on a free port in a background thread."""
    server = play.make_server(seed=SEED, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def read_events(body):
    """Split a Server-Sent Events body into (event name, data) pairs, with "message" for unnamed events."""
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        name = next((line[len("event: "):] for line in lines if line.startswith("event: ")), "message")
        data = next(line[len("data: "):] for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


def test_full_market_episode_streams_every_day(server_url):
    """The stream delivers 90 day frames, each with five shops and a leaderboard, then one end event."""
    url = f"{server_url}/api/stream?seed={SEED}&speed={FAST_SPEED}"
    with urllib.request.urlopen(url, timeout=120) as response:
        events = read_events(response.read().decode())
    days = [data for name, data in events if name == "message"]
    ends = [data for name, data in events if name == "end"]
    assert [frame["day"] for frame in days] == list(range(1, EPISODE_DAYS + 1))
    for frame in days:
        assert len(frame["shops"]) == SHOP_COUNT
        assert len(frame["leaderboard"]) == SHOP_COUNT
        for shop in frame["shops"]:
            assert SHOP_KEYS <= set(shop), SHOP_KEYS - set(shop)
        worths = [row["net_worth"] for row in frame["leaderboard"]]
        assert worths == sorted(worths, reverse=True)
    assert len(ends) == 1
    assert len(ends[0]["leaderboard"]) == SHOP_COUNT


def test_top_scores_come_from_a_distribution_that_sums_to_one(shops):
    """For every loaded agent the softmax over all 13 actions sums to 1, and the top 3 are its largest parts."""
    agents = [shop for shop in shops if shop["module"] is not None]
    assert agents, "no checkpoints in models/best/"
    runner = MarketRunner(shops, seed=SEED)
    for agent, lane in ((shop, lane) for shop, lane in zip(shops, runner.lanes) if shop["module"] is not None):
        distribution = action_distribution(agent["module"], lane["obs"])
        best = top_scores(distribution)
        assert distribution.sum() == pytest.approx(1.0)
        assert len(best) == 3
        assert [entry["score"] for entry in best] == sorted(np.sort(distribution)[-3:].tolist(), reverse=True)
        assert sum(entry["score"] for entry in best) <= 1.0 + 1e-9
    frame = next(iter(runner.frames()))
    for shop_frame, shop in zip(frame["shops"], shops):
        assert (shop_frame["scores"] is None) == (shop["module"] is None)


def test_missing_checkpoints_fall_back_to_placeholders(tmp_path):
    """With no checkpoints, the four agent shops run rule_policy, are labelled placeholders, and a day runs."""
    shops = load_shops(tmp_path)
    placeholders = [shop for shop in shops if shop["placeholder"]]
    assert len(placeholders) == len([spec for spec in SHOP_SPECS if spec["checkpoint"]])
    assert all(shop["name"].endswith("(placeholder)") for shop in placeholders)
    frame = next(iter(MarketRunner(shops, seed=SEED).frames()))
    assert json.loads(json.dumps(to_jsonable(frame)))["day"] == 1


def test_every_shipped_asset_is_credited():
    """Each texture and vendor file under assets/web/ is named, with its source, in assets/CREDITS.md."""
    credits = (play.ASSETS_DIR / "CREDITS.md").read_text()
    shipped = [path for folder in ("textures", "vendor") for path in (play.ASSETS_DIR / "web" / folder).iterdir()
               if path.name != "LICENSE"]
    assert shipped
    assert [path.name for path in shipped if f"`{path.name}`" not in credits] == []


def test_textures_are_served_as_images(server_url):
    """The page can fetch its photo-scanned textures, sent with an image content type."""
    with urllib.request.urlopen(f"{server_url}/assets/web/textures/ground_color.jpg", timeout=10) as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == "image/jpeg"


def test_meta_endpoint_returns_json(server_url):
    """GET /api/meta answers 200 with JSON naming the five shops."""
    with urllib.request.urlopen(f"{server_url}/api/meta", timeout=10) as response:
        assert response.status == 200
        meta = json.loads(response.read())
    assert len(meta["shops"]) == SHOP_COUNT
    assert meta["episode_days"] == EPISODE_DAYS
