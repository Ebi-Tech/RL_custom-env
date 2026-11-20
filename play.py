"""Watch five shops trade side by side on a 3D market street, streamed live from the environment.

Run:  uv run play.py [--seed 1000] [--market default|harsh] [--speed 1] [--no-browser]

The four trained agents in models/best/ and Mama Ngozi, a typical owner following the rule
policy, run the same 90 days in the same market. A small standard-library HTTP server serves
the web page and streams one Server-Sent Event per day; each day is also printed here.
"""
import argparse
import json
import mimetypes
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from environment.rendering import (
    DEFAULT_MODEL_DIR,
    DEFAULT_SEED,
    MARKETS,
    TYPICAL_OWNER,
    MarketRunner,
    load_shops,
    market_meta,
    to_jsonable,
)

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
INDEX_PAGE = ASSETS_DIR / "web" / "index.html"
DEFAULT_PORT = 8765
HOST = "127.0.0.1"
SECONDS_PER_DAY_AT_1X = 8.0
MAX_SPEED = 100000.0
CONTENT_TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".json": "application/json"}


def money(value):
    """Format a money amount with thousands separators and no decimals."""
    return f"{value:,.0f}"


def day_block(frame):
    """Return the compact terminal block of one market day: one line per shop."""
    lines = [f"day {frame['day']:>3}  customers {frame['total_customers']:>4}  leader "
             f"{frame['leaderboard'][0]['name']}"]
    for shop in frame["shops"]:
        lines.append(
            f"   {shop['name'][:30]:<30} {shop['action_label']:<24} ret {shop['cumulative_return']:>8.2f}  "
            f"nw {money(shop['net_worth']):>10}  trust {shop['trust']:.2f}  lost {shop['turned_away']:>2}"
        )
    return "\n".join(lines)


def final_leaderboard(summary):
    """Return the end of episode leaderboard with each shop's difference from Mama Ngozi."""
    lines = [f"\nFinal leaderboard: seed {summary['seed']}, market {summary['market']}, {summary['days']} days",
             f"  {'#':<3}{'shop':<34}{'net worth':>12}{'vs Mama':>12}{'return':>9}{'vs Mama':>9}"
             f"{'turned away':>13}{'bad debt':>11}"]
    for row in summary["leaderboard"]:
        lines.append(
            f"  {row['rank']:<3}{row['name'][:33]:<34}{money(row['net_worth']):>12}"
            f"{row['net_worth_vs_owner']:>+12,.0f}{row['cumulative_return']:>9.2f}{row['return_vs_owner']:>+9.2f}"
            f"{row['turned_away_total']:>13}{money(row['bad_debt_total']):>11}"
        )
    lines.append("  Ran the same 90 days in the same market.")
    return "\n".join(lines)


def sse_message(data, event=None):
    """Encode one Server-Sent Event, with an optional event name."""
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(to_jsonable(data))}\n\n".encode()


def query_value(query, name, default, kind):
    """Read one value from a parsed query string, falling back to the default when missing or bad."""
    try:
        return kind(query.get(name, [default])[0])
    except (TypeError, ValueError):
        return default


class MarketHandler(BaseHTTPRequestHandler):
    """Serve the web page, its static files, the market metadata and the live day stream."""

    settings = {}

    def log_message(self, format, *args):
        """Keep the terminal for the day blocks instead of one log line per request."""

    def do_GET(self):
        """Route one GET request to the page, a static file, the metadata or the stream."""
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self.send_file(INDEX_PAGE)
        elif url.path.startswith("/assets/"):
            self.send_static(url.path[len("/assets/"):])
        elif url.path == "/api/meta":
            settings = self.settings
            self.send_json(market_meta(settings["shops"], settings["seed"], settings["market"]))
        elif url.path == "/api/stream":
            self.send_stream(parse_qs(url.query))
        else:
            self.send_error(404)

    def send_json(self, data):
        """Send a JSON response."""
        body = json.dumps(to_jsonable(data)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, relative):
        """Send a file from assets/, refusing any path that leaves that folder."""
        path = (ASSETS_DIR / relative).resolve()
        if not path.is_relative_to(ASSETS_DIR) or not path.is_file():
            self.send_error(404)
            return
        self.send_file(path)

    def send_file(self, path):
        """Send one file with a content type taken from its extension."""
        body = path.read_bytes()
        kind = CONTENT_TYPES.get(path.suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_stream(self, query):
        """Stream the market episode as Server-Sent Events, one per day, paced by the speed in the query.

        from_day lets the page resume after a pause or a speed change: earlier days are replayed
        silently, so the episode is identical to an unbroken run.
        """
        seed = query_value(query, "seed", self.settings["seed"], int)
        market = query_value(query, "market", self.settings["market"], str)
        if market not in MARKETS:
            self.send_error(400, f"market must be one of {MARKETS}")
            return
        speed = min(max(query_value(query, "speed", self.settings["speed"], float), 0.01), MAX_SPEED)
        from_day = max(query_value(query, "from_day", 1, int), 1)
        runner = MarketRunner(self.settings["shops"], seed, market)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.stream_days(runner, from_day, SECONDS_PER_DAY_AT_1X / speed)
        except (BrokenPipeError, ConnectionResetError):
            return

    def stream_days(self, runner, from_day, delay):
        """Write the init event, then each day from from_day onward, then the end summary."""
        if from_day == 1:
            print(f"\n--- market stream: seed {runner.seed}, market {runner.market} ---", flush=True)
            self.write_event(runner.initial_frame(), "init")
        for frame in runner.frames():
            if frame["day"] < from_day:
                continue
            self.write_event(frame)
            print(day_block(frame), flush=True)
            time.sleep(delay)
        summary = runner.summary()
        self.write_event(summary, "end")
        print(final_leaderboard(summary), flush=True)

    def write_event(self, data, event=None):
        """Send one event and flush it so the browser receives it at once."""
        self.wfile.write(sse_message(data, event))
        self.wfile.flush()


def make_server(seed=DEFAULT_SEED, market="default", speed=1.0, port=DEFAULT_PORT, model_dir=DEFAULT_MODEL_DIR):
    """Load the five shops and build the HTTP server, without starting it; port 0 picks a free port."""
    handler = type("ConfiguredMarketHandler", (MarketHandler,), {})
    handler.settings = {"shops": load_shops(model_dir), "seed": seed, "market": market, "speed": speed}
    server = ThreadingHTTPServer((HOST, port), handler)
    server.daemon_threads = True
    return server


def parse_args(argv=None):
    """Read the play.py command line options."""
    parser = argparse.ArgumentParser(description="Watch five shops trade on a live 3D market street.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="episode seed (default 1000, a test seed)")
    parser.add_argument("--market", choices=MARKETS, default="default")
    parser.add_argument("--speed", type=float, default=1.0, help="1 means 8 seconds per day")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--models", default=str(DEFAULT_MODEL_DIR), help="folder with dqn.pt, reinforce.pt, a2c.pt, ppo.pt")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    return parser.parse_args(argv)


def describe_shops(shops):
    """Return one line per shop saying what drives it."""
    lines = []
    for shop in shops:
        driver = "rule_policy (placeholder)" if shop["placeholder"] else (
            "rule_policy" if shop["kind"] == "rule" else f"{shop['checkpoint']} ({shop['score_label']})")
        lines.append(f"  {shop['name']:<34} {driver}")
    return "\n".join(lines)


def main(argv=None):
    """Start the server, open the browser unless asked not to, and serve until Ctrl+C."""
    args = parse_args(argv)
    server = make_server(args.seed, args.market, args.speed, args.port, args.models)
    url = f"http://{HOST}:{server.server_address[1]}/?seed={args.seed}&speed={args.speed:g}&market={args.market}"
    print(f"Kimironko Market, Nigerian Provisions Lane: seed {args.seed}, market {args.market}")
    print(describe_shops(server.RequestHandlerClass.settings["shops"]))
    print(f"Open {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
