"""Watch an agent run the Beyond Sale Shop in a 3D browser view, streamed live from the environment.

Run:  uv run play.py [--policy rule|random|<checkpoint.pt>] [--seed 1000] [--speed 1] [--no-browser]

A small standard-library HTTP server serves the web page and streams one Server-Sent Event per
trading day. The same days are printed in this terminal as they stream.
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

from environment.rendering import DEFAULT_SEED, EpisodeRunner, episode_meta, load_policy, to_jsonable

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
INDEX_PAGE = ASSETS_DIR / "web" / "index.html"
DEFAULT_CHECKPOINT = ROOT / "models" / "best" / "best_agent.pt"
DEFAULT_PORT = 8765
HOST = "127.0.0.1"
SECONDS_PER_DAY_AT_1X = 2.0
MAX_SPEED = 100000.0
MARKETS = ("default", "harsh", "price_scale")
CONTENT_TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".json": "application/json"}


def default_policy_spec():
    """Return the best saved agent when there is one, else the rule policy."""
    return str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.exists() else "rule"


def money(value):
    """Format a money amount with thousands separators and no decimals."""
    return f"{value:,.0f}"


def day_line(frame):
    """Return the one-line terminal summary of a streamed day."""
    state = frame["state"]
    return (
        f"day {frame['day']:>3}  {frame['action_label']:<24} r {frame['reward']:>+7.2f}  "
        f"ret {frame['cumulative_return']:>8.2f}  cash {money(state['cash']):>10}  "
        f"nw {money(frame['net_worth']):>10}  trust {state['trust']:.2f}  band {state['price_band']:<6} "
        f"credit {state['credit_policy']:<6} lost {frame['lost_customers']:>2}  "
        f"| rule ret {frame['rule']['cumulative_return']:>8.2f}"
    )


def summary_block(summary, policy_name):
    """Return the end of episode comparison of the agent and the rule shop as printable text."""
    agent, rule = summary["agent"], summary["rule"]
    rows = [
        ("return", f"{agent['return']:.2f}", f"{rule['return']:.2f}"),
        ("net worth", money(agent["net_worth"]), money(rule["net_worth"])),
        ("cash", money(agent["cash"]), money(rule["cash"])),
        ("unpaid debt", money(agent["debt"]), money(rule["debt"])),
        ("trust", f"{agent['trust']:.2f}", f"{rule['trust']:.2f}"),
    ]
    lines = [f"\nEpisode over: seed {summary['seed']}, {summary['days']} days, end: {summary['end_reason']}",
             f"  {'':<12} {policy_name:>14} {'rule shop':>14}"]
    lines += [f"  {name:<12} {mine:>14} {theirs:>14}" for name, mine, theirs in rows]
    return "\n".join(lines)


def sse_message(data, event=None):
    """Encode one Server-Sent Event, with an optional event name."""
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(to_jsonable(data))}\n\n".encode()


def query_number(query, name, default, kind):
    """Read one number from a parsed query string, falling back to the default when missing or bad."""
    try:
        return kind(query.get(name, [default])[0])
    except (TypeError, ValueError):
        return default


class ShopHandler(BaseHTTPRequestHandler):
    """Serve the web page, its static files, the episode metadata and the live day stream."""

    settings = {}

    def log_message(self, format, *args):
        """Keep the terminal for the day lines instead of one log line per request."""

    def do_GET(self):
        """Route one GET request to the page, a static file, the metadata or the stream."""
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self.send_file(INDEX_PAGE)
        elif url.path.startswith("/assets/"):
            self.send_static(url.path[len("/assets/"):])
        elif url.path == "/api/meta":
            self.send_json(episode_meta(self.settings["policy_name"], self.settings["seed"], self.settings["market"]))
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
        """Stream the episode as Server-Sent Events, one per day, paced by the speed in the query.

        from_day lets the page resume after a pause or a speed change: earlier days are replayed
        silently, so the episode is identical to an unbroken run.
        """
        seed = query_number(query, "seed", self.settings["seed"], int)
        speed = min(max(query_number(query, "speed", self.settings["speed"], float), 0.01), MAX_SPEED)
        from_day = max(query_number(query, "from_day", 1, int), 1)
        runner = EpisodeRunner(self.settings["policy_fn"], seed, self.settings["market"])
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
            print(f"\n--- streaming seed {runner.seed}, policy {self.settings['policy_name']} ---", flush=True)
            self.write_event(runner.initial_frame(), "init")
        for frame in runner.frames():
            if frame["day"] < from_day:
                continue
            self.write_event(frame)
            print(day_line(frame), flush=True)
            time.sleep(delay)
        summary = runner.summary()
        self.write_event(summary, "end")
        print(summary_block(summary, self.settings["policy_name"]), flush=True)

    def write_event(self, data, event=None):
        """Send one event and flush it so the browser receives it at once."""
        self.wfile.write(sse_message(data, event))
        self.wfile.flush()


def make_server(policy_spec="rule", seed=DEFAULT_SEED, market="default", speed=1.0, port=DEFAULT_PORT):
    """Build the HTTP server for one policy, without starting it; port 0 picks a free port."""
    policy_name, policy_fn = load_policy(policy_spec)
    handler = type("ConfiguredShopHandler", (ShopHandler,), {})
    handler.settings = {"policy_name": policy_name, "policy_fn": policy_fn, "seed": seed, "market": market,
                        "speed": speed}
    server = ThreadingHTTPServer((HOST, port), handler)
    server.daemon_threads = True
    return server


def parse_args(argv=None):
    """Read the play.py command line options."""
    parser = argparse.ArgumentParser(description="Watch the Beyond Sale Shop in a live 3D browser view.")
    parser.add_argument("--policy", default=None,
                        help="rule, random, or a checkpoint path (default: models/best/best_agent.pt, else rule)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="episode seed (default 1000, a test seed)")
    parser.add_argument("--market", choices=MARKETS, default="default")
    parser.add_argument("--speed", type=float, default=1.0, help="1 means 2 seconds per day")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    return parser.parse_args(argv)


def main(argv=None):
    """Start the server, open the browser unless asked not to, and serve until Ctrl+C."""
    args = parse_args(argv)
    policy_spec = args.policy or default_policy_spec()
    server = make_server(policy_spec, args.seed, args.market, args.speed, args.port)
    url = f"http://{HOST}:{server.server_address[1]}/?seed={args.seed}&speed={args.speed:g}"
    print(f"Beyond Sale Shop: policy {server.RequestHandlerClass.settings['policy_name']}, seed {args.seed}, "
          f"market {args.market}")
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
