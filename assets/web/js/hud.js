const ROW_HEIGHT = 46;
const LOG_DAYS_KEPT = 90;
const OWNER_ID = "mama";

const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
export const money = (value) => `₦${whole.format(value)}`;
const millions = (value) => `₦${(value / 1e6).toFixed(2)}M`;
const signedMoney = (value) => `${value >= 0 ? "+" : "-"}₦${whole.format(Math.abs(value))}`;
const signed = (value, digits = 2) => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
const el = (id) => document.getElementById(id);

const hud = { meta: null, rows: {}, cards: {}, history: {}, logs: {}, openShop: null, latest: null };

function make(tag, className, parent, text) {
  /** Create an element with a class, append it to a parent, and set its text. */
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}

export function initHud(meta, handlers) {
  /** Build the leaderboard rows, chart legend and shop cards for the five shops. */
  hud.meta = meta;
  const board = el("board");
  board.style.height = `${meta.shops.length * ROW_HEIGHT}px`;
  const legend = el("chart-legend");
  meta.shops.forEach((shop, index) => {
    const row = make("div", "board-row", board);
    row.innerHTML = `<span class="rank"></span><span class="dot"></span><span class="who"></span>
      <span class="worth"></span><span class="ret"></span><span class="diff"></span>`;
    row.querySelector(".dot").style.background = shop.color;
    row.querySelector(".who").textContent = shop.name;
    row.title = shop.name;
    row.style.transform = `translateY(${index * ROW_HEIGHT}px)`;
    row.addEventListener("click", () => handlers.onShopClick(index));
    hud.rows[shop.id] = row;
    const item = make("span", "legend-item", legend);
    item.innerHTML = `<i class="${shop.id === OWNER_ID ? "dashed" : ""}" style="--c:${shop.color}"></i>${shortName(shop)}`;
    hud.cards[shop.id] = buildCard(shop, index, handlers);
  });
  resetHud();
}

function shortName(shop) {
  /** Return a short legend name for a shop. */
  return shop.id === OWNER_ID ? "Mama Ngozi" : shop.id.toUpperCase();
}

function buildCard(shop, index, handlers) {
  /** Build one small shop card: action in plain words, trust bar, customers turned away and unpaid debt. */
  const card = make("button", "shop-card", el("cards"));
  card.type = "button";
  card.style.setProperty("--c", shop.color);
  card.title = `Press ${index + 1} to step inside`;
  card.innerHTML = `<div class="card-head"><span class="rank-badge">Rank -</span><span class="card-name"></span></div>
    <div class="card-action">waiting to open</div>
    <div class="card-trust"><span>trust</span><div class="bar"><div></div></div></div>
    <div class="card-stats"><span class="away"></span><span class="debt"></span></div>`;
  card.querySelector(".card-name").textContent = shop.name;
  card.addEventListener("click", () => handlers.onShopClick(index));
  return card;
}

export function resetHud() {
  /** Clear the chart, logs, cards and panels for a new episode. */
  for (const shop of hud.meta.shops) {
    hud.history[shop.id] = [];
    hud.logs[shop.id] = [];
  }
  hud.latest = null;
  el("end-card").hidden = true;
  el("day-num").textContent = "Day 0";
  el("day-progress").style.width = "0";
  drawChart();
}

export function showOpening(frame) {
  /** Show the market as it opens on day 0: starting net worth on the board and chart. */
  updateBoard(frame.leaderboard);
  for (const shop of frame.shops) hud.history[shop.id] = [shop.net_worth];
  drawChart();
}

export function updateDay(frame) {
  /** Update every HUD part for a new day. */
  hud.latest = frame;
  const days = hud.meta.episode_days;
  el("day-num").textContent = `Day ${frame.day}`;
  el("day-progress").style.width = `${(frame.day / days) * 100}%`;
  updateBoard(frame.leaderboard);
  for (const shop of frame.shops) {
    hud.history[shop.id].push(shop.net_worth);
    hud.logs[shop.id].unshift({ day: frame.day, action: shop.action_label, events: shop.events });
    hud.logs[shop.id].length = Math.min(hud.logs[shop.id].length, LOG_DAYS_KEPT);
    updateCard(shop);
  }
  drawChart();
  if (hud.openShop !== null) fillShopPanel(hud.openShop);
}

function updateBoard(leaderboard) {
  /** Move each leaderboard row to its rank with a slide, and fill its numbers. */
  for (const entry of leaderboard) {
    const row = hud.rows[entry.id];
    const previous = row.dataset.rank;
    row.style.transform = `translateY(${(entry.rank - 1) * ROW_HEIGHT}px)`;
    row.querySelector(".rank").textContent = entry.rank;
    row.querySelector(".worth").textContent = millions(entry.net_worth);
    row.querySelector(".ret").textContent = entry.cumulative_return.toFixed(1);
    const diff = row.querySelector(".diff");
    diff.textContent = entry.id === OWNER_ID ? "baseline" : signedMoney(entry.net_worth_vs_owner);
    diff.className = `diff ${entry.id === OWNER_ID ? "" : entry.net_worth_vs_owner >= 0 ? "good" : "bad"}`;
    if (previous && Number(previous) !== entry.rank) {
      row.classList.remove("moved");
      void row.offsetWidth;
      row.classList.add("moved");
    }
    row.dataset.rank = entry.rank;
    row.classList.toggle("leader", entry.rank === 1);
    const badge = hud.cards[entry.id].querySelector(".rank-badge");
    badge.textContent = `Rank ${entry.rank}`;
    badge.classList.toggle("first", entry.rank === 1);
  }
}

function updateCard(shop) {
  /** Fill one shop card with today's action, trust, customers turned away and unpaid debt. */
  const card = hud.cards[shop.id];
  card.querySelector(".card-action").textContent = shop.closed ? "closed (bankrupt)" : shop.action_label;
  const fill = card.querySelector(".bar div");
  fill.style.width = `${shop.trust * 100}%`;
  fill.style.background = shop.trust < 0.4 ? "var(--bad)" : shop.trust < 0.6 ? "var(--accent)" : "var(--good)";
  card.querySelector(".away").textContent = `${shop.turned_away} turned away`;
  card.querySelector(".debt").textContent = `debt ${money(shop.state.recent_debt + shop.state.old_debt)}`;
  card.classList.toggle("bad-day", shop.turned_away >= 5);
}

export function drawChart() {
  /** Draw every shop's net worth over the 90 days on labelled axes: solid lines for agents, dashed for Mama Ngozi. */
  const canvas = el("chart");
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height || !hud.meta) return;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const days = hud.meta.episode_days;
  const series = hud.meta.shops.map((shop) => ({ shop, values: hud.history[shop.id] || [] }));
  const all = series.flatMap((item) => item.values);
  const pad = { left: 54, right: 10, top: 10, bottom: 34 };
  const x = (day) => pad.left + (day / days) * (width - pad.left - pad.right);
  context.font = "11px system-ui, sans-serif";
  context.fillStyle = "rgba(251,239,223,0.7)";
  context.strokeStyle = "rgba(255,255,255,0.1)";
  context.textAlign = "center";
  for (let day = 0; day <= days; day += 15) {
    context.beginPath();
    context.moveTo(x(day), pad.top);
    context.lineTo(x(day), height - pad.bottom);
    context.stroke();
    context.fillText(String(day), x(day), height - pad.bottom + 14);
  }
  context.fillText("Day", (pad.left + width - pad.right) / 2, height - 4);
  if (!all.length) return;
  const low = Math.min(...all) * 0.985;
  const high = Math.max(...all) * 1.015;
  const y = (value) => height - pad.bottom - ((value - low) / (high - low || 1)) * (height - pad.top - pad.bottom);
  context.textAlign = "right";
  for (let step = 0; step <= 3; step += 1) {
    const value = low + ((high - low) * step) / 3;
    context.beginPath();
    context.moveTo(pad.left, y(value));
    context.lineTo(width - pad.right, y(value));
    context.stroke();
    context.fillText(millions(value), pad.left - 6, y(value) + 4);
  }
  const today = Math.max(...series.map((item) => item.values.length - 1), 0);
  context.strokeStyle = "rgba(242,165,65,0.45)";
  context.beginPath();
  context.moveTo(x(today), pad.top);
  context.lineTo(x(today), height - pad.bottom);
  context.stroke();
  for (const { shop, values } of series) {
    if (!values.length) continue;
    context.strokeStyle = shop.color;
    context.lineWidth = shop.id === OWNER_ID ? 2 : 2.4;
    context.setLineDash(shop.id === OWNER_ID ? [6, 4] : []);
    context.beginPath();
    values.forEach((value, day) => (day ? context.lineTo(x(day), y(value)) : context.moveTo(x(0), y(value))));
    context.stroke();
  }
  context.setLineDash([]);
}

export function openShopPanel(index) {
  /** Show the detail panel and event log of one shop, or hide it with null. */
  hud.openShop = index;
  el("shop-panel").hidden = index === null;
  if (index !== null) fillShopPanel(index);
}

function fillShopPanel(index) {
  /** Fill the shop panel with that shop's numbers, top scores and scrolling event log. */
  const meta = hud.meta.shops[index];
  const shop = hud.latest && hud.latest.shops[index];
  el("panel-name").textContent = meta.name;
  el("shop-panel").style.setProperty("--c", meta.color);
  el("panel-kind").textContent = meta.kind === "rule" ? "Typical owner: fixed rules of thumb" :
    `Trained ${meta.id.toUpperCase()} agent, greedy choice`;
  if (!shop) return;
  const state = shop.state;
  const owner = hud.latest.leaderboard.find((row) => row.id === meta.id);
  el("panel-stats").innerHTML = [
    ["Today", shop.action_label], ["Reward today", signed(shop.reward)], ["Return so far", shop.cumulative_return.toFixed(2)],
    ["Cash", money(state.cash)], ["Net worth", money(shop.net_worth)],
    ["vs Mama Ngozi", meta.id === OWNER_ID ? "baseline" : signedMoney(owner.net_worth_vs_owner)],
    ["Unpaid debt", `${money(state.recent_debt)} recent, ${money(state.old_debt)} old`],
    ["Trust", state.trust.toFixed(2)], ["Price band", state.price_band], ["Credit", state.credit_policy === "open" ? "open credit" : "cash only"],
    ["Customers today", `${shop.customers}, ${shop.turned_away} turned away`],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  const scores = el("panel-scores");
  scores.innerHTML = shop.scores ? `<h3>Top 3 choices (${shop.score_label})</h3>` + shop.scores.map((entry) =>
    `<div class="score"><span>${entry.label}</span><i style="width:${Math.max(2, entry.score * 100)}%"></i><b>${Math.round(entry.score * 100)}%</b></div>`).join("")
    : "<h3>How Mama Ngozi decides</h3><p>Restock when low, open a sack when the bowl runs low, chase old debts, keep cash safe.</p>";
  el("panel-log").innerHTML = hud.logs[meta.id].map((entry) => `<div class="log-day"><div class="log-head">Day ${entry.day}
    <span>${escapeHtml(entry.action)}</span></div>${(entry.events.length ? entry.events : ["A quiet day"]).map((line) =>
    `<div class="log-line ${tone(line)}">${escapeHtml(line)}</div>`).join("")}</div>`).join("");
}

function escapeHtml(text) {
  /** Escape text for safe use inside HTML. */
  return String(text).replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
}

function tone(text) {
  /** Return "bad" for losses and "good" for sales and repayments, so the log reads at a glance. */
  if (/lost|walked away|invalid|default|refused|bankrupt|waste|high prices/i.test(text)) return "bad";
  if (/sold|repaid|delivery arrived/i.test(text)) return "good";
  return "";
}

export function showEnd(summary) {
  /** Show the final leaderboard card for all five shops. */
  const rows = summary.leaderboard.map((row) => `<tr><td><i style="background:${row.color}"></i>${row.rank}. ${row.name}</td>
    <td>${money(row.net_worth)}</td><td>${row.cumulative_return.toFixed(2)}</td><td>${row.turned_away_total}</td>
    <td>${money(row.bad_debt_total)}</td></tr>`).join("");
  el("end-body").innerHTML = `<table><tr><th>Shop</th><th>Net worth</th><th>Return</th><th>Turned away</th><th>Bad debt</th></tr>
    ${rows}</table><p class="muted">Ran the same 90 days in the same market (seed ${summary.seed}, ${summary.market}).</p>`;
  el("end-card").hidden = false;
}

export function setPhase(text) {
  /** Show the time of day next to the day counter. */
  el("day-phase").textContent = text;
}

export function setStatus(text) {
  /** Show or hide the small status pill. */
  const status = el("status");
  status.hidden = !text;
  status.textContent = text || "";
}
