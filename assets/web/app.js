import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

const MS_PER_DAY_AT_1X = 2000;
const MAX_CUSTOMERS_SHOWN = 12;
const CUSTOMERS_PER_FIGURE = 3;
const LOW_STOCK_SHARE = 0.15;
const LOG_DAYS_KEPT = 120;
const DOOR = new THREE.Vector3(6.6, 0, 2.2);
const ENTRY = new THREE.Vector3(4.2, 0, 2.6);
const COLORS = {
  floor: 0xc9875a, wall: 0xe9c48f, wallTrim: 0xb86b3c, wood: 0x8a5a36, woodDark: 0x5e3b22,
  counterTop: 0xd9a36a, noodles: 0xf0b13c, soap: 0xa9d8e6, sack: 0xe3d2a8, rice: 0xfaf3e1,
  bowl: 0x2f6f9f, cashBox: 0x3c7d5a, notes: 0x9ccf8c, book: 0xa63d2e, skin: 0x7a4a2c,
  wrapper: 0xd9573b, wrapperAlt: 0xf2c14e, red: 0xe5533d,
};
const CLOTHES = [0x3d7ea6, 0x6a994e, 0xbc4b51, 0x8e6bbf, 0xe09f3e, 0x2a9d8f, 0x9c6644, 0x577590];
const SHELVES = {
  indomie: { x: -3.2, slots: 60, tiers: 3, cols: 10, rows: 2, size: [0.22, 0.17, 0.2], shape: "box", color: COLORS.noodles },
  soap: { x: 0, slots: 48, tiers: 3, cols: 8, rows: 2, size: [0.24, 0.1, 0.15], shape: "box", color: COLORS.soap },
  rice_bag: { x: 3.2, slots: 9, tiers: 3, cols: 3, rows: 1, size: [0.24, 0.5, 0.24], shape: "sack", color: COLORS.sack },
};
const SHELF_WIDTH = 2.6;
const SHELF_DEPTH = 0.75;
const SHELF_Z = -2.45;
const TIER_HEIGHTS = [0.32, 1.12, 1.92];
const COUNTER = { x: 0, z: 0.95, width: 4.2, depth: 0.9, height: 1.0 };

const app = {
  meta: null,
  seed: 1000,
  speed: 1,
  playing: true,
  source: null,
  queue: [],
  lastReceivedDay: 0,
  ended: false,
  summary: null,
  current: null,
  dayElapsed: 0,
  shown: {},
  history: { agent: [], rule: [] },
  lowStock: {},
  stockShare: {},
};

const scene = new THREE.Scene();
const container = document.getElementById("scene");
const labels = document.getElementById("labels");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.insertBefore(renderer.domElement, labels);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
camera.position.set(6.8, 5.6, 10.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.1, 0);
controls.enableDamping = true;
controls.maxPolarAngle = 1.45;
controls.minDistance = 5;
controls.maxDistance = 22;

const clock = new THREE.Clock();
const scratch = new THREE.Object3D();

function material(color, extra = {}) {
  /** Return a flat shaded material in one of the shop colours. */
  return new THREE.MeshStandardMaterial({ color, flatShading: true, roughness: 0.85, metalness: 0.02, ...extra });
}

function addBox(size, position, color, parent = scene) {
  /** Add a shadow casting box of the given size and position, and return it. */
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material(color));
  mesh.position.set(...position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function buildLights() {
  /** Add warm sky light, a soft fill and a sun that casts shadows into the shop. */
  scene.background = new THREE.Color(0xf6d7a7);
  scene.fog = new THREE.Fog(0xf6d7a7, 22, 40);
  scene.add(new THREE.HemisphereLight(0xfff1dc, 0x8a5a36, 1.1));
  const sun = new THREE.DirectionalLight(0xffe2b8, 1.6);
  sun.position.set(6, 10, 7);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -9, right: 9, top: 9, bottom: -9 });
  scene.add(sun);
}

function buildRoom() {
  /** Build the floor, the walls with a doorway on the right, and a step outside the door. */
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 10), material(COLORS.floor));
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0, 1.5);
  floor.receiveShadow = true;
  scene.add(floor);
  addBox([11.4, 3.6, 0.2], [0, 1.8, -3.1], COLORS.wall);
  addBox([11.4, 0.35, 0.24], [0, 0.17, -2.98], COLORS.wallTrim);
  addBox([0.2, 3.6, 6.4], [-5.6, 1.8, 0.1], COLORS.wall);
  addBox([0.2, 3.6, 4.2], [5.6, 1.8, -1.0], COLORS.wall);
  addBox([0.2, 0.5, 2.2], [5.6, 3.35, 2.2], COLORS.wall);
  addBox([1.6, 0.06, 2.2], [6.4, 0.03, 2.2], COLORS.wallTrim);
}

function buildShelf(config) {
  /** Build one wooden shelf unit and return its boards, so low stock can make them pulse. */
  const boards = [];
  for (const height of TIER_HEIGHTS) {
    boards.push(addBox([SHELF_WIDTH, 0.08, SHELF_DEPTH], [config.x, height - 0.04, SHELF_Z], COLORS.wood));
  }
  for (const side of [-1, 1]) {
    boards.push(addBox([0.08, 2.5, SHELF_DEPTH], [config.x + side * SHELF_WIDTH / 2, 1.25, SHELF_Z], COLORS.woodDark));
  }
  addBox([SHELF_WIDTH, 2.5, 0.05], [config.x, 1.25, SHELF_Z - SHELF_DEPTH / 2], COLORS.woodDark);
  return boards;
}

function slotPositions(config) {
  /** Return the resting position of every item slot on a shelf, bottom tier and back row first. */
  const positions = [];
  const [width, height, depth] = config.size;
  const spanX = SHELF_WIDTH - 0.3;
  for (let tier = 0; tier < config.tiers; tier += 1) {
    for (let row = 0; row < config.rows; row += 1) {
      for (let col = 0; col < config.cols; col += 1) {
        const x = config.x - spanX / 2 + (col + 0.5) * (spanX / config.cols);
        const z = SHELF_Z - 0.15 + row * (depth + 0.08) - (config.rows - 1) * 0.04;
        positions.push(new THREE.Vector3(x, TIER_HEIGHTS[tier] + height / 2, z));
      }
    }
  }
  return positions.slice(0, config.slots);
}

function itemGeometry(config) {
  /** Return a box for packs and bars, or a squat low-poly cylinder for rice sacks. */
  const [width, height, depth] = config.size;
  if (config.shape === "sack") return new THREE.CylinderGeometry(width * 0.85, width, height, 7);
  return new THREE.BoxGeometry(width, height, depth);
}

function buildShelves() {
  /** Build the three shelves with one instanced mesh of items each, and their floating labels. */
  const shelves = {};
  for (const [key, config] of Object.entries(SHELVES)) {
    const mesh = new THREE.InstancedMesh(itemGeometry(config), material(config.color), config.slots);
    mesh.castShadow = true;
    mesh.count = 0;
    scene.add(mesh);
    const label = makeLabel("shelf-label");
    shelves[key] = { config, mesh, slots: slotPositions(config), boards: buildShelf(config), label,
      anchor: new THREE.Vector3(config.x, 2.75, SHELF_Z) };
  }
  return shelves;
}

function buildCounter() {
  /** Build the counter with the rice bowl, the cash box with its notes, and the debt book on top. */
  const top = COUNTER.height;
  addBox([COUNTER.width, top - 0.08, COUNTER.depth], [COUNTER.x, (top - 0.08) / 2, COUNTER.z], COLORS.woodDark);
  addBox([COUNTER.width + 0.2, 0.08, COUNTER.depth + 0.15], [COUNTER.x, top - 0.04, COUNTER.z], COLORS.counterTop);
  const bowl = new THREE.Mesh(new THREE.CylinderGeometry(0.46, 0.32, 0.36, 12, 1, true),
    material(COLORS.bowl, { side: THREE.DoubleSide }));
  bowl.position.set(-1.35, top + 0.18, COUNTER.z);
  bowl.castShadow = true;
  scene.add(bowl);
  addBox([0.62, 0.02, 0.62], [-1.35, top + 0.01, COUNTER.z], COLORS.bowl);
  const rice = new THREE.Mesh(new THREE.CylinderGeometry(0.43, 0.33, 1, 12), material(COLORS.rice));
  rice.position.set(-1.35, top, COUNTER.z);
  scene.add(rice);
  const cashBox = addBox([0.7, 0.32, 0.5], [1.4, top + 0.16, COUNTER.z], COLORS.cashBox);
  const notes = addBox([0.5, 1, 0.3], [1.4, top + 0.32, COUNTER.z], COLORS.notes);
  const book = addBox([0.55, 1, 0.42], [0.35, top, COUNTER.z + 0.05], COLORS.book);
  addBox([0.5, 0.02, 0.38], [0.35, top, COUNTER.z + 0.05], 0xf4ead5).name = "pages";
  return { rice, cashBox, notes, book, top, bowlAnchor: new THREE.Vector3(-1.35, top + 0.6, COUNTER.z),
    bubbleAnchor: new THREE.Vector3(0.2, 2.45, COUNTER.z + 0.3) };
}

function buildPerson(clothes, skin = COLORS.skin) {
  /** Return a simple capsule figure with a head and a small carried bag. */
  const group = new THREE.Group();
  const bodyMaterial = material(clothes);
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 0.55, 3, 8), bodyMaterial);
  body.position.y = 0.5;
  body.castShadow = true;
  const head = new THREE.Mesh(new THREE.IcosahedronGeometry(0.17, 1), material(skin));
  head.position.y = 1.08;
  head.castShadow = true;
  const bag = addBox([0.18, 0.16, 0.12], [0.26, 0.45, 0.05], 0xf4ead5, group);
  bag.visible = false;
  group.add(body, head);
  return { group, body, bodyMaterial, bag, baseColor: new THREE.Color(clothes) };
}

function buildShopkeeper() {
  /** Put the shopkeeper behind the counter in a bright wrapper with a head tie. */
  const keeper = buildPerson(COLORS.wrapper);
  keeper.group.scale.setScalar(1.15);
  keeper.group.position.set(-0.4, 0, COUNTER.z - 0.95);
  const tie = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.22, 6), material(COLORS.wrapperAlt));
  tie.position.y = 1.27;
  keeper.group.add(tie);
  scene.add(keeper.group);
  return keeper;
}

function buildCustomers() {
  /** Create the pool of customer figures, hidden outside the door until a day needs them. */
  return Array.from({ length: MAX_CUSTOMERS_SHOWN }, (_, index) => {
    const person = buildPerson(CLOTHES[index % CLOTHES.length]);
    person.group.visible = false;
    scene.add(person.group);
    return person;
  });
}

function signTexture(text, background, color = "#2b1d14") {
  /** Draw sign text on a canvas and return it as a texture. */
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 160;
  const context = canvas.getContext("2d");
  context.fillStyle = background;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "rgba(43,29,20,0.6)";
  context.lineWidth = 10;
  context.strokeRect(5, 5, canvas.width - 10, canvas.height - 10);
  context.fillStyle = color;
  context.font = "700 64px system-ui, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, canvas.width / 2, canvas.height / 2 + 4);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function buildSign(x) {
  /** Hang one sign above the counter and return its mesh, whose texture is swapped as state changes. */
  const sign = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 0.53), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  sign.position.set(x, 2.95, COUNTER.z - 0.2);
  scene.add(sign);
  for (const side of [-0.7, 0.7]) addBox([0.03, 0.5, 0.03], [x + side, 3.45, COUNTER.z - 0.2], COLORS.woodDark);
  return sign;
}

function makeLabel(className) {
  /** Create an HTML label that follows a point in the scene. */
  const element = document.createElement("div");
  element.className = `label ${className}`;
  labels.appendChild(element);
  return element;
}

buildLights();
buildRoom();
const shelves = buildShelves();
const counter = buildCounter();
const keeper = buildShopkeeper();
const customers = buildCustomers();
const bandSign = buildSign(-1.0);
const creditSign = buildSign(1.0);
const bubble = makeLabel("bubble");
const jaraFloat = makeLabel("float");
jaraFloat.textContent = "+ jara";
const signCache = {};

const formatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const money = (value) => `₦${formatter.format(value)}`;
const signed = (value) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
const el = (id) => document.getElementById(id);

function setSign(sign, key, text, background, color) {
  /** Show a cached sign texture for this text, creating it the first time. */
  if (!signCache[key]) signCache[key] = signTexture(text, background, color);
  sign.material.map = signCache[key];
  sign.material.needsUpdate = true;
}

function applySigns(state) {
  /** Colour the price sign by band and word the credit sign by policy. */
  const band = state.price_band;
  const bandStyle = { low: ["#8fd49b", "#1d3b22"], normal: ["#d9d4cc", "#2b1d14"], high: ["#ef6f5e", "#3a0f08"] }[band];
  setSign(bandSign, `band-${band}`, `Price: ${band.toUpperCase()}`, ...bandStyle);
  const open = state.credit_policy === "open";
  setSign(creditSign, `credit-${open}`, open ? "Open credit" : "Cash only", open ? "#9cc9e6" : "#f4ead5");
}

function itemCount(key, stock) {
  /** Return how many item models stand for a stock level: one per rice sack, scaled for packs and bars. */
  const config = SHELVES[key];
  if (key === "rice_bag") return Math.min(config.slots, Math.round(stock));
  const cap = productCap(key);
  const count = Math.round((stock / cap) * config.slots);
  return Math.min(config.slots, stock > 0 ? Math.max(1, count) : 0);
}

function productCap(key) {
  /** Return a product's storage cap from the metadata. */
  return app.meta.products.find((product) => product.key === key).cap;
}

function productName(key) {
  /** Return a product's display name from the metadata. */
  return app.meta.products.find((product) => product.key === key).name;
}

function layoutShelf(key, before, after, removeProgress, addProgress) {
  /** Place a shelf's item instances: sold items drop and shrink, newly arrived ones grow in. */
  const shelf = shelves[key];
  const total = Math.max(before, after);
  for (let index = 0; index < total; index += 1) {
    const slot = shelf.slots[index];
    let scale = 1;
    let drop = 0;
    if (index >= after) {
      scale = 1 - removeProgress;
      drop = removeProgress * 0.35;
    } else if (index >= before) {
      scale = addProgress;
    }
    scratch.position.set(slot.x, slot.y - drop, slot.z);
    scratch.scale.setScalar(Math.max(scale, 0.0001));
    scratch.rotation.set(0, (index % 3) * 0.12 - 0.12, 0);
    scratch.updateMatrix();
    shelf.mesh.setMatrixAt(index, scratch.matrix);
  }
  shelf.mesh.count = total;
  shelf.mesh.instanceMatrix.needsUpdate = true;
}

function applyCounterProps(state) {
  /** Size the loose rice level, the cash notes and the debt book to the shop state. */
  const riceShare = Math.min(state.stock.loose_rice / productCap("loose_rice"), 1);
  const riceHeight = 0.02 + riceShare * 0.32;
  counter.rice.scale.y = riceHeight;
  counter.rice.position.y = counter.top + 0.02 + riceHeight / 2;
  const cashShare = Math.min(Math.max(state.cash, 0) / (app.meta.start_cash * 4), 1);
  const notesHeight = 0.02 + cashShare * 0.45;
  counter.notes.scale.y = notesHeight;
  counter.notes.position.y = counter.top + 0.32 + notesHeight / 2;
  const debt = state.recent_debt + state.old_debt;
  const thickness = 0.03 + Math.min(debt / 150000, 1) * 0.4;
  counter.book.scale.y = thickness;
  counter.book.position.y = counter.top + thickness / 2;
}

function applyState(state, settle = true) {
  /** Show a shop state at rest: signs, counter props, shelf items and low-stock warnings. */
  applySigns(state);
  applyCounterProps(state);
  for (const key of Object.keys(SHELVES)) {
    const count = itemCount(key, state.stock[key]);
    if (settle) layoutShelf(key, count, count, 1, 1);
    app.shown[key] = count;
    const share = state.stock[key] / productCap(key);
    app.lowStock[key] = share < LOW_STOCK_SHARE;
    app.stockShare[key] = state.stock[key];
    const label = shelves[key].label;
    label.textContent = `${productName(key)}  ${formatter.format(state.stock[key])}`;
    label.classList.toggle("low", app.lowStock[key]);
  }
}

function planCustomers(frame) {
  /** Decide how many figures walk in today, which leave empty-handed, and where each stands. */
  if (!frame.customers) return [];
  const shown = Math.min(MAX_CUSTOMERS_SHOWN, Math.max(1, Math.round(frame.customers / CUSTOMERS_PER_FIGURE)));
  const walkedHaggles = frame.haggles.filter((haggle) => haggle.outcome === "walked_away").length;
  const failed = frame.lost_customers + frame.price_walkaways + frame.refused_credit + walkedHaggles;
  let unhappy = Math.round((shown * failed) / frame.customers);
  if (failed > 0) unhappy = Math.max(1, unhappy);
  unhappy = Math.min(unhappy, shown);
  return Array.from({ length: shown }, (_, index) => ({
    person: customers[index],
    unhappy: index >= shown - unhappy,
    start: (index / shown) * 0.3,
    spot: new THREE.Vector3(-1.6 + (index % 6) * 0.62, 0, COUNTER.z + 1.0 + Math.floor(index / 6) * 0.6),
  }));
}

function startDay(frame) {
  /** Begin animating one day: update the HUD and plan the customers, shelf changes and bubbles. */
  const before = { ...app.shown };
  const after = {};
  for (const key of Object.keys(SHELVES)) after[key] = itemCount(key, frame.state.stock[key]);
  app.current = { frame, before, after, plan: planCustomers(frame), haggle: frame.haggles[0] || null };
  app.dayElapsed = 0;
  for (const person of customers) person.group.visible = false;
  applySigns(frame.state);
  applyCounterProps(frame.state);
  updateHud(frame);
}

function finishDay() {
  /** Settle the scene at the end of the day's animation. */
  const { frame } = app.current;
  applyState(frame.state);
  for (const person of customers) person.group.visible = false;
  bubble.style.opacity = 0;
  jaraFloat.style.opacity = 0;
  app.current = null;
}

function walkPosition(plan, progress) {
  /** Return where a customer is at this point of the day, with a small walking bob. */
  const local = (progress - plan.start) / 0.65;
  if (local <= 0 || local >= 1) return null;
  const position = new THREE.Vector3();
  if (local < 0.4) {
    const t = local / 0.4;
    position.lerpVectors(t < 0.3 ? DOOR : ENTRY, t < 0.3 ? ENTRY : plan.spot, t < 0.3 ? t / 0.3 : (t - 0.3) / 0.7);
  } else if (local < 0.6) {
    position.copy(plan.spot);
  } else {
    const t = (local - 0.6) / 0.4;
    position.lerpVectors(plan.spot, DOOR, t);
  }
  const walking = local < 0.4 || local > 0.6;
  position.y = walking ? Math.abs(Math.sin(local * Math.PI * 14)) * 0.06 : 0;
  return { position, local };
}

function animateCustomers(progress) {
  /** Move today's customers in, pause them at the counter, then send them out, buyers with a bag. */
  for (const plan of app.current.plan) {
    const step = walkPosition(plan, progress);
    const { person } = plan;
    if (!step) {
      person.group.visible = false;
      continue;
    }
    person.group.visible = true;
    person.group.position.copy(step.position);
    const target = step.local < 0.6 ? plan.spot : DOOR;
    person.group.lookAt(target.x, 0, target.z);
    const served = step.local >= 0.5;
    person.bag.visible = served && !plan.unhappy;
    const red = served && plan.unhappy;
    person.bodyMaterial.color.copy(red ? new THREE.Color(COLORS.red) : person.baseColor);
  }
}

function animateShelves(progress) {
  /** Drop sold items during the buying part of the day and grow in deliveries at its start. */
  const removeProgress = THREE.MathUtils.clamp((progress - 0.3) / 0.4, 0, 1);
  const addProgress = THREE.MathUtils.clamp(progress / 0.2, 0, 1);
  for (const key of Object.keys(SHELVES)) {
    layoutShelf(key, app.current.before[key], app.current.after[key], removeProgress, addProgress);
  }
}

function animateBubble(progress) {
  /** Show a haggle over the counter: the offer first, then whether it sold or the buyer walked away. */
  const haggle = app.current.haggle;
  const visible = haggle && progress > 0.32 && progress < 0.85;
  bubble.style.opacity = visible ? 1 : 0;
  if (!visible) return;
  const decided = progress > 0.55;
  bubble.classList.toggle("sold", decided && haggle.outcome === "sold");
  bubble.classList.toggle("walked", decided && haggle.outcome !== "sold");
  if (!decided) {
    bubble.textContent = haggle.price ? `${haggle.item}: Offer ${formatter.format(haggle.price)}` : `${haggle.item}: haggling...`;
  } else {
    bubble.textContent = haggle.outcome === "sold" ? "Sold" : "Walked away";
  }
}

function animateJara(progress) {
  /** Float a small "+ jara" up from the rice bowl on days when loose rice buyers got an extra. */
  const active = app.current.frame.jara_count > 0 && progress > 0.4 && progress < 0.85;
  jaraFloat.style.opacity = active ? 1 - Math.max(0, (progress - 0.7) / 0.15) : 0;
  jaraFloat.dataset.rise = active ? String((progress - 0.4) * 1.2) : "0";
}

function pulseLowStock(time) {
  /** Make the boards of any low-stock shelf glow red in a slow pulse. */
  const glow = 0.35 + 0.35 * Math.sin(time * 4);
  for (const [key, shelf] of Object.entries(shelves)) {
    for (const board of shelf.boards) {
      board.material.emissive.setHex(app.lowStock[key] ? COLORS.red : 0x000000);
      board.material.emissiveIntensity = app.lowStock[key] ? glow : 0;
    }
  }
}

function idleKeeper(time) {
  /** Give the shopkeeper a gentle sway so the shop feels alive between days. */
  keeper.group.rotation.y = Math.sin(time * 0.8) * 0.25;
}

function placeLabel(element, point, liftPixels = 0) {
  /** Move an HTML label to the screen position of a scene point, hiding it when behind the camera. */
  const projected = point.clone().project(camera);
  const width = container.clientWidth;
  const height = container.clientHeight;
  const hidden = projected.z > 1;
  element.style.display = hidden ? "none" : "";
  const x = ((projected.x + 1) / 2) * width;
  const y = ((1 - projected.y) / 2) * height - liftPixels;
  element.style.transform = `translate(-50%, -100%) translate(${x}px, ${y}px)`;
}

function placeLabels() {
  /** Keep the shelf labels, the haggle bubble and the jara float pinned to their scene points. */
  for (const shelf of Object.values(shelves)) placeLabel(shelf.label, shelf.anchor);
  placeLabel(bubble, counter.bubbleAnchor);
  placeLabel(jaraFloat, counter.bowlAnchor, Number(jaraFloat.dataset.rise || 0) * 60);
}

function dayDuration() {
  /** Return how long one day lasts on screen at the current speed. */
  return MS_PER_DAY_AT_1X / app.speed;
}

function tick() {
  /** Advance the animation by one browser frame and start the next queued day when this one ends. */
  const delta = clock.getDelta();
  const time = clock.elapsedTime;
  if (app.current && app.playing) app.dayElapsed += delta * 1000;
  if (app.current) {
    const progress = Math.min(app.dayElapsed / dayDuration(), 1);
    animateCustomers(progress);
    animateShelves(progress);
    animateBubble(progress);
    animateJara(progress);
    if (progress >= 1) finishDay();
  }
  if (!app.current && app.playing && app.queue.length) startDay(app.queue.shift());
  if (!app.current && !app.queue.length && app.ended && app.summary) showEndCard();
  pulseLowStock(time);
  idleKeeper(time);
  controls.update();
  renderer.render(scene, camera);
  placeLabels();
  requestAnimationFrame(tick);
}

function updateHud(frame) {
  /** Fill the top bar, the side panel, the log and the comparison strip for one day. */
  const state = frame.state;
  const days = app.meta.episode_days;
  el("day-num").textContent = `Day ${frame.day}`;
  el("day-progress").style.width = `${(frame.day / days) * 100}%`;
  el("action-now").textContent = frame.action_label;
  for (const item of el("action-list").children) item.classList.toggle("active", item.dataset.action === frame.action);
  setNumber("stat-reward", signed(frame.reward), frame.reward);
  setNumber("stat-return", frame.cumulative_return.toFixed(2), frame.cumulative_return);
  el("stat-cash").textContent = money(state.cash);
  el("stat-networth").textContent = money(frame.net_worth);
  el("stat-debt").textContent = money(state.recent_debt + state.old_debt);
  el("stat-debt-split").textContent = `${money(state.recent_debt)} / ${money(state.old_debt)}`;
  updateTrust(state.trust);
  updateStatusChips(state);
  addLogDay(frame);
  updateStrip(frame);
}

function setNumber(id, text, value) {
  /** Write a number into the HUD, green when positive and red when negative. */
  const node = el(id);
  node.textContent = text;
  node.classList.toggle("good", value > 0);
  node.classList.toggle("bad", value < 0);
}

function updateTrust(trust) {
  /** Fill the trust bar and colour it from red at low trust to green at high trust. */
  el("trust-value").textContent = trust.toFixed(2);
  const fill = el("trust-fill");
  fill.style.width = `${trust * 100}%`;
  fill.style.background = trust < 0.4 ? "var(--bad)" : trust < 0.6 ? "var(--accent)" : "var(--good)";
}

function updateStatusChips(state) {
  /** Show the price band and the credit policy as coloured chips. */
  const band = el("band-chip");
  band.textContent = `Price ${state.price_band}`;
  band.className = `chip band-${state.price_band}`;
  const credit = el("credit-chip");
  const open = state.credit_policy === "open";
  credit.textContent = open ? "Open credit" : "Cash only";
  credit.className = `chip credit-${open ? "open" : "strict"}`;
}

function eventTone(text) {
  /** Return "bad" for losses and "good" for sales and repayments, so the log reads at a glance. */
  if (/lost|walked away|invalid|default|refused|bankrupt|waste/i.test(text)) return "bad";
  if (/sold|repaid|delivery arrived/i.test(text)) return "good";
  return "";
}

function addLogDay(frame) {
  /** Add the day's events to the top of the scrolling log, keeping only the latest days. */
  const block = document.createElement("div");
  block.className = "log-day";
  const head = document.createElement("div");
  head.className = "log-day-head";
  head.innerHTML = `Day ${frame.day} <span class="muted">${frame.action_label}</span>`;
  block.appendChild(head);
  const lines = frame.events.length ? frame.events : ["A quiet day"];
  for (const text of lines) {
    const line = document.createElement("div");
    line.className = `log-line ${eventTone(text)}`;
    line.textContent = text;
    block.appendChild(line);
  }
  const list = el("log-list");
  list.prepend(block);
  while (list.children.length > LOG_DAYS_KEPT) list.lastChild.remove();
}

function updateStrip(frame) {
  /** Update the agent and rule shop numbers and redraw the return chart. */
  setNumber("agent-return", frame.cumulative_return.toFixed(2), frame.cumulative_return);
  el("agent-networth").textContent = money(frame.net_worth);
  setNumber("rule-return", frame.rule.cumulative_return.toFixed(2), frame.rule.cumulative_return);
  el("rule-networth").textContent = money(frame.rule.net_worth);
  app.history.agent.push(frame.cumulative_return);
  app.history.rule.push(frame.rule.cumulative_return);
  drawChart();
}

function drawChart() {
  /** Draw both cumulative return lines over the episode on the 2D canvas, with a zero line. */
  const canvas = el("chart");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const all = [...app.history.agent, ...app.history.rule, 0];
  const low = Math.min(...all);
  const high = Math.max(...all, 1);
  const pad = 8;
  const days = app.meta.episode_days;
  const x = (day) => pad + (day / days) * (width - 2 * pad);
  const y = (value) => height - pad - ((value - low) / (high - low || 1)) * (height - 2 * pad);
  context.strokeStyle = "rgba(255,255,255,0.18)";
  context.setLineDash([4, 4]);
  context.beginPath();
  context.moveTo(pad, y(0));
  context.lineTo(width - pad, y(0));
  context.stroke();
  context.setLineDash([]);
  drawLine(context, app.history.rule, x, y, getComputedStyle(document.documentElement).getPropertyValue("--rule"));
  drawLine(context, app.history.agent, x, y, getComputedStyle(document.documentElement).getPropertyValue("--agent"));
  context.fillStyle = "rgba(251,239,223,0.6)";
  context.font = "11px system-ui, sans-serif";
  context.fillText("cumulative return", pad + 2, pad + 10);
}

function drawLine(context, values, x, y, color) {
  /** Draw one series as a line with a dot on its latest value. */
  if (!values.length) return;
  context.strokeStyle = color.trim();
  context.lineWidth = 2.5;
  context.beginPath();
  values.forEach((value, index) => (index ? context.lineTo(x(index + 1), y(value)) : context.moveTo(x(1), y(value))));
  context.stroke();
  context.fillStyle = color.trim();
  context.beginPath();
  context.arc(x(values.length), y(values[values.length - 1]), 3.5, 0, Math.PI * 2);
  context.fill();
}

function showEndCard() {
  /** Show the episode result comparing the agent with the rule shop. */
  const card = el("end-card");
  if (!card.hidden) return;
  const { agent, rule } = app.summary;
  const better = agent.return >= rule.return;
  el("end-body").innerHTML = `
    <div class="verdict ${better ? "good" : "bad"}">${app.meta.policy} ${better ? "beat" : "trailed"} the rule shop</div>
    <div class="muted">Seed ${app.summary.seed}, ${app.summary.days} days, ended by ${app.summary.end_reason}</div>
    <table>
      <tr><th></th><th>${app.meta.policy}</th><th>Rule shop</th></tr>
      <tr><td>Return</td><td>${agent.return.toFixed(2)}</td><td>${rule.return.toFixed(2)}</td></tr>
      <tr><td>Net worth</td><td>${money(agent.net_worth)}</td><td>${money(rule.net_worth)}</td></tr>
      <tr><td>Cash</td><td>${money(agent.cash)}</td><td>${money(rule.cash)}</td></tr>
      <tr><td>Unpaid debt</td><td>${money(agent.debt)}</td><td>${money(rule.debt)}</td></tr>
      <tr><td>Trust</td><td>${agent.trust.toFixed(2)}</td><td>${rule.trust.toFixed(2)}</td></tr>
    </table>`;
  card.hidden = false;
}

function setStatus(text) {
  /** Show or hide the small connection status pill. */
  const status = el("status");
  status.hidden = !text;
  status.textContent = text || "";
}

function closeStream() {
  /** Stop receiving days from the server. */
  if (app.source) app.source.close();
  app.source = null;
}

function connect(fromDay) {
  /** Open the day stream from a given day at the current seed and speed. */
  closeStream();
  const url = `/api/stream?seed=${app.seed}&speed=${app.speed}&from_day=${fromDay}`;
  const source = new EventSource(url);
  app.source = source;
  source.addEventListener("init", (event) => applyState(JSON.parse(event.data).state));
  source.onmessage = (event) => {
    const frame = JSON.parse(event.data);
    app.lastReceivedDay = frame.day;
    app.queue.push(frame);
    setStatus("");
  };
  source.addEventListener("end", (event) => {
    app.summary = JSON.parse(event.data);
    app.ended = true;
    closeStream();
  });
  source.onerror = () => {
    if (app.ended || app.source !== source) return;
    setStatus("Lost the connection to play.py. Is it still running?");
    closeStream();
  };
}

function resetEpisode() {
  /** Clear everything shown and start the episode for the seed in the input from day 1. */
  closeStream();
  app.seed = Number(el("seed-input").value) || app.meta.seed;
  Object.assign(app, { queue: [], lastReceivedDay: 0, ended: false, summary: null, current: null, dayElapsed: 0 });
  app.history = { agent: [], rule: [] };
  for (const key of Object.keys(SHELVES)) app.shown[key] = 0;
  el("log-list").replaceChildren();
  el("end-card").hidden = true;
  el("day-num").textContent = "Day 0";
  el("day-progress").style.width = "0";
  drawChart();
  setPlaying(true);
  connect(1);
}

function setPlaying(playing) {
  /** Pause or resume: pausing freezes the day and stops the stream, resuming picks up the next day. */
  app.playing = playing;
  el("play-btn").textContent = playing ? "Pause" : "Play";
  if (!playing) {
    closeStream();
  } else if (!app.ended && !app.source && app.meta) {
    connect(app.lastReceivedDay + 1);
  }
}

function setSpeed(speed) {
  /** Change the playback speed and ask the server to pace the next days to match. */
  app.speed = speed;
  for (const button of el("speed-group").children) button.classList.toggle("active", Number(button.dataset.speed) === speed);
  if (app.playing && !app.ended && app.source) connect(app.lastReceivedDay + 1);
}

function buildActionList() {
  /** List all 13 actions in the panel so today's choice can be highlighted. */
  const list = el("action-list");
  app.meta.actions.forEach((name, index) => {
    const item = document.createElement("li");
    item.dataset.action = name;
    item.textContent = app.meta.action_labels[index];
    list.appendChild(item);
  });
}

function resize() {
  /** Fit the renderer and camera to the scene container. */
  const width = container.clientWidth;
  const height = container.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  drawChart();
}

function bindControls() {
  /** Wire up play and pause, the speed buttons, restart and the seed input. */
  el("play-btn").addEventListener("click", () => setPlaying(!app.playing));
  for (const button of el("speed-group").children) {
    button.addEventListener("click", () => setSpeed(Number(button.dataset.speed)));
  }
  el("restart-btn").addEventListener("click", resetEpisode);
  el("end-restart").addEventListener("click", resetEpisode);
  el("seed-input").addEventListener("keydown", (event) => event.key === "Enter" && resetEpisode());
  window.addEventListener("resize", resize);
}

async function start() {
  /** Load the metadata, set up the HUD from the URL options, and start the first episode. */
  app.meta = await (await fetch("/api/meta")).json();
  const params = new URLSearchParams(window.location.search);
  app.seed = Number(params.get("seed")) || app.meta.seed;
  const speed = Number(params.get("speed"));
  el("seed-input").value = app.seed;
  el("day-total").textContent = app.meta.episode_days;
  el("policy-chip").textContent = `Policy: ${app.meta.policy}`;
  el("agent-name").textContent = app.meta.policy;
  buildActionList();
  bindControls();
  resize();
  setSpeed(speed > 0 ? speed : 1);
  resetEpisode();
  requestAnimationFrame(tick);
}

start().catch((error) => setStatus(`Could not start: ${error.message}`));
