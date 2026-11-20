import * as THREE from "three";
import { mergeGeometries } from "three/addons/BufferGeometryUtils.js";
import { LAYOUT, seeded } from "./layout.js";
import { accentMaterial } from "./world.js";
import { StallFrame, buildStall } from "./stalls.js";
import { addPerson, randomLook } from "./people.js";
import { counterPrintTexture, createChalkboard, drawChalkboard, packTexture, signTexture } from "./textures.js";

export const MAX_CUSTOMERS_PER_SHOP = 8;
const SLAB = 0.12;
const COUNTER_Z = -2.75;
const COUNTER_TOP = SLAB + 1.0;
const SHELF_Z = -7.55;
const TIER_HEIGHTS = [0.35, 1.05, 1.75];
const SHELF_WIDTH = 1.7;
const LOW_STOCK_SHARE = 0.15;
const CUSTOMERS_PER_FIGURE = 4;
const LANE_Y = 0.04;
const KEEPER_Z = -3.6;
const GOODS = {
  indomie: { offset: -1.85, slots: 36, cols: 6, rows: 2, size: [0.22, 0.16, 0.2], shape: "box", art: "noodles" },
  soap: { offset: 0, slots: 36, cols: 6, rows: 2, size: [0.2, 0.09, 0.13], shape: "box", art: "soap" },
  rice_bag: { offset: 1.85, slots: 9, cols: 3, rows: 1, size: [0.22, 0.45, 0.22], shape: "sack", art: "rice" },
};
const PIDGIN = {
  offer: ["Abeg reduce am small", "Na last price be this?", "Customer, how much you go pay?", "Make we do am small"],
  sold: ["Oya, I go take am", "Deal! Wrap am for me", "Na fair price be that"],
  walked: ["E too cost, I dey go", "I no fit pay that one", "Abeg, I go come back"],
};
const BAND_STYLES = { low: ["#8fd49b", "#1d3b22"], normal: ["#efe7da", "#2b1d14"], high: ["#ef6f5e", "#3a0f08"] };
const RED = 0xe23a2a;

const holder = new THREE.Object3D();

function mergedMesh(parts, material) {
  /** Merge a list of [geometry, position] parts into one mesh. */
  const geometries = parts.map(([geometry, position]) => geometry.translate(...position).toNonIndexed());
  const mesh = new THREE.Mesh(mergeGeometries(geometries), material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function shelfUnit(world, x) {
  /** Build one shelf unit as a single mesh whose material can switch to the low-stock pulse. */
  const parts = TIER_HEIGHTS.map((height) => [new THREE.BoxGeometry(SHELF_WIDTH, 0.06, 0.6), [x, SLAB + height - 0.03, SHELF_Z]]);
  for (const side of [-1, 1]) parts.push([new THREE.BoxGeometry(0.06, 2.3, 0.6), [x + side * SHELF_WIDTH / 2, SLAB + 1.15, SHELF_Z]]);
  parts.push([new THREE.BoxGeometry(SHELF_WIDTH, 2.3, 0.03), [x, SLAB + 1.15, SHELF_Z - 0.3]]);
  return mergedMesh(parts, world.materials.shelfWood);
}

function slotPositions(x0, config) {
  /** Return the resting position of every item slot on a shelf unit, bottom tier and back row first. */
  const positions = [];
  const [, height, depth] = config.size;
  const span = SHELF_WIDTH - 0.2;
  const tiers = Math.ceil(config.slots / (config.cols * config.rows));
  for (let tier = 0; tier < tiers; tier += 1) {
    for (let row = 0; row < config.rows; row += 1) {
      for (let col = 0; col < config.cols; col += 1) {
        positions.push(new THREE.Vector3(
          x0 + config.offset - span / 2 + (col + 0.5) * (span / config.cols),
          SLAB + TIER_HEIGHTS[tier] + height / 2,
          SHELF_Z - 0.12 + row * (depth + 0.06),
        ));
      }
    }
  }
  return positions.slice(0, config.slots);
}

function goodsGeometry(config) {
  /** Return a box for packs and bars, or a squat sack for rice. */
  const [width, height, depth] = config.size;
  if (config.shape === "sack") return new THREE.CylinderGeometry(width * 0.85, width, height, 9);
  return new THREE.BoxGeometry(width, height, depth);
}

function buildStructure(world, batch, shop) {
  /** Queue one shop's lock-up stall, painted in its accent, with its counter, bowl and cash box. */
  const m = world.materials;
  const accent = accentMaterial(world, shop.color);
  const { width, front, back } = LAYOUT.shop;
  const { low, high } = LAYOUT.northRoof;
  const frame = new StallFrame(shop.x0, front, 1);
  buildStall(world, batch, shop.props, frame, { width, depth: front - back, front: low, back: high, overhang: 1.1,
    wall: "corrTeal", counter: false, fascia: accent, post: accent });
  const x0 = shop.x0;
  batch.box([width - 0.25, 3.6, 0.04], [x0, SLAB + 1.8, back + 0.08], m.shopPlaster);
  for (const side of [-1, 1]) batch.box([0.04, 3.6, 5.6], [x0 + side * (width / 2 - 0.09), SLAB + 1.8, (front + back) / 2], m.shopPlaster);
  batch.box([3.8, 1.0, 0.7], [x0, SLAB + 0.5, COUNTER_Z], m.planksDark);
  batch.box([4.0, 0.08, 0.85], [x0, COUNTER_TOP - 0.04, COUNTER_Z], m.planks);
  batch.add(new THREE.PlaneGeometry(3.7, 0.8), [x0, SLAB + 0.5, COUNTER_Z + 0.36], shop.counterMaterial);
  batch.add(new THREE.CylinderGeometry(0.42, 0.3, 0.32, 14, 1, true), [x0 - 1.35, COUNTER_TOP + 0.16, COUNTER_Z], m.bowl);
  batch.cylinder(0.3, 0.3, 0.02, [x0 - 1.35, COUNTER_TOP + 0.01, COUNTER_Z], m.bowl, 14);
  batch.box([0.64, 0.3, 0.46], [x0 + 1.3, COUNTER_TOP + 0.15, COUNTER_Z], m.cashBox);
  batch.box([0.5, 0.02, 0.38], [x0 + 0.35, COUNTER_TOP + 0.01, COUNTER_Z + 0.05], m.cream);
  batch.box([3.3, 1.5, 0.08], [x0, 3.05, back + 0.12], m.woodDark);
  batch.box([0.05, 0.05, 0.9], [x0 + 2.45, 3.75, -1.45], m.steelDark);
}

function doubleSign(world, map, size, position, turn) {
  /** Return two back-to-back planes so a sign reads correctly from both sides. */
  const material = new THREE.MeshLambertMaterial({ map });
  const group = new THREE.Group();
  for (const flip of [0, Math.PI]) {
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(...size), material);
    plane.rotation.y = turn + flip;
    plane.position.set(Math.sin(turn + flip) * 0.01, 0, Math.cos(turn + flip) * 0.01);
    group.add(plane);
  }
  group.position.set(...position);
  world.scene.add(group);
  return group;
}

function buildSigns(world, shop) {
  /** Add the big painted signboard, the blade sign over the lane, the price and credit plaques and the chalkboard. */
  const { front, back, width } = LAYOUT.shop;
  const nameArt = signTexture(shop.baseName, { background: shop.color, color: "#fffaf0", border: "#fffaf0",
    subtitle: shop.subtitle || "", font: "800 110px Georgia, serif" });
  const board = new THREE.Mesh(new THREE.PlaneGeometry(5.3, 1.0), new THREE.MeshLambertMaterial({ map: nameArt }));
  board.position.set(shop.x0, LAYOUT.northRoof.low + 0.55, front + 1.1 + 0.03);
  const bladeArt = signTexture(shop.baseName, { background: shop.color, color: "#fffaf0", border: "#fffaf0",
    width: 768, height: 300, font: "800 96px Georgia, serif", subtitle: shop.subtitle || "" });
  doubleSign(world, bladeArt, [1.3, 0.55], [shop.x0 + 2.45, 3.4, -1.45], Math.PI / 2);
  const band = new THREE.Mesh(new THREE.PlaneGeometry(1.05, 0.36), new THREE.MeshLambertMaterial({ color: 0xffffff }));
  band.position.set(shop.x0 - width / 2 + 0.6, 2.55, front + 1.0);
  const credit = new THREE.Mesh(new THREE.PlaneGeometry(1.05, 0.36), new THREE.MeshLambertMaterial({ color: 0xffffff }));
  credit.position.set(shop.x0 + width / 2 - 0.6, 2.55, front + 1.0);
  const chalkboard = createChalkboard();
  const chalk = new THREE.Mesh(new THREE.PlaneGeometry(3.1, 1.3), new THREE.MeshBasicMaterial({ map: chalkboard.texture }));
  chalk.position.set(shop.x0, 3.05, back + 0.17);
  world.scene.add(board, band, credit, chalk);
  return { band, credit, chalkboard };
}

function sharedGoods(world, count) {
  /** Create one instanced mesh per product shared by all five shops, so shelves cost three draw calls. */
  const meshes = {};
  for (const [key, config] of Object.entries(GOODS)) {
    const art = packTexture(config.art);
    const mesh = new THREE.InstancedMesh(goodsGeometry(config), new THREE.MeshLambertMaterial({ map: art, emissive: 0x6a6a6a, emissiveMap: art }), config.slots * count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.castShadow = true;
    mesh.frustumCulled = false;
    holder.position.set(0, -50, 0);
    holder.scale.setScalar(0.0001);
    holder.updateMatrix();
    for (let i = 0; i < config.slots * count; i += 1) mesh.setMatrixAt(i, holder.matrix);
    world.scene.add(mesh);
    meshes[key] = mesh;
  }
  return meshes;
}

function counterProps(world, count) {
  /** Create the instanced data props on every counter: the loose rice level, the cash notes and the debt book. */
  const make = (geometry, material) => {
    const mesh = new THREE.InstancedMesh(geometry, material, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    world.scene.add(mesh);
    return mesh;
  };
  return {
    rice: make(new THREE.CylinderGeometry(0.4, 0.3, 1, 14), world.materials.rice),
    notes: make(new THREE.BoxGeometry(0.46, 1, 0.28), world.materials.notes),
    books: make(new THREE.BoxGeometry(0.5, 1, 0.4), world.materials.book),
  };
}

function makeLabel(container, className) {
  /** Create an HTML label that follows a point in the scene. */
  const element = document.createElement("div");
  element.className = `label ${className}`;
  element.style.opacity = 0;
  container.appendChild(element);
  return element;
}

function shopperLook(shopIndex, slot) {
  /** Return a stable outfit for one of a shop's real customers, with free hands for the shop's bag. */
  let i = 0;
  const look = randomLook(() => seeded(shopIndex * 97 + slot * 13 + i++, 5));
  if (["bag", "jerrycan", "basket", "sack"].includes(look.carry)) look.carry = "none";
  return look;
}

export function buildShops(world, shopsMeta, labelsContainer, market, crowd) {
  /** Build the five shops in a row along the provisions lane, with their shoppers and keepers, and return their live parts. */
  const batch = market.batch;
  const goods = sharedGoods(world, shopsMeta.length);
  const counters = counterProps(world, shopsMeta.length);
  const shops = shopsMeta.map((meta, index) => {
    const x0 = LAYOUT.shopXs[index];
    const shop = { ...meta, index, x0, baseName: meta.name.replace(" (placeholder)", ""), color: meta.color, props: market.props };
    if (meta.placeholder) shop.subtitle = "placeholder: rule policy";
    const print = counterPrintTexture(meta.color);
    shop.counterMaterial = new THREE.MeshLambertMaterial({ map: print, emissive: 0x3a3a3a, emissiveMap: print });
    buildStructure(world, batch, shop);
    shop.signs = buildSigns(world, shop);
    shop.units = {};
    shop.goods = {};
    for (const [key, config] of Object.entries(GOODS)) {
      shop.units[key] = shelfUnit(world, x0 + config.offset);
      world.scene.add(shop.units[key]);
      shop.goods[key] = { mesh: goods[key], base: index * config.slots, config, slots: slotPositions(x0, config), shown: 0 };
    }
    shop.hitBox = new THREE.Mesh(new THREE.BoxGeometry(LAYOUT.shop.width, 4.4, 6.6), new THREE.MeshBasicMaterial({ visible: false }));
    shop.hitBox.position.set(x0, 2.2, -4.8);
    shop.hitBox.userData.shopIndex = index;
    world.scene.add(shop.hitBox);
    shop.bubble = makeLabel(labelsContainer, "bubble");
    shop.float = makeLabel(labelsContainer, "float");
    shop.float.textContent = "+ jara";
    shop.anchors = {
      bubble: new THREE.Vector3(x0 - 0.2, 2.35, -1.5),
      bowl: new THREE.Vector3(x0 - 1.35, COUNTER_TOP + 0.55, COUNTER_Z),
      inside: new THREE.Vector3(x0 + 1.7, 2.15, -0.9),
      look: new THREE.Vector3(x0 - 0.5, 1.75, LAYOUT.shop.back + 0.6),
      front: new THREE.Vector3(x0, 1.6, 1.2),
    };
    shop.lowStock = {};
    shop.state = null;
    shop.frame = null;
    shop.day = null;
    shop.shoppers = Array.from({ length: MAX_CUSTOMERS_PER_SHOP }, (_, slot) => {
      const person = addPerson(crowd, shopperLook(index, slot));
      person.active = false;
      person.y = LANE_Y;
      return person;
    });
    const keeperLook = randomLook(() => seeded(index * 31 + 7, 9), index % 2 ? "man" : "woman");
    Object.assign(keeperLook, { top: parseInt(meta.color.slice(1), 16), topPrint: [0, 0, 0, 0], carry: "none" });
    shop.keeper = addPerson(crowd, keeperLook);
    Object.assign(shop.keeper, { x: x0 - 0.3, z: KEEPER_Z, y: SLAB, yaw: 0, mode: "idle" });
    return shop;
  });
  return { shops, counters, hitBoxes: shops.map((shop) => shop.hitBox), world, crowd, customerCap: MAX_CUSTOMERS_PER_SHOP };
}

function itemCount(key, stock, cap) {
  /** Return how many item models stand for a stock level: one per rice sack, scaled for packs and bars. */
  const config = GOODS[key];
  if (key === "rice_bag") return Math.min(config.slots, Math.round(stock));
  const count = Math.round((stock / cap) * config.slots);
  return Math.min(config.slots, stock > 0 ? Math.max(1, count) : 0);
}

function layoutGoods(goods, before, after, removeProgress, addProgress) {
  /** Place a shelf's items: sold items drop and shrink, newly arrived ones grow in, the rest stay hidden. */
  const total = Math.max(before, after);
  for (let index = 0; index < goods.config.slots; index += 1) {
    const slot = goods.slots[index];
    let scale = 1;
    let drop = 0;
    if (index >= total) scale = 0;
    else if (index >= after) {
      scale = 1 - removeProgress;
      drop = removeProgress * 0.4;
    } else if (index >= before) {
      scale = addProgress;
    }
    holder.position.set(slot.x, scale > 0 ? slot.y - drop : -50, slot.z);
    holder.scale.setScalar(Math.max(scale, 0.0001));
    holder.rotation.set(0, ((index % 3) - 1) * 0.12, 0);
    holder.updateMatrix();
    goods.mesh.setMatrixAt(goods.base + index, holder.matrix);
  }
  goods.mesh.instanceMatrix.needsUpdate = true;
}

function setInstance(mesh, index, x, y, z, scaleY = 1, scaleXZ = 1) {
  /** Place one instance of an instanced mesh with a vertical and horizontal scale. */
  holder.position.set(x, y, z);
  holder.rotation.set(0, 0, 0);
  holder.scale.set(scaleXZ, scaleY, scaleXZ);
  holder.updateMatrix();
  mesh.setMatrixAt(index, holder.matrix);
  mesh.instanceMatrix.needsUpdate = true;
}

function applyCounterProps(view, shop, state, meta) {
  /** Size the loose rice level, the cash notes and the debt book of a shop to its state. */
  const { counters } = view;
  const riceShare = Math.min(state.stock.loose_rice / meta.caps.loose_rice, 1);
  const riceHeight = 0.02 + riceShare * 0.28;
  setInstance(counters.rice, shop.index, shop.x0 - 1.35, COUNTER_TOP + 0.02 + riceHeight / 2, COUNTER_Z, riceHeight);
  const cashHeight = 0.02 + Math.min(Math.max(state.cash, 0) / (meta.startCash * 5), 1) * 0.5;
  setInstance(counters.notes, shop.index, shop.x0 + 1.3, COUNTER_TOP + 0.3 + cashHeight / 2, COUNTER_Z, cashHeight);
  const debt = state.recent_debt + state.old_debt;
  const thickness = 0.03 + Math.min(debt / 150000, 1) * 0.45;
  setInstance(counters.books, shop.index, shop.x0 + 0.35, COUNTER_TOP + thickness / 2, COUNTER_Z + 0.05, thickness);
}

function applySigns(shop, state) {
  /** Show the price band and the credit policy on the post plaques, using cached textures. */
  const [background, color] = BAND_STYLES[state.price_band];
  shop.signs.band.material.map = signTexture(`Price: ${state.price_band.toUpperCase()}`,
    { background, color, width: 512, height: 170, font: "800 66px system-ui, sans-serif" });
  const open = state.credit_policy === "open";
  shop.signs.credit.material.map = signTexture(open ? "Open credit" : "Cash only",
    { background: open ? "#9cc9e6" : "#f4ead5", width: 512, height: 170, font: "800 66px system-ui, sans-serif" });
  shop.signs.band.material.needsUpdate = true;
  shop.signs.credit.material.needsUpdate = true;
}

export function settleShop(view, shop, shopFrame, meta) {
  /** Show a shop's state at rest: signs, counter props, shelves, low-stock warnings and the chalkboard. */
  const state = shopFrame.state;
  shop.state = state;
  applySigns(shop, state);
  applyCounterProps(view, shop, state, meta);
  for (const [key, goods] of Object.entries(shop.goods)) {
    const count = itemCount(key, state.stock[key], meta.caps[key]);
    layoutGoods(goods, count, count, 1, 1);
    goods.shown = count;
    shop.lowStock[key] = state.stock[key] / meta.caps[key] < LOW_STOCK_SHARE;
    shop.units[key].material = shop.lowStock[key] ? view.world.materials.lowStock : view.world.materials.shelfWood;
  }
}

function planCustomers(shop, frame, cap) {
  /** Decide how many of today's real customers to show, which leave unhappy, and the path each one walks. */
  if (!frame.customers || frame.closed) return [];
  const shown = Math.min(cap, Math.max(1, Math.round(frame.customers / CUSTOMERS_PER_FIGURE)));
  const walked = frame.haggles.filter((haggle) => haggle.outcome === "walked_away").length;
  const failed = frame.turned_away + walked;
  let unhappy = Math.round((shown * failed) / frame.customers);
  if (failed > 0) unhappy = Math.max(1, unhappy);
  unhappy = Math.min(unhappy, shown);
  return Array.from({ length: shown }, (_, i) => {
    const side = i % 2 ? 1 : -1;
    const lane = 1.0 + (i % 3) * 0.45;
    return {
      unhappy: i >= shown - unhappy,
      haggler: i === 0 && frame.haggles.length > 0,
      start: (i / shown) * 0.36,
      from: new THREE.Vector3(shop.x0 + side * 3.6, LANE_Y, lane),
      spot: new THREE.Vector3(shop.x0 - 1.5 + (i % 4) * 1.0, LANE_Y, -1.72 + Math.floor(i / 4) * 0.5),
      to: new THREE.Vector3(shop.x0 - side * 3.8, LANE_Y, lane + 0.25),
    };
  });
}

function pidgin(kind, day, index) {
  /** Pick a Pidgin line of a kind, the same one every time for this day and shop. */
  const lines = PIDGIN[kind];
  return lines[(day * 7 + index * 3) % lines.length];
}

export function startShopDay(view, shop, shopFrame, meta) {
  /** Begin one shop's day: update its signs, props and chalkboard, and plan its shoppers and bubbles. */
  const before = {};
  const after = {};
  for (const [key, goods] of Object.entries(shop.goods)) {
    before[key] = goods.shown;
    after[key] = itemCount(key, shopFrame.state.stock[key], meta.caps[key]);
  }
  shop.frame = shopFrame;
  shop.day = { before, after, plan: planCustomers(shop, shopFrame, view.customerCap), haggle: shopFrame.haggles[0] || null };
  applySigns(shop, shopFrame.state);
  applyCounterProps(view, shop, shopFrame.state, meta);
  drawChalkboard(shop.signs.chalkboard, shop.baseName, shopFrame, shopFrame.score_label);
}

function stepAlong(plan, progress) {
  /** Return where a shopper is at this point of the day and what they are doing, or null when off stage. */
  const local = (progress - plan.start) / 0.62;
  if (local <= 0 || local >= 1) return null;
  const position = new THREE.Vector3();
  let phase = "arrive";
  let target = plan.spot;
  if (local < 0.36) position.lerpVectors(plan.from, plan.spot, THREE.MathUtils.smoothstep(local / 0.36, 0, 1));
  else if (local < 0.62) {
    position.copy(plan.spot);
    phase = "counter";
  } else {
    position.lerpVectors(plan.spot, plan.to, THREE.MathUtils.smoothstep((local - 0.62) / 0.38, 0, 1));
    phase = "leave";
    target = plan.to;
  }
  return { position, local, phase, target };
}

function animateShoppers(view, shop, progress) {
  /** Walk a shop's real shoppers in from the lane, let them buy or haggle at the counter, then send them on. */
  const accent = parseInt(shop.color.slice(1), 16);
  let atCounter = 0;
  shop.shoppers.forEach((person, i) => {
    const plan = shop.day && shop.day.plan[i];
    const step = plan && stepAlong(plan, progress);
    if (!step) {
      person.active = false;
      return;
    }
    person.active = true;
    person.x = step.position.x;
    person.z = step.position.z;
    person.y = LANE_Y;
    const served = step.local >= 0.5;
    const unhappy = served && plan.unhappy;
    person.ring = unhappy ? RED : accent;
    person.bagColor = served && !plan.unhappy ? accent : null;
    if (step.phase === "counter") {
      atCounter += 1;
      person.yaw = Math.PI;
      person.mode = plan.haggler ? "haggle" : unhappy ? "upset" : "talk";
    } else {
      const dx = step.target.x - person.x;
      const dz = step.target.z - person.z;
      if (Math.hypot(dx, dz) > 0.05) person.yaw = Math.atan2(dx, dz);
      person.mode = unhappy ? "upset" : "idle";
    }
  });
  return atCounter;
}

function animateGoods(shop, progress) {
  /** Drop sold goods during the buying part of the day and grow in deliveries at its start. */
  const remove = THREE.MathUtils.clamp((progress - 0.3) / 0.4, 0, 1);
  const add = THREE.MathUtils.clamp(progress / 0.2, 0, 1);
  for (const [key, goods] of Object.entries(shop.goods)) layoutGoods(goods, shop.day.before[key], shop.day.after[key], remove, add);
}

function animateBubble(shop, progress) {
  /** Show the day's real haggle in Pidgin, or a jara request on measured rice sales with an extra. */
  const frame = shop.frame;
  const haggle = shop.day.haggle;
  const jara = frame.jara_count > 0;
  const bubble = shop.bubble;
  let text = "";
  let tone = "";
  if (haggle && progress > 0.2 && progress < 0.75) {
    const decided = progress > 0.42;
    tone = decided ? (haggle.outcome === "sold" ? "sold" : "walked") : "";
    if (!decided) text = pidgin("offer", frame.day, shop.index);
    else if (haggle.outcome === "sold") text = `${pidgin("sold", frame.day, shop.index)}  ₦${Math.round(haggle.price).toLocaleString("en-US")}`;
    else text = pidgin("walked", frame.day, shop.index);
  } else if (!haggle && jara && progress > 0.25 && progress < 0.6) {
    text = "Add jara for me!";
  }
  bubble.style.opacity = text ? 1 : 0;
  bubble.dataset.on = text ? "1" : "";
  if (text && bubble.textContent !== text) bubble.textContent = text;
  bubble.classList.toggle("sold", tone === "sold");
  bubble.classList.toggle("walked", tone === "walked");
  const floating = jara && progress > 0.36 && progress < 0.8;
  shop.float.style.opacity = floating ? 1 - Math.max(0, (progress - 0.66) / 0.14) : 0;
  shop.float.dataset.on = floating ? "1" : "";
  shop.floatRise = floating ? (progress - 0.36) * 1.4 : 0;
}

export function animateShop(view, shop, progress) {
  /** Advance one shop's day animation. */
  if (!shop.day) return;
  shop.atCounter = animateShoppers(view, shop, progress);
  animateGoods(shop, progress);
  animateBubble(shop, progress);
}

export function finishShopDay(view, shop, meta) {
  /** Settle a shop at the end of its day and clear its shoppers and bubbles. */
  if (!shop.frame) return;
  settleShop(view, shop, shop.frame, meta);
  for (const person of shop.shoppers) person.active = false;
  shop.atCounter = 0;
  shop.bubble.style.opacity = 0;
  shop.bubble.dataset.on = "";
  shop.float.style.opacity = 0;
  shop.float.dataset.on = "";
  shop.day = null;
}

export function animateKeepers(view, time, progress) {
  /** Keep each shopkeeper behind the counter: serving when shoppers wait, restocking shelves after a delivery. */
  for (const shop of view.shops) {
    const keeper = shop.keeper;
    const action = shop.frame ? shop.frame.action_label || "" : "";
    const restocking = shop.day && /restock|open rice/i.test(action) && progress < 0.3;
    if (shop.frame && shop.frame.closed) {
      keeper.active = false;
      continue;
    }
    keeper.active = true;
    if (restocking) {
      keeper.x = shop.x0 + Math.sin(time * 0.8) * 1.2;
      keeper.z = SHELF_Z + 0.75;
      keeper.yaw = Math.PI;
      keeper.mode = "arrange";
    } else {
      keeper.x = shop.x0 - 0.3 + Math.sin(time * 0.5 + shop.index) * 0.6;
      keeper.z = KEEPER_Z;
      keeper.yaw = Math.sin(time * 0.3 + shop.index) * 0.25;
      keeper.mode = shop.atCounter ? "talk" : "idle";
    }
  }
}

export function pulseLowStock(world, time) {
  /** Pulse the shared low-stock shelf material red. */
  world.materials.lowStock.emissiveIntensity = 0.25 + 0.55 * (0.5 + 0.5 * Math.sin(time * 4));
}

