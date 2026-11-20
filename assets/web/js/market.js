import * as THREE from "three";
import { mergeGeometries } from "three/addons/BufferGeometryUtils.js";
import { LAYOUT, addCollider, seeded } from "./layout.js";
import { StaticBatch, placement } from "./world.js";
import { Props, StallFrame, buildStall, counterFront, defineProps, fillStall, neighbourSign } from "./stalls.js";
import { bannerTexture, glowTexture, signTexture } from "./textures.js";

const NORTH_TRADES = { "-24": "plastics", "-18": "oil", 18: "bags" };
const SOUTH_FRONT_TRADES = ["produce", "household", "grains", "shoes", "plastics", "fruit", "grains", "household", "oil", "produce", "fruit"];
const SOUTH_BACK_TRADES = ["clothes", "bags", "shoes", "clothes", "shoes", "bags", "clothes", "shoes", "clothes", "bags", "clothes"];
const SOUTH_ROW_TRADES = ["kitenge", "clothes", "bags", "clothes", "shoes", "kitenge", "clothes", "bags", "clothes", "kitenge", "shoes", "clothes"];
const WALL_TINTS = ["corrTeal", "corrBlue", "corrTeal", "corrGreen", "corrTeal", "corrBlue"];
const SIGN_COLORS = [["#f4ead5", "#1d3b5a"], ["#1f6fb5", "#fffaf0"], ["#f2c230", "#2b1d14"], ["#2a8a4a", "#fffaf0"], ["#c8352e", "#fffaf0"]];
const HALL_BLOCKS = [[-17.6, -15.6, "wall"], [-13, -8, "potatoes"], [-5.2, -1.6, "mixed"], [3.1, 6.7, "roots"], [9.5, 13.2, "fruit"], [15.8, 19.6, "grains"]];
const HALL_COLUMN_X = [22, 30, 38, 46, 54, 62];
const HALL_COLUMN_Z = [-18, -10.5, -3.4, 4.9, 11.35, 17.7, 22];
const ACROSS_COLORS = ["plasterCream", "plasterYellow", "plasterBlue", "plasterPink", "plasterWhite", "plasterGreen"];
const ACROSS_SIGNS = ["Alimentation", "Salon de coiffure", "Boutique", "Restaurant", "Quincaillerie", "Pharmacie", "Mobile Money", "Atelier", "Librairie", "Hotel"];

const counterMaterials = new Map();

function quad(world, batch, size, position, material, rotation = null) {
  /** Queue a flat plane of a size at a position. */
  batch.add(new THREE.PlaneGeometry(...size), position, material, rotation);
}

function buildGround(world, batch) {
  /** Lay the red earth, the sandy lanes, the concrete hall floor, the curb, the road and the far pavement. */
  const m = world.materials;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(900, 900).rotateX(-Math.PI / 2), m.dirtRed);
  const uv = ground.geometry.attributes.uv;
  for (let i = 0; i < uv.count; i += 1) uv.setXY(i, uv.getX(i) * 180, uv.getY(i) * 180);
  ground.receiveShadow = true;
  world.scene.add(ground);
  batch.box([52.6, 0.04, 5.6], [-4, 0.02, 0.7], m.earthLane);
  batch.box([52.6, 0.04, 4.2], [-4, 0.02, LAYOUT.clothes.center], m.earthLane);
  for (const x of LAYOUT.passages) batch.box([2.4, 0.04, 9], [x, 0.02, 8], m.earthLane);
  batch.box([40, 0.1, 40], [42, 0.05, 2], m.concrete);
  batch.box([3.6, 0.08, 46], [-31.8, 0.04, 3], m.concrete);
  batch.box([0.5, 0.22, 46], [-33.85, 0.11, 3], m.concreteDark);
  batch.box([0.6, 0.06, 46], [-34.4, 0.03, 3], m.steelDark);
  batch.box([0.6, 0.24, 300], [-48.2, 0.12, 0], m.concreteDark);
  batch.box([LAYOUT.road.width, 0.05, 300], [LAYOUT.road.x, 0.025, 0], m.asphalt);
  for (let z = -148; z < 150; z += 6) batch.box([0.15, 0.06, 2.6], [LAYOUT.road.x, 0.04, z], m.cream);
  batch.box([0.6, 0.24, 300], [-56.8, 0.12, 0], m.concreteDark);
  batch.box([5.5, 0.18, 300], [-59.8, 0.09, 0], m.concrete);
}

function buildFacade(world, batch, props) {
  /** Build the long yellow outer building: grille front, yellow awning on grey pillars, and a row of gables above. */
  const m = world.materials;
  const x = LAYOUT.facadeX;
  const segments = [[-18, LAYOUT.lane.zMin], [LAYOUT.lane.zMax, LAYOUT.clothes.zMin], [LAYOUT.clothes.zMax, 26]];
  for (const [z0, z1] of segments) {
    const length = z1 - z0;
    const middle = (z0 + z1) / 2;
    batch.box([0.3, 0.35, length], [x, 0.18, middle], m.plasterYellow);
    batch.box([0.1, 0.1, length], [x, 2.85, middle], m.steelDark);
    for (let z = z0 + 0.1; z < z1; z += 0.16) batch.box([0.03, 2.5, 0.03], [x, 1.6, z], m.steelDark);
    addCollider(x - 0.2, z0, x + 0.2, z1);
  }
  batch.box([0.25, 3.0, 46], [x, 4.5, 4], m.corrYellow);
  const slope = Math.atan2(0.5, 3.6);
  batch.add(new THREE.BoxGeometry(3.9, 0.05, 46), [x - 1.8, 3.15, 4], m.corrYellow, [0, 0, slope]);
  batch.box([0.08, 0.25, 46], [x - 3.65, 2.86, 4], m.corrYellow);
  for (let z = -18; z <= 26; z += 4.4) {
    batch.box([0.22, 2.9, 0.22], [x - 3.45, 1.45, z], m.steel);
    addCollider(x - 3.6, z - 0.15, x - 3.3, z + 0.15);
  }
  for (let z = -18; z < 26; z += 7.33) {
    const shape = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(7.33, 0), new THREE.Vector2(3.665, 2.4)]);
    const gable = new THREE.ExtrudeGeometry(shape, { depth: 0.2, bevelEnabled: false }).rotateY(Math.PI / 2);
    batch.add(gable, [x - 0.1, 6.0, z + 7.33], m.corrYellow);
    const roofSlope = Math.atan2(2.4, 3.665);
    const length = Math.hypot(2.4, 3.665) + 0.15;
    for (const side of [-1, 1]) {
      batch.add(new THREE.BoxGeometry(3.4, 0.05, length), [x + 1.6, 7.2, z + 3.665 + side * 1.83], m.corrYellow, [side * roofSlope, 0, 0]);
    }
    batch.box([0.4, 6, 0.4], [x - 0.05, 3, z], m.corrYellow);
  }
  const sign = new THREE.MeshLambertMaterial({ map: signTexture("KIMIRONKO MARKET", { background: "#1d5b3a", color: "#fff3c4",
    border: "#f2c14e", width: 1024, height: 170, font: "900 120px Georgia, serif" }) });
  quad(world, batch, [6.4, 1.06], [x - 0.15, 4.6, LAYOUT.lane.center], sign, [0, -Math.PI / 2, 0]);
  quad(world, batch, [6.4, 1.06], [x + 0.15, 4.6, LAYOUT.lane.center], sign, [0, Math.PI / 2, 0]);
  const clothesSign = new THREE.MeshLambertMaterial({ map: signTexture("Imyenda / Clothes & Bags", { background: "#f4ead5", color: "#8a2a1a",
    width: 1024, height: 170, font: "800 100px Georgia, serif" }) });
  quad(world, batch, [5.2, 0.86], [x - 0.15, 4.6, LAYOUT.clothes.center], clothesSign, [0, -Math.PI / 2, 0]);
  quad(world, batch, [5.2, 0.86], [x + 0.15, 4.6, LAYOUT.clothes.center], clothesSign, [0, Math.PI / 2, 0]);
  const banner = new THREE.MeshLambertMaterial({ map: bannerTexture("Murakaza neza"), side: THREE.DoubleSide });
  quad(world, batch, [4.2, 0.8], [x - 3.62, 2.35, LAYOUT.lane.center], banner, [0, -Math.PI / 2, 0]);
  props.add("cord", placement([x - 3.62, 2.85, LAYOUT.lane.center - 2.1], null, [1, 0.12, 1]));
}

function buildNorthTerrace(world, batch, props, vendors) {
  /** Build the neighbour stalls on our side of the lane and the teal warehouse behind the whole row. */
  const m = world.materials;
  const { shop, northRoof } = LAYOUT;
  const stalls = [];
  for (const [x, trade] of Object.entries(NORTH_TRADES)) {
    const frame = new StallFrame(Number(x), shop.front, 1);
    const stall = buildStall(world, batch, props, frame, { width: shop.width, depth: shop.front - shop.back, front: northRoof.low,
      back: northRoof.high, overhang: 1.1, wall: "corrTeal", sign: neighbourSign(trade, Number(x) > 0 ? 1 : 0),
      signColor: SIGN_COLORS[Math.abs(Number(x)) % 5][0], signInk: SIGN_COLORS[Math.abs(Number(x)) % 5][1] });
    vendors.push(fillStall(props, stall, trade, Number(x) * 3 + 11, world));
    batch.addObject(counterMesh(world, frame, shop.width, "#2a6fb5"));
    stalls.push(stall);
  }
  const z0 = -8.35;
  const z1 = -16.5;
  batch.box([52, 6.4, 0.2], [-4, 3.2, z1], m.corrTeal);
  batch.box([52, 1.2, 0.2], [-4, 5.8, z0], m.corrTeal);
  for (const x of [-30, 22]) batch.box([0.2, 6.4, z0 - z1], [x, 3.2, (z0 + z1) / 2], m.corrTeal);
  const half = (z0 - z1) / 2;
  const slope = Math.atan2(2.4, half);
  const length = Math.hypot(half, 2.4) + 0.3;
  for (const side of [-1, 1]) {
    batch.add(new THREE.BoxGeometry(52.6, 0.06, length), [-4, 6.4 + 1.2, (z0 + z1) / 2 + side * half / 2], m.corrGreen, [side * slope, 0, 0]);
  }
  for (const x of [-30.05, 22.05]) {
    const shape = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(z0 - z1, 0), new THREE.Vector2(half, 2.4)]);
    batch.add(new THREE.ShapeGeometry(shape), [x, 6.4, z0], m.corrTeal, [0, Math.PI / 2, 0]);
  }
  addCollider(-30, z1, 22, z0);
  return stalls;
}

function counterMesh(world, frame, width, hex) {
  /** Return a printed counter front whose material is shared by every counter of that colour. */
  const mesh = counterFront(world, frame, width, hex);
  if (!counterMaterials.has(hex)) counterMaterials.set(hex, mesh.material);
  mesh.material = counterMaterials.get(hex);
  mesh.matrixWorld.copy(mesh.matrix);
  const group = new THREE.Group();
  group.add(mesh);
  mesh.matrixAutoUpdate = false;
  return group;
}

function buildSouthSide(world, batch, props, vendors) {
  /** Build the back-to-back stalls across the lane, the clothes and bags lane, and the blue sheds behind it. */
  const m = world.materials;
  const rows = [
    [LAYOUT.southFront, -1, SOUTH_FRONT_TRADES, 3.5, 4.2, 1.0],
    [LAYOUT.southBack, 1, SOUTH_BACK_TRADES, 3.4, 4.1, 1.0],
    [LAYOUT.southRow, -1, SOUTH_ROW_TRADES, 3.4, 4.3, 1.1],
  ];
  rows.forEach(([row, facing, trades, low, high, overhang], rowIndex) => {
    row.xs.forEach((x, index) => {
      const trade = trades[index % trades.length];
      const depth = Math.abs(row.back - row.front);
      const frame = new StallFrame(x, row.front, facing);
      const signed = (index + rowIndex) % 2 === 0;
      const colors = SIGN_COLORS[(index + rowIndex * 2) % SIGN_COLORS.length];
      const stall = buildStall(world, batch, props, frame, { width: 3.9, depth, front: low, back: high, overhang,
        wall: WALL_TINTS[(index + rowIndex) % WALL_TINTS.length], roof: (index * 7 + rowIndex) % 5 === 0 ? "corrRust" : "corrRoof",
        sign: signed ? neighbourSign(trade, index) : null, signColor: colors[0], signInk: colors[1],
        counter: !["plastics", "bags"].includes(trade) });
      vendors.push(fillStall(props, stall, trade, index * 13 + rowIndex * 101 + 5, world));
      if (!["plastics", "bags"].includes(trade)) batch.addObject(counterMesh(world, frame, 3.9, ["#c8352e", "#2a6fb5", "#2a8a4a"][(index + rowIndex) % 3]));
    });
  });
  const z0 = LAYOUT.southRow.back + 0.1;
  const z1 = 30;
  for (const [x0, x1, wall, roof] of [[-30, -6, "corrBlue", "corrRoof"], [-6, 22, "corrTeal", "corrGreen"]]) {
    const width = x1 - x0;
    const middle = (x0 + x1) / 2;
    batch.box([width, 6.2, 0.2], [middle, 3.1, z0], m[wall]);
    batch.box([width, 6.2, 0.2], [middle, 3.1, z1], m[wall]);
    const half = (z1 - z0) / 2;
    const slope = Math.atan2(2.6, half);
    const length = Math.hypot(half, 2.6) + 0.3;
    for (const side of [-1, 1]) {
      batch.add(new THREE.BoxGeometry(width + 0.4, 0.06, length), [middle, 6.2 + 1.3, (z0 + z1) / 2 + side * half / 2], m[roof], [side * slope, 0, 0]);
    }
    for (const x of [x0, x1]) {
      batch.box([0.2, 6.2, z1 - z0], [x, 3.1, (z0 + z1) / 2], m[wall]);
      const shape = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(z1 - z0, 0), new THREE.Vector2(half, 2.6)]);
      batch.add(new THREE.ShapeGeometry(shape), [x, 6.2, z0], m[wall], [0, -Math.PI / 2, 0]);
    }
  }
  addCollider(-30, z0, 22, z1);
}

function buildHall(world, batch, props, vendors) {
  /** Build the big covered produce hall: steel lattice trusses, skylights, fans, and long tables of produce. */
  const m = world.materials;
  const { xMin, xMax, zMin, zMax, eave, ridge, ridgeZ } = LAYOUT.hall;
  const chord = 6.6;
  const roofY = (z) => (z < ridgeZ ? eave + ((z - zMin) / (ridgeZ - zMin)) * (ridge - eave) : eave + ((zMax - z) / (zMax - ridgeZ)) * (ridge - eave));
  for (const x of HALL_COLUMN_X) {
    for (const z of HALL_COLUMN_Z) {
      batch.box([0.32, chord, 0.32], [x, chord / 2, z], m.steel);
      batch.box([0.6, 0.3, 0.6], [x, 0.15, z], m.concreteDark);
      addCollider(x - 0.25, z - 0.25, x + 0.25, z + 0.25);
    }
    batch.box([0.16, 0.16, zMax - zMin], [x, chord, (zMin + zMax) / 2], m.steelDark);
    for (const [z0, z1] of [[zMin, ridgeZ], [ridgeZ, zMax]]) {
      const length = Math.hypot(z1 - z0, ridge - eave);
      const slope = Math.atan2(roofY(z1) - roofY(z0), z1 - z0);
      batch.add(new THREE.BoxGeometry(0.16, 0.18, length), [x, (roofY(z0) + roofY(z1)) / 2 - 0.15, (z0 + z1) / 2], m.steelDark, [-slope, 0, 0]);
    }
    for (let z = zMin; z < zMax - 0.1; z += 2) {
      const top = roofY(z) - 0.2;
      batch.box([0.08, top - chord, 0.08], [x, (top + chord) / 2, z], m.steelDark);
      const nextTop = roofY(z + 2) - 0.2;
      const rise = nextTop - chord;
      const diagonal = Math.hypot(2, rise);
      batch.add(new THREE.BoxGeometry(0.06, 0.06, diagonal), [x, (chord + nextTop) / 2, z + 1], m.steelDark, [-Math.atan2(rise, 2), 0, 0]);
    }
  }
  for (let z = zMin; z <= zMax; z += 2.5) batch.box([xMax - xMin + 0.6, 0.12, 0.12], [(xMin + xMax) / 2, roofY(z) - 0.12, z], m.steelDark);
  const roofSheets = new StaticBatch(world);
  const skylights = [[-13, -10.5], [-4.5, -2], [5.5, 8], [14.5, 17]];
  const sheetZ = [];
  let z = zMin - 0.6;
  for (const [s0, s1] of skylights) {
    sheetZ.push([z, s0]);
    z = s1;
  }
  sheetZ.push([z, zMax + 0.6]);
  const addSlab = (z0, z1, material, target) => {
    const parts = z0 < ridgeZ && z1 > ridgeZ ? [[z0, ridgeZ], [ridgeZ, z1]] : [[z0, z1]];
    for (const [a, b] of parts) {
      const y0 = roofY(Math.max(Math.min(a, zMax), zMin)) + (a < zMin ? -(zMin - a) * 0.17 : a > zMax ? -(a - zMax) * 0.17 : 0);
      const y1 = roofY(Math.max(Math.min(b, zMax), zMin)) + (b > zMax ? -(b - zMax) * 0.17 : b < zMin ? -(zMin - b) * 0.17 : 0);
      const length = Math.hypot(b - a, y1 - y0);
      target.add(new THREE.BoxGeometry(xMax - xMin + 1.2, 0.05, length), [(xMin + xMax) / 2, (y0 + y1) / 2, (a + b) / 2], material, [-Math.atan2(y1 - y0, b - a), 0, 0]);
    }
  };
  for (const [z0, z1] of sheetZ) addSlab(z0, z1, m.corrRoof, roofSheets);
  for (const [s0, s1] of skylights) addSlab(s0, s1, m.skylight, roofSheets);
  const roofMeshes = roofSheets.build(world.scene);
  for (const [zEdge] of [[zMin], [zMax]]) batch.box([xMax - xMin + 1.2, 0.45, 0.08], [(xMin + xMax) / 2, eave - 0.05, zEdge + (zEdge < 0 ? -0.6 : 0.6)], m.fasciaBlue);
  for (const [x0, x1, zWall] of [[xMin, xMax, zMin], [xMin, xMax, zMax]]) {
    for (let x = x0; x < x1; x += 8) {
      if (Math.abs(x + 4 - 42) < 1) continue;
      batch.box([7.6, 2.3, 0.1], [x + 4, 1.15, zWall], m.corrBlue);
      addCollider(x, zWall - 0.1, x + 8, zWall + 0.1);
    }
  }
  for (const zc of [-12, 10]) {
    batch.box([0.1, 2.3, 10], [xMax, 1.15, zc], m.corrBlue);
    addCollider(xMax - 0.1, zc - 5, xMax + 0.1, zc + 5);
  }
  buildHallTables(world, batch, props, vendors);
  const fans = buildFans(world);
  const shafts = buildShafts(world, skylights);
  const lamps = [];
  for (const aisle of LAYOUT.hallAisles) for (let x = 26; x < 60; x += 7) lamps.push(new THREE.Vector3(x, 5.2, aisle));
  for (const lamp of lamps) {
    props.add("cord", placement([lamp.x, chord, lamp.z], null, [1, chord - 5.3, 1]));
    props.add("bulb", placement([lamp.x, lamp.y, lamp.z]));
  }
  return { roofMeshes, fans, shafts, lamps };
}

function buildHallTables(world, batch, props, vendors) {
  /** Fill each table block of the hall with produce like the photos: mountains of potatoes, tomatoes, fruit and grains. */
  const m = world.materials;
  const segments = [[23.2, 34.3], [36.7, 47.3], [49.7, 60.8]];
  HALL_BLOCKS.forEach(([z0, z1, kind], blockIndex) => {
    for (const [x0, x1] of segments) {
      const depth = z1 - z0;
      const zc = (z0 + z1) / 2;
      addCollider(x0, z0, x1, z1);
      if (kind === "wall") {
        batch.box([x1 - x0, 0.7, depth], [(x0 + x1) / 2, 0.35, zc], m.planksDark);
        for (let x = x0 + 0.5; x < x1; x += 0.95) {
          props.add(`fabric${Math.floor(seeded(x, 3) * 6)}`, placement([x, 3.4, z0 + 0.2]));
          props.add("crate", placement([x, 0.7, zc]));
          props.add(seeded(x, 4) < 0.5 ? "oranges" : "avocados", placement([x, 0.98, zc], null, [0.6, 0.6, 0.6]));
        }
        batch.box([x1 - x0, 0.06, 0.06], [(x0 + x1) / 2, 3.42, z0 + 0.2], m.steelDark);
        vendors.push({ position: new THREE.Vector3((x0 + x1) / 2, 0.1, z1 + 0.6), yaw: 0, mode: "talk" });
        continue;
      }
      batch.box([x1 - x0, 0.08, depth], [(x0 + x1) / 2, 0.8, zc], m.planks);
      batch.box([x1 - x0, 0.75, depth - 0.3], [(x0 + x1) / 2, 0.4, zc], m.planksDark);
      batch.box([x1 - x0 + 0.05, 0.02, depth + 0.05], [(x0 + x1) / 2, 0.85, zc], m.sackCloth);
      const rows = depth > 3.6 ? [z0 + 1.0, z1 - 1.0] : [zc];
      for (const rz of rows) {
        for (let x = x0 + 0.9; x < x1 - 0.5; x += kind === "potatoes" ? 1.8 : 1.15) {
          const r = seeded(x * 7 + rz, blockIndex);
          if (kind === "potatoes") props.add(r < 0.8 ? "potatoes" : "sweetPotatoes", placement([x, 0.86, rz], [0, r * 6, 0], [1, 1 + r * 0.5, 1]));
          else if (kind === "mixed") props.add(["tomatoes", "onions", "peppers", "carrots", "beets"][Math.floor(r * 5)], placement([x, 0.86, rz], [0, r * 6, 0], [1.3, 1.3, 1.3]));
          else if (kind === "roots") props.add(["sweetPotatoes", "cassava", "avocados", "potatoes"][Math.floor(r * 4)], placement([x, 0.86, rz], [0, r * 6, 0], [0.8, 0.8, 0.8]));
          else if (kind === "fruit") {
            if (r < 0.35) for (let k = 0; k < 6; k += 1) props.add("pineapple", placement([x - 0.3 + (k % 3) * 0.25, 0.86, rz - 0.15 + Math.floor(k / 3) * 0.3]));
            else if (r < 0.65) for (let k = 0; k < 5; k += 1) props.add("watermelon", placement([x - 0.4 + k * 0.22, 0.86 + (k % 2) * 0.05, rz + (k % 2) * 0.2], [0, k, 0]), 0x2f6a2a);
            else if (r < 0.85) props.add("bananas", placement([x, 0.86, rz], [0, r * 6, 0], [1.2, 1.2, 1.2]));
            else props.add("oranges", placement([x, 0.86, rz]));
          } else if (kind === "grains") {
            props.add("tub", placement([x - 0.25, 0.86, rz]));
            props.add("tub", placement([x + 0.25, 0.86, rz]));
            props.add("sackBody", placement([x, 0.1, rz + depth / 2 + 0.35]));
            props.add("grainTop", placement([x, 0.1, rz + depth / 2 + 0.35]), [0x9a2a2a, 0xe8c23a, 0x8a5a3a, 0xc9a46a][Math.floor(r * 4)]);
          }
        }
      }
      for (let x = x0 + 1.4; x < x1 - 0.8; x += 3.4) {
        const side = seeded(x, blockIndex + 20) < 0.5 ? -1 : 1;
        const zEdge = side < 0 ? z0 - 0.45 : z1 + 0.45;
        const r = seeded(x, blockIndex + 30);
        props.add("basket", placement([x + 0.8, 0.1, zEdge], null, [0.8, 0.8, 0.8]));
        if (kind === "potatoes" && r < 0.5) {
          props.add("sackBody", placement([x - 0.7, 0.1, zEdge]));
          props.add("grainTop", placement([x - 0.7, 0.1, zEdge]), 0xc98a4b);
        }
        const mode = r < 0.45 ? "sit" : r < 0.75 ? "talk" : "arrange";
        if (mode === "sit") props.add("stool", placement([x, 0.1, zEdge]));
        vendors.push({ position: new THREE.Vector3(x, 0.1, zEdge), yaw: side < 0 ? Math.PI : 0, mode, seat: 0.43 });
      }
    }
  });
}

function buildFans(world) {
  /** Hang slow ceiling fans from the hall trusses, as one instanced mesh of blades turned every frame. */
  const spots = [];
  for (const x of [26, 34, 42, 50, 58]) for (const z of [-6.6, 8.1]) spots.push(new THREE.Vector3(x, 6.0, z));
  const blades = new THREE.InstancedMesh(new THREE.BoxGeometry(1.1, 0.02, 0.14).translate(0.55, 0, 0), world.materials.steelDark, spots.length * 3);
  const hubs = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.1, 0.1, 0.12, 8), world.materials.steelDark, spots.length);
  const rods = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.02, 0.02, 0.6, 4), world.materials.steelDark, spots.length);
  spots.forEach((spot, index) => {
    hubs.setMatrixAt(index, placement([spot.x, spot.y, spot.z]));
    rods.setMatrixAt(index, placement([spot.x, spot.y + 0.32, spot.z]));
  });
  blades.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  world.scene.add(blades, hubs, rods);
  return { blades, spots };
}

function buildShafts(world, skylights) {
  /** Add faint slanted light shafts under the skylights, shown on high quality only. */
  const group = new THREE.Group();
  for (const [s0, s1] of skylights) {
    for (const x of [28, 41, 53]) {
      const shaft = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 9.5), world.materials.shaft);
      shaft.position.set(x + seeded(x, s0) * 3, 4.6, (s0 + s1) / 2 + 1.2);
      shaft.rotation.set(0.25, 0.35, 0.18);
      group.add(shaft);
    }
  }
  group.userData.detail = true;
  world.scene.add(group);
  return group;
}

const paints = new Map();

function paintMaterial(hex) {
  /** Return the shared vehicle paint of a colour, so parked vehicles merge into few draw calls. */
  if (!paints.has(hex)) paints.set(hex, new THREE.MeshLambertMaterial({ color: hex }));
  return paints.get(hex);
}

export function bakeGroup(group) {
  /** Bake a multi-part vehicle into one vertex-coloured mesh, so a moving vehicle costs one draw call. */
  group.updateMatrixWorld(true);
  const parts = [];
  group.traverse((child) => {
    if (!child.isMesh) return;
    const geometry = (child.geometry.index ? child.geometry.toNonIndexed() : child.geometry.clone()).applyMatrix4(child.matrix);
    const tint = child.material.color;
    const colors = new Float32Array(geometry.attributes.position.count * 3);
    for (let i = 0; i < colors.length; i += 3) colors.set([tint.r, tint.g, tint.b], i);
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    for (const name of Object.keys(geometry.attributes)) if (!["position", "normal", "color"].includes(name)) geometry.deleteAttribute(name);
    parts.push(geometry);
  });
  const mesh = new THREE.Mesh(mergeGeometries(parts), new THREE.MeshLambertMaterial({ vertexColors: true }));
  mesh.castShadow = true;
  return mesh;
}

export function motoGroup(world, hex) {
  /** Return a low-poly moto taxi: two wheels, body, seat, tank, handlebar, headlight and exhaust. */
  const m = world.materials;
  const body = paintMaterial(hex);
  const group = new THREE.Group();
  const add = (geometry, material, position, rotation = [0, 0, 0]) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...position);
    mesh.rotation.set(...rotation);
    mesh.castShadow = true;
    group.add(mesh);
  };
  for (const x of [-0.62, 0.66]) {
    add(new THREE.TorusGeometry(0.27, 0.07, 6, 14), m.tyre, [x, 0.33, 0]);
    add(new THREE.CylinderGeometry(0.12, 0.12, 0.08, 8), m.steel, [x, 0.33, 0], [Math.PI / 2, 0, 0]);
  }
  add(new THREE.BoxGeometry(0.9, 0.22, 0.24), body, [0.02, 0.62, 0]);
  add(new THREE.BoxGeometry(0.42, 0.2, 0.3), body, [0.3, 0.8, 0]);
  add(new THREE.BoxGeometry(0.62, 0.1, 0.28), m.black, [-0.24, 0.84, 0]);
  add(new THREE.BoxGeometry(0.06, 0.62, 0.06), m.steel, [0.6, 0.72, 0], [0, 0, -0.38]);
  add(new THREE.BoxGeometry(0.05, 0.05, 0.72), m.black, [0.52, 1.04, 0]);
  add(new THREE.CylinderGeometry(0.07, 0.06, 0.08, 8), m.bulb, [0.66, 0.9, 0], [0, 0, Math.PI / 2]);
  add(new THREE.CylinderGeometry(0.035, 0.04, 0.6, 6), m.steel, [-0.3, 0.42, 0.16], [0, 0, Math.PI / 2 - 0.1]);
  add(new THREE.BoxGeometry(0.4, 0.04, 0.3), m.black, [-0.62, 0.65, 0]);
  return group;
}

export function carGroup(world, hex, kind = "hatch") {
  /** Return a low-poly car: a hatchback, a sedan or a minibus. */
  const m = world.materials;
  const paint = paintMaterial(hex);
  const glass = m.glass;
  const group = new THREE.Group();
  const add = (geometry, material, position, rotation = [0, 0, 0]) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...position);
    mesh.rotation.set(...rotation);
    mesh.castShadow = true;
    group.add(mesh);
  };
  const long = kind === "bus" ? 5.4 : kind === "sedan" ? 4.4 : 3.8;
  const tall = kind === "bus" ? 1.6 : 0.62;
  add(new THREE.BoxGeometry(long, 0.62, 1.72), paint, [0, 0.62, 0]);
  if (kind === "bus") {
    add(new THREE.BoxGeometry(long - 0.4, tall, 1.68), paint, [-0.1, 1.55, 0]);
    for (const side of [-1, 1]) add(new THREE.PlaneGeometry(long - 1.0, 0.6), glass, [-0.2, 1.75, side * 0.85], [0, side > 0 ? 0 : Math.PI, 0]);
    add(new THREE.PlaneGeometry(1.5, 0.7), glass, [long / 2 - 0.19, 1.7, 0], [0, Math.PI / 2, 0]);
  } else {
    const cabin = kind === "sedan" ? 2.1 : 2.2;
    add(new THREE.BoxGeometry(cabin, tall, 1.56), glass, [kind === "sedan" ? -0.2 : -0.45, 1.24, 0]);
    add(new THREE.BoxGeometry(cabin - 0.1, 0.06, 1.6), paint, [kind === "sedan" ? -0.2 : -0.45, 1.57, 0]);
  }
  for (const x of [-long / 2 + 0.75, long / 2 - 0.75]) {
    for (const z of [-0.82, 0.82]) add(new THREE.CylinderGeometry(0.32, 0.32, 0.22, 12), m.tyre, [x, 0.32, z], [Math.PI / 2, 0, 0]);
  }
  for (const z of [-0.6, 0.6]) add(new THREE.BoxGeometry(0.05, 0.12, 0.3), m.bulb, [long / 2 + 0.01, 0.75, z]);
  return group;
}

function umbrellaGeometry(colors) {
  /** Return a big market umbrella canopy with alternating coloured panels. */
  const canopy = new THREE.ConeGeometry(1.5, 0.55, 8, 1, true).toNonIndexed();
  const shade = new Float32Array(canopy.attributes.position.count * 3);
  const color = new THREE.Color();
  for (let i = 0; i < canopy.attributes.position.count; i += 1) {
    color.setHex(colors[Math.floor(i / 6) % colors.length]);
    shade.set([color.r, color.g, color.b], i * 3);
  }
  canopy.setAttribute("color", new THREE.BufferAttribute(shade, 3));
  canopy.translate(0, 2.35, 0);
  const pole = new THREE.CylinderGeometry(0.03, 0.03, 2.3, 5).toNonIndexed().translate(0, 1.15, 0);
  const grey = new Float32Array(pole.attributes.position.count * 3).fill(0.45);
  pole.setAttribute("color", new THREE.BufferAttribute(grey, 3));
  return mergeGeometries([canopy, pole]);
}

function buildYard(world, batch, props, vendors, riders) {
  /** Furnish the entrance yard: moto taxi ranks, umbrella sellers, parked cars, trees and street lamps. */
  const m = world.materials;
  const motoColors = [0xc1121f, 0x1d1d1d, 0x1d3557, 0xc1121f, 0x8a1a1a, 0x2a2a2a];
  const ranks = [[-41.5, 7, 9], [-41.5, -15, 7], [-36.5, 21, 4]];
  let count = 0;
  for (const [x, z0, n] of ranks) {
    for (let i = 0; i < n; i += 1) {
      const moto = motoGroup(world, motoColors[count % motoColors.length]);
      const z = z0 + i * 1.05;
      moto.position.set(x + (i % 2) * 0.3, 0, z);
      moto.rotation.y = Math.PI + 0.35 + (i % 3) * 0.08;
      batch.addObject(moto);
      if (count % 3 === 0) riders.push({ position: new THREE.Vector3(x - 0.15 + (i % 2) * 0.3, 0, z + 0.05), yaw: moto.rotation.y + Math.PI / 2 });
      count += 1;
    }
    addCollider(x - 1.1, z0 - 0.6, x + 1.2, z0 + n * 1.05);
  }
  const umbrellas = new THREE.Mesh(mergeGeometries([
    umbrellaGeometry([0xf6f3ec, 0xf2c230]).translate(-39, 0, -3.5),
    umbrellaGeometry([0xd62828, 0xd62828, 0xf6f3ec]).translate(-44.5, 0, 14.5),
    umbrellaGeometry([0xf6f3ec, 0xf2c230]).translate(-36.5, 0, 24),
    umbrellaGeometry([0x2a6fb5, 0xf6f3ec]).translate(-45, 0, -12),
  ]), new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  umbrellas.castShadow = true;
  world.scene.add(umbrellas);
  for (const [x, z] of [[-39, -3.5], [-44.5, 14.5], [-36.5, 24], [-45, -12]]) {
    batch.box([0.8, 0.05, 0.6], [x + 0.4, 0.75, z], m.planks);
    batch.box([0.7, 0.72, 0.5], [x + 0.4, 0.37, z], m.planksDark);
    props.add("stool", placement([x - 0.35, 0, z + 0.1]));
    vendors.push({ position: new THREE.Vector3(x - 0.35, 0, z + 0.1), yaw: Math.PI / 2, mode: "sitTalk", seat: 0.43 });
    addCollider(x - 0.2, z - 0.4, x + 0.9, z + 0.4);
    for (let k = 0; k < 6; k += 1) props.add(["oranges", "avocados", "bananas"][k % 3], placement([x + 0.2 + (k % 3) * 0.2, 0.78, z - 0.15 + Math.floor(k / 3) * 0.3], null, [0.35, 0.35, 0.35]));
  }
  const cars = [[0xc9ccd0, "hatch", -38.5, -9, 0.15], [0x22262b, "sedan", -45, 22, -0.1], [0xb8bcc0, "sedan", -45.5, -18, 0.05]];
  for (const [hex, kind, x, z, turn] of cars) {
    const car = carGroup(world, hex, kind);
    car.position.set(x, 0, z);
    car.rotation.y = Math.PI / 2 + turn;
    batch.addObject(car);
    addCollider(x - 1.0, z - 2.3, x + 1.0, z + 2.3);
  }
  const lamps = [];
  for (let z = -40; z <= 50; z += 15) {
    batch.cylinder(0.07, 0.09, 6.5, [-47.8, 3.25, z], m.steelDark, 6);
    batch.box([1.4, 0.08, 0.1], [-47.2, 6.45, z], m.steelDark);
    props.add("bulb", placement([-46.6, 6.3, z], null, [1.6, 1, 1.6]));
    lamps.push(new THREE.Vector3(-46.6, 6.3, z));
  }
  return { lamps };
}

function buildAcrossRoad(world, batch) {
  /** Line the far side of the road with two-storey shop buildings painted in Kigali colours. */
  const m = world.materials;
  let z = -70;
  let index = 0;
  const signs = new Map();
  while (z < 80) {
    const width = 8 + seeded(index, 1) * 6;
    const floors = seeded(index, 2) < 0.3 ? 1 : 2;
    const height = floors * 3.4 + 0.6;
    const depth = 9;
    const x = LAYOUT.across.x - depth / 2;
    const wall = m[ACROSS_COLORS[index % ACROSS_COLORS.length]];
    batch.box([depth, height, width - 0.3], [x, height / 2, z + width / 2], wall);
    batch.box([depth + 0.4, 0.4, width], [x, height + 0.2, z + width / 2], m.plasterWhite);
    batch.box([0.1, 2.6, width - 1.2], [x + depth / 2 + 0.02, 1.4, z + width / 2], m.corrGrey);
    if (floors === 2) {
      batch.box([1.2, 0.15, width - 0.8], [x + depth / 2 + 0.6, 3.6, z + width / 2], m.concrete);
      batch.box([0.05, 0.9, width - 0.8], [x + depth / 2 + 1.2, 4.1, z + width / 2], m.steelDark);
      for (let w = 1.4; w < width - 1; w += 2.4) batch.box([0.06, 1.3, 1.2], [x + depth / 2 + 0.03, 5.0, z + w], m.glass);
    }
    const text = ACROSS_SIGNS[index % ACROSS_SIGNS.length];
    if (!signs.has(text)) {
      signs.set(text, new THREE.MeshLambertMaterial({ map: signTexture(text, { background: ["#f4ead5", "#1f6fb5", "#c8352e", "#2a8a4a"][index % 4],
        color: index % 4 === 0 ? "#2b1d14" : "#fffaf0", width: 768, height: 140, font: "800 80px Georgia, serif" }) }));
    }
    batch.add(new THREE.PlaneGeometry(Math.min(width - 1.5, 6), 0.8), [x + depth / 2 + 0.08, 3.05, z + width / 2], signs.get(text), [0, Math.PI / 2, 0]);
    z += width;
    index += 1;
  }
}

function hillHeight(radius, angle) {
  /** Return the height of the green Kigali hills at a distance and bearing from the market. */
  const rise = Math.max(0, radius - 150);
  const wave = 0.55 + 0.3 * Math.sin(angle * 3 + 1.2) + 0.18 * Math.sin(angle * 7 + 0.4) + 0.1 * Math.sin(angle * 13);
  return rise * 0.22 * wave + Math.sin(radius * 0.03 + angle * 5) * 3;
}

function buildHills(world) {
  /** Ring the market with green hills dotted with houses and trees, fading into the haze. */
  const rings = 14;
  const steps = 96;
  const positions = [];
  const colors = [];
  const color = new THREE.Color();
  const point = (ring, step) => {
    const radius = 150 + ring * 22;
    const angle = (step / steps) * Math.PI * 2;
    return [Math.cos(angle) * radius, hillHeight(radius, angle), Math.sin(angle) * radius];
  };
  for (let ring = 0; ring < rings; ring += 1) {
    for (let step = 0; step < steps; step += 1) {
      const a = point(ring, step);
      const b = point(ring + 1, step);
      const c = point(ring + 1, step + 1);
      const d = point(ring, step + 1);
      for (const vertex of [a, c, b, a, d, c]) {
        positions.push(...vertex);
        const shade = seeded(ring * 131 + step, 4);
        color.setHex(shade < 0.15 ? 0x9a6a4a : shade < 0.6 ? 0x5f8a45 : 0x4f7a3a).multiplyScalar(0.85 + seeded(step, ring) * 0.25);
        colors.push(color.r, color.g, color.b);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  world.scene.add(new THREE.Mesh(geometry, world.materials.vertex));
  const houses = new THREE.InstancedMesh(new THREE.BoxGeometry(5, 3, 6).translate(0, 1.5, 0), new THREE.MeshLambertMaterial(), 520);
  const roofs = new THREE.InstancedMesh(new THREE.ConeGeometry(4.6, 1.8, 4).rotateY(Math.PI / 4).scale(1.1, 1, 1.3).translate(0, 3.9, 0),
    new THREE.MeshLambertMaterial(), 520);
  const trees = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(3, 0).translate(0, 5, 0), new THREE.MeshLambertMaterial({ color: 0x3f6a32 }), 380);
  const holder = new THREE.Object3D();
  const tint = new THREE.Color();
  for (let i = 0; i < 520; i += 1) {
    const radius = 165 + seeded(i, 1) * 170;
    const angle = seeded(i, 2) * Math.PI * 2;
    holder.position.set(Math.cos(angle) * radius, hillHeight(radius, angle) - 0.5, Math.sin(angle) * radius);
    holder.rotation.set(0, seeded(i, 3) * 3, 0);
    holder.scale.setScalar(0.8 + seeded(i, 4) * 0.6);
    holder.updateMatrix();
    houses.setMatrixAt(i, holder.matrix);
    roofs.setMatrixAt(i, holder.matrix);
    houses.setColorAt(i, tint.setHex([0xf1e6cf, 0xe9d8b8, 0xf6f3ec, 0xd9b88f, 0xc9a07a][i % 5]));
    roofs.setColorAt(i, tint.setHex([0x9a4a2a, 0x8c949a, 0x7a3a22, 0xa0623a, 0x5a6a72][(i * 3) % 5]));
  }
  for (let i = 0; i < 380; i += 1) {
    const radius = 155 + seeded(i, 7) * 200;
    const angle = seeded(i, 8) * Math.PI * 2;
    holder.position.set(Math.cos(angle) * radius, hillHeight(radius, angle) - 1, Math.sin(angle) * radius);
    holder.scale.set(1 + seeded(i, 9), 1 + seeded(i, 10) * 1.2, 1 + seeded(i, 9));
    holder.updateMatrix();
    trees.setMatrixAt(i, holder.matrix);
  }
  world.scene.add(houses, roofs, trees);
}

function buildTrees(world) {
  /** Plant a few big shade trees around the yard and along the road. */
  const spots = [[-47, -26], [-46.5, 30], [-34, -24], [-59, -35], [-59, 18], [-59, 46], [66, -24], [68, 10], [66, 30], [-6, -24], [10, -22]];
  const trunks = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.22, 0.35, 5, 6).translate(0, 2.5, 0), world.materials.woodDark, spots.length);
  const crowns = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(2.6, 1).scale(1.3, 0.8, 1.3).translate(0, 6, 0),
    new THREE.MeshLambertMaterial({ color: 0x4a7a35 }), spots.length * 2);
  spots.forEach(([x, z], i) => {
    trunks.setMatrixAt(i, placement([x, 0, z]));
    crowns.setMatrixAt(i * 2, placement([x, 0, z], [0, i, 0], [1, 1, 1]));
    crowns.setMatrixAt(i * 2 + 1, placement([x + 1.2, 1.1, z - 0.8], [0, i * 2, 0], [0.7, 0.7, 0.7]));
  });
  trunks.castShadow = crowns.castShadow = true;
  world.scene.add(trunks, crowns);
}

function buildGlow(world, points) {
  /** Draw a warm glow on every bulb at dusk as one cheap point cloud. */
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.PointsMaterial({ map: glowTexture(), size: 3.2, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, opacity: 0, color: 0xffd9a0 });
  const glow = new THREE.Points(geometry, material);
  glow.renderOrder = 2;
  world.scene.add(glow);
  return material;
}

export function buildMarket(world) {
  /** Build the whole of Kimironko around our lane and return the parts other modules animate or read. */
  const batch = new StaticBatch(world);
  const props = new Props(world);
  props.batch = batch;
  defineProps(props, world);
  const vendors = [];
  const riders = [];
  buildGround(world, batch);
  buildFacade(world, batch, props);
  const northStalls = buildNorthTerrace(world, batch, props, vendors);
  buildSouthSide(world, batch, props, vendors);
  const hall = buildHall(world, batch, props, vendors);
  const yard = buildYard(world, batch, props, vendors, riders);
  buildAcrossRoad(world, batch);
  buildHills(world);
  buildTrees(world);
  const bulbs = [];
  for (const [name, kind] of props.kinds) {
    if (name === "bulb") for (const matrix of kind.matrices) bulbs.push(new THREE.Vector3().setFromMatrixPosition(matrix));
  }
  return { batch, props, vendors, riders, northStalls, hall, yard, bulbs, finish: () => finishMarket(world, batch, props, bulbs, hall) };
}

function finishMarket(world, batch, props, bulbs, hall) {
  /** Merge the static batch, build the instanced goods and the dusk glow, once the shops have added their parts. */
  batch.build(world.scene);
  const meshes = props.build();
  const glow = buildGlow(world, bulbs);
  return { meshes, glow, roofMeshes: hall.roofMeshes, fans: hall.fans };
}

const fanHolder = new THREE.Object3D();

export function animateMarket(market, world, time) {
  /** Turn the hall fans, fade the bulb glow with the lamps, and keep the skylights as bright as the sky. */
  const { fans } = market.hall;
  fans.spots.forEach((spot, index) => {
    for (let blade = 0; blade < 3; blade += 1) {
      fanHolder.position.copy(spot);
      fanHolder.rotation.set(0, time * 2.2 + index + (blade * Math.PI * 2) / 3, 0);
      fanHolder.updateMatrix();
      fans.blades.setMatrixAt(index * 3 + blade, fanHolder.matrix);
    }
  });
  fans.blades.instanceMatrix.needsUpdate = true;
  if (market.finished) market.finished.glow.opacity = world.lampLevel * 0.9;
  const daylight = THREE.MathUtils.clamp(world.sunDirection.y * 2 + 0.3, 0.25, 1);
  world.materials.skylight.color.setRGB(daylight, daylight * 0.98, daylight * 0.93);
}

export function cutawayRoofs(market, camera) {
  /** Hide the hall roof when the camera is above it, so a top view looks into the hall. */
  if (!market.finished) return;
  const { xMin, xMax, zMin, zMax, ridge } = LAYOUT.hall;
  const over = camera.position.x > xMin && camera.position.x < xMax && camera.position.z > zMin && camera.position.z < zMax;
  const hide = over && camera.position.y > ridge + 0.5;
  for (const mesh of market.finished.roofMeshes) mesh.visible = !hide;
}

