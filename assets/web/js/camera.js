import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { PointerLockControls } from "three/addons/PointerLockControls.js";
import { LAYOUT, pushOutOfColliders } from "./layout.js";

const OVERVIEW = { position: new THREE.Vector3(-25, 6.8, 4.6), target: new THREE.Vector3(2, 1.3, -2.2) };
const WALK_START = { position: new THREE.Vector3(-44, 1.6, 2.2), look: new THREE.Vector3(-30, 1.8, 1.2) };
const EYE_HEIGHT = 1.62;
const WALK_SPEED = 3.2;
const RUN_FACTOR = 2.2;
const WALK_RADIUS = 0.35;
const FLIGHT_SECONDS = 1.8;
const TOUR_HOLD = 7;
const TOUR_RETURN = 3.5;
const TOUR_KEYS = [
  [0, [-82, 24, 34], [-30, 4, 2]],
  [8, [-52, 6.5, 15], [-38, 1.4, 4]],
  [15, [-41, 2.5, 3.6], [-30, 3.6, 1]],
  [21, [-28.5, 2.3, 1.7], [-10, 2.3, 0.4]],
  [29, [-14, 3.3, 2.3], [6, 2.0, -1.2]],
  [36, [9, 5.8, 2.2], [30, 3.0, 1]],
  [44, [29, 4.2, 0.8], [45, 1.3, -6.5]],
  [52, [50, 6.2, 3.5], [40, 1.2, 12]],
  [59, [34, 4.6, 14.5], [10, 2.0, 14.2]],
  [66, [5.95, 2.7, 12.6], [5.95, 2.2, 0]],
];

export function easeInOut(t) {
  /** Smooth sine easing from 0 to 1. */
  return 0.5 - 0.5 * Math.cos(Math.PI * THREE.MathUtils.clamp(t, 0, 1));
}

function easeFlight(t) {
  /** Cubic ease-in-out for camera flights. */
  const x = THREE.MathUtils.clamp(t, 0, 1);
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

function groundUnder(position) {
  /** Return the floor height under the walker: the hall floor is raised a little. */
  const { hall } = LAYOUT;
  return position.x > hall.xMin && position.x < hall.xMax && position.z > hall.zMin && position.z < hall.zMax ? 0.1 : 0.04;
}

export function createCameraRig(world, dom, shops, hooks) {
  /** Create the camera rig with overview, walk, shop and tour modes, and return its controller. */
  const camera = world.camera;
  const orbit = new OrbitControls(camera, dom);
  orbit.enableDamping = true;
  orbit.maxPolarAngle = 1.48;
  orbit.minDistance = 4;
  orbit.maxDistance = 120;
  const walk = new PointerLockControls(camera, dom);
  const rig = {
    mode: "overview", orbit, walk, look: OVERVIEW.target.clone(), flight: null, tour: null, shopIndex: null,
    keys: new Set(), raycaster: new THREE.Raycaster(), pointer: new THREE.Vector2(),
  };
  camera.position.copy(OVERVIEW.position);
  orbit.target.copy(OVERVIEW.target);

  function flyTo(position, look, then, seconds = FLIGHT_SECONDS) {
    /** Fly the camera smoothly from where it is to a position and look target, then run a callback. */
    orbit.enabled = false;
    rig.flight = { fromPosition: camera.position.clone(), fromLook: rig.look.clone(), position: position.clone(),
      look: look.clone(), time: 0, seconds, then };
  }

  function setMode(mode) {
    /** Switch camera mode, flying between views where it helps. */
    if (rig.walk.isLocked) rig.walk.unlock();
    rig.tour = null;
    rig.mode = mode;
    rig.shopIndex = null;
    hooks.onModeChange(mode);
    if (mode === "overview") flyTo(OVERVIEW.position, OVERVIEW.target, () => { orbit.target.copy(OVERVIEW.target); orbit.enabled = true; });
    if (mode === "walk") flyTo(WALK_START.position, WALK_START.look, () => camera.lookAt(WALK_START.look));
    if (mode === "tour") startTour(true);
  }

  function enterShop(index) {
    /** Fly inside a shop, where its chalkboard shows today's decision. */
    if (rig.walk.isLocked) rig.walk.unlock();
    rig.tour = null;
    rig.mode = "shop";
    rig.shopIndex = index;
    hooks.onModeChange("shop", index);
    flyTo(shops[index].anchors.inside, shops[index].anchors.look);
  }

  function leaveShop() {
    /** Fly back out of a shop to the overview. */
    setMode("overview");
  }

  function tourPath(leader) {
    /** Build the tour: over the road, through the gate, down the lane, round the hall, back by the clothes lane, into the leader's shop. */
    const shop = shops[leader];
    const keys = TOUR_KEYS.concat([
      [72, [shop.x0 + 3.2, 2.3, 2.2], [shop.x0, 1.9, -3]],
      [77, shop.anchors.inside.toArray(), shop.anchors.look.toArray()],
    ]);
    const vector = (values) => new THREE.Vector3(...values);
    return {
      leader, times: keys.map(([time]) => time), duration: keys[keys.length - 1][0],
      positions: new THREE.CatmullRomCurve3(keys.map(([, position]) => vector(position)), false, "centripetal"),
      looks: new THREE.CatmullRomCurve3(keys.map(([, , look]) => vector(look)), false, "centripetal"),
    };
  }

  function startTour(fromHere) {
    /** Start a tour loop that ends in the shop leading the market right now. */
    orbit.enabled = false;
    rig.tour = { time: 0, ...tourPath(hooks.leaderIndex()) };
    if (fromHere) flyTo(rig.tour.positions.getPoint(0), rig.tour.looks.getPoint(0), null, TOUR_RETURN);
  }

  function tourParameter(tour, time) {
    /** Map tour time onto the keyframe curves so each stretch takes the time its keyframes give it. */
    const { times } = tour;
    let i = 1;
    while (i < times.length - 1 && times[i] < time) i += 1;
    const local = THREE.MathUtils.clamp((time - times[i - 1]) / (times[i] - times[i - 1]), 0, 1);
    return (i - 1 + local) / (times.length - 1);
  }

  function updateTour(delta) {
    /** Move the camera along the tour, hold in the leader's shop, then fly back to the start and loop. */
    const tour = rig.tour;
    tour.time += delta;
    if (tour.time <= tour.duration) {
      const eased = tour.duration * easeInOut(tour.time / tour.duration) * 0.25 + tour.time * 0.75;
      const u = tourParameter(tour, eased);
      camera.position.copy(tour.positions.getPoint(u));
      rig.look.copy(tour.looks.getPoint(u));
      camera.lookAt(rig.look);
    } else if (tour.time > tour.duration + TOUR_HOLD) {
      startTour(true);
    }
  }

  function updateFlight(delta) {
    /** Advance a camera flight and finish it at the destination. */
    const flight = rig.flight;
    flight.time += delta / flight.seconds;
    const t = easeFlight(flight.time);
    camera.position.lerpVectors(flight.fromPosition, flight.position, t);
    rig.look.lerpVectors(flight.fromLook, flight.look, t);
    camera.lookAt(rig.look);
    if (flight.time >= 1) {
      rig.flight = null;
      if (flight.then) flight.then();
    }
  }

  function updateWalk(delta) {
    /** Move the walker with WASD, sliding along stalls, walls, tables, motos and cars instead of passing through. */
    if (!rig.walk.isLocked) return;
    const speed = WALK_SPEED * (rig.keys.has("ShiftLeft") || rig.keys.has("ShiftRight") ? RUN_FACTOR : 1) * delta;
    const forward = (rig.keys.has("KeyW") || rig.keys.has("ArrowUp") ? 1 : 0) - (rig.keys.has("KeyS") || rig.keys.has("ArrowDown") ? 1 : 0);
    const right = (rig.keys.has("KeyD") || rig.keys.has("ArrowRight") ? 1 : 0) - (rig.keys.has("KeyA") || rig.keys.has("ArrowLeft") ? 1 : 0);
    rig.walk.moveForward(forward * speed);
    rig.walk.moveRight(right * speed);
    const position = camera.position;
    const inside = position.x > LAYOUT.facadeX;
    position.x = THREE.MathUtils.clamp(position.x, -62, LAYOUT.hall.xMax - 0.5);
    position.z = THREE.MathUtils.clamp(position.z, inside ? LAYOUT.hall.zMin + 0.5 : -40, inside ? LAYOUT.hall.zMax - 0.5 : 40);
    pushOutOfColliders(position, WALK_RADIUS);
    position.y = groundUnder(position) + EYE_HEIGHT;
    rig.look.copy(position).add(camera.getWorldDirection(new THREE.Vector3()).multiplyScalar(8));
  }

  function pick(event) {
    /** Return the index of the shop under the pointer, or null. */
    const rect = dom.getBoundingClientRect();
    rig.pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    rig.raycaster.setFromCamera(rig.pointer, camera);
    const hit = rig.raycaster.intersectObjects(hooks.hitBoxes, false)[0];
    return hit ? hit.object.userData.shopIndex : null;
  }

  let downAt = null;
  dom.addEventListener("pointerdown", (event) => { downAt = [event.clientX, event.clientY]; });
  dom.addEventListener("click", (event) => {
    const dragged = downAt && Math.hypot(event.clientX - downAt[0], event.clientY - downAt[1]) > 5;
    if (rig.mode === "walk") {
      if (!rig.walk.isLocked && !rig.flight) rig.walk.lock();
      return;
    }
    if (!dragged && (rig.mode === "overview" || rig.mode === "tour")) {
      const index = pick(event);
      if (index !== null) enterShop(index);
    }
  });
  walk.addEventListener("lock", () => hooks.onWalkLock(true));
  walk.addEventListener("unlock", () => hooks.onWalkLock(false));
  window.addEventListener("keydown", (event) => rig.keys.add(event.code));
  window.addEventListener("keyup", (event) => rig.keys.delete(event.code));
  window.addEventListener("blur", () => rig.keys.clear());

  function update(delta) {
    /** Advance the active camera mode by one frame. */
    if (rig.flight) updateFlight(delta);
    else if (rig.mode === "tour" && rig.tour) updateTour(delta);
    else if (rig.mode === "walk") updateWalk(delta);
    else if (rig.mode === "overview") {
      orbit.update();
      rig.look.copy(orbit.target);
    }
  }

  return { rig, update, setMode, enterShop, leaveShop };
}
