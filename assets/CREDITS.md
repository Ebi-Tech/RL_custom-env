# Credits

Every third-party file shipped with the market visualisation, where it came from and its licence.
Everything else in `assets/web/` (the people, buildings, stalls, goods, vehicles, hills, signs, prints and the
market sound) is generated in code by this project.

## Textures (`assets/web/textures/`)

All from [ambientCG](https://ambientcg.com), released under **CC0 1.0** (public domain). The 1K JPG colour
and normal maps were downloaded, resized to 512 x 512 and recompressed; they are tinted in code.

| File | Source material | Link |
| --- | --- | --- |
| `corrugated_color.jpg`, `corrugated_normal.jpg` | Corrugated Steel 005 | https://ambientcg.com/view?id=CorrugatedSteel005 |
| `ground_color.jpg`, `ground_normal.jpg` | Ground 103 | https://ambientcg.com/view?id=Ground103 |
| `concrete_color.jpg` | Concrete 031 | https://ambientcg.com/view?id=Concrete031 |
| `planks_color.jpg`, `planks_normal.jpg` | Planks 023 A | https://ambientcg.com/view?id=Planks023A |
| `plaster_color.jpg` | Plaster 001 | https://ambientcg.com/view?id=Plaster001 |

## Libraries (`assets/web/vendor/`)

From [three.js](https://github.com/mrdoob/three.js) release r160 (npm `three@0.160.0`), **MIT licence**,
copyright the three.js authors. The licence text is in `assets/web/vendor/LICENSE`.

| File | Origin in three.js |
| --- | --- |
| `three.module.js` | `build/three.module.js` |
| `OrbitControls.js` | `examples/jsm/controls/OrbitControls.js` |
| `PointerLockControls.js` | `examples/jsm/controls/PointerLockControls.js` |
| `BufferGeometryUtils.js` | `examples/jsm/utils/BufferGeometryUtils.js` |
| `Sky.js` | `examples/jsm/objects/Sky.js` (sky brightness scaled at runtime, file unchanged) |

## Not shipped

The photos in `reference_photos/` were studied for layout, colours, goods and clothing only. None of them is
copied into the app or loaded by it.
