![Crates.io](https://img.shields.io/crates/v/bevy_firework) ![Crates.io](https://img.shields.io/crates/d/bevy_firework)
[![CI](https://github.com/mbrea-c/bevy_firework/actions/workflows/ci.yaml/badge.svg)](https://github.com/mbrea-c/bevy_firework/actions/workflows/ci.yaml)

# Bevy Firework 🎆

Bevy firework is a particle system plugin where particles are simulated on the
CPU and use GPU batching for rendering. This allows each particle system to be
rendered in a single draw call (rather than one per particle).

While not as fast as a pure GPU-based particle system, this provides a massive
speed-up from the naive approach to CPU-based particles (making it possible to
render tens of thousands of particles without noticeable framerate drops) and maintains
much of the flexibility of CPU-based particle systems (e.g. easy access to
physics data for particle collision, simplified particle system animation).

![collision example](https://github.com/mbrea-c/bevy_firework/assets/31567043/11122ee3-3b66-4d18-8afd-3e7b1cb3b617)
![pbr example](/pbr_example.jpg) ![sparks example](/sparks_example.jpg)

## Current features

- _Billboarded_ particles.
- Configurable integration with Bevy's PBR rendering (i.e. particles can receive
  shadows, are affected by fog and lighting changes).
- Particle collision with arbitrary `bevy_xpbd` colliders.
- WASM and WebGL compatible.
- Soft particle edges.
- Animated properties: certain parameters can be defined as a custom curve to
  express changes over a particle's lifetime:
  - Scale
  - Color
- Randomized properties: certain properties can be randomized, so that they are
  randomly sampled for every particle:
  - Particle lifetime
  - Initial linear velocity
  - Initial radial velocity
  - Initial scale
- Emission shapes:
  - Point
  - Disk
  - Sphere
- One-shot emission mode, or continuous emission.

## Current limitations

- Can't use custom images for particles.

## Version table

| `bevy_firework` | `bevy` | `bevy_utilitarian` |
| --------------- | ------ | ------------------ |
| 0.1             | 0.12   | 0.2                |
| 0.2             | 0.12   | 0.3                |
| 0.3             | 0.13   | 0.4                |
| 0.4             | 0.14   | 0.5                |
