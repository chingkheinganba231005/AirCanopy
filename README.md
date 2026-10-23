# AirCanopy

Plant trees virtually anywhere in the world and see, in 3D, how they change street-level heat and air pollution.

AirCanopy builds a 3D model of a real place from open data (buildings, terrain, trees, roads), then simulates the wind, the sun, the heat and the chemistry of the air on a graphics card. A fast AI model trained on those simulations lets you try many tree layouts in seconds and find the best trade-off between cooler streets and cleaner air, with the physics available to check any answer.

## Status

Early development. AirCanopy is a rebuild of [Trees vs Breath](https://github.com/chingkheinganba231005/Trees-vs-breathe), a 2D prototype that reached the top 8 of HacKU 2026, into a 3D tool meant for real use.

The plan, the progress and a step-by-step explanation of how everything is built are in [`JOURNAL.md`](JOURNAL.md).

## What it will model

- 3D wind through real buildings and terrain, at real scale (large-eddy simulation with the lattice Boltzmann method)
- Trees as 3D crowns of leaves that slow the wind, bend in strong wind, shade, evaporate water and collect pollution
- Traffic emissions, nitrogen oxides, ozone and fine particles, with chemistry, deposition and the turbulence that traffic stirs up
- Sun, sky and heat in 3D, surfaces that store heat, buoyant warm air, and the UTCI "feels-like" temperature for people walking
- An AI surrogate with uncertainty, a design search, and a physics check for its answers

## Licence

MIT, see [`LICENSE`](LICENSE). Map and environmental data come from third parties under their own licences; each dataset's source and licence is recorded with it.
