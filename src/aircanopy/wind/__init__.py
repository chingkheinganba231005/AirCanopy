"""The 3D wind solver: lattice Boltzmann large-eddy simulation (phase P2 in JOURNAL.md).

- `lattice.py`: the velocity sets D3Q19 and D3Q27.
- `units.py`: metres and seconds to lattice units and back.
- `lbm.py`: one time step, written once for NumPy and JAX.
- `solver.py`: runs steps with NumPy (for checking) or JAX (compiled, fast).
"""
