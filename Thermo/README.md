# Transient Thermo-Mechanical FEM — Hyperelastic Tube (FEniCSx)

Sequential one-way thermo-mechanical coupling for a hyperelastic tube
under prescribed thermal loading, built from scratch in FEniCSx.

Part of a two-project portfolio — see the companion project
[`explicit_dynamics/`](../explicit_dynamics/) which uses the same
geometry and material under dynamic shock loading, allowing direct
comparison of static vs dynamic stress fields.

---

## Physics

- Transient heat equation — implicit Euler time integration
- Quasi-static large-deformation mechanics — total Lagrangian
- One-way sequential coupling: thermal field drives mechanical solve
- Multiplicative decomposition: F = F_e · F_th⁻¹
- Compressible Neo-Hookean strain energy on elastic part F_e
- ν = 0.49 (near-incompressible), E = 5 MPa, ρ = 1250 kg/m³

Thermal boundary conditions: imposed temperature, Newton convection,
insulated surface.

---

## Numerical details

**Time integration — heat equation**

Implicit Euler on the transient heat equation:
```
ρ·c_p·(T^{n+1} - T^n)/Δt + K·T^{n+1} = f_th
```
Unconditionally stable — no CFL constraint on the thermal step.
Large time steps can be used as long as the thermal gradient
evolution is slow relative to the mechanical response.

**Nonlinear mechanical solve — Newton with line-search**

At each thermal step, the mechanical problem is solved by Newton
iteration with backtracking line-search (Armijo condition):
```
K_T · δU = -R(U)
U ← U + β · δU      (β ∈ (0,1] found by backtracking)
```
Preconditioner: hypre algebraic multigrid via PETSc,
effective for the near-incompressible system.

**Thermal expansion — multiplicative decomposition**

The deformation gradient is split as:
```
F = F_e · F_th⁻¹
F_th = (1 + α·ΔT)·I     (isotropic thermal expansion)
```
The Neo-Hookean strain energy is evaluated on F_e only,
so thermal strains generate no stress when unconstrained.
Cauchy stress is obtained by push-forward of the PK1 stress.

**Stress post-processing**

Von Mises stress and full stress tensor are projected onto
a discontinuous Galerkin space (DG0 — piecewise constant per element)
for ParaView output:
```python
V_dg = fem.functionspace(mesh, ("DG", 0))
sigma_vm = fem.Function(V_dg, name="von_mises")
```

---

## What the simulation produces

- 🔹 Temperature field T(x,t) — XDMF → ParaView
- 🔹 Displacement field u(x,t)
- 🔹 Von Mises stress and full stress tensor (DG0 projection)
- 🔹 Time history of max temperature and max von Mises stress


## References

- Bonet, Wood. *Nonlinear Continuum Mechanics for Finite Element Analysis*, 2nd ed. Cambridge, 2008.
- Logg, Mardal, Wells. *Automated Solution of Differential Equations by the FEM.*
  Springer, 2012. — [fenicsproject.org/book](https://fenicsproject.org/book)
- Dokken. *The FEniCSx Tutorial*, 2023. — [jsdokken.com/dolfinx-tutorial](https://jsdokken.com/dolfinx-tutorial)
