# Thermo–Hyperelastic Tube (FEniCSx)

3-D thermo-mechanical simulation of a cylindrical tube using FEniCSx.
The model couples transient heat conduction with finite-strain hyperelasticity
through a one-way staggered scheme, and applies realistic thermal and
mechanical boundary conditions.

The same geometry and mesh are used in a companion project on
[explicit nonlinear dynamics](https://github.com/elsayedahmadingmeca-wq/explicit-nonlinear-dynamics-fenicsx),
which adds inertia effects and wave propagation to the same tube.

---

## Physics

- Transient heat conduction — backward Euler (implicit), convection at bottom surface
- Prescribed temperature ramp at top surface
- Hyperelastic mechanics — compressible Neo-Hookean with thermal expansion
- Multiplicative decomposition: **F = Fₑ · Fth⁻¹**
- Top face fixed, bottom face: linear ramp + sinusoidal displacement

Coupling is one-way: **T → Fth(T) → u**

Temperature drives expansion and stress. Mechanical deformation does not
feed back into the heat equation (standard reduced thermoelastic assumption,
consistent with Bonnet and Ogden).

Full governing equations and weak form: [`physics.pdf`](physics.pdf)

---

## Model overview

**Thermal problem** (solved first at each step)

Find T such that:
```
∫ ρcp (T_new - T_old)/Δt · w dV  +  ∫ k ∇T · ∇w dV  +  ∫ h T w dS  =  ∫ h T∞ w dS
```
Backward Euler — unconditionally stable, first-order in time.

**Mechanical problem** (solved second, using updated T)

Find u such that:
```
∫ P(Fe) : ∇v dV = 0
```
where Fe = F · Fth⁻¹ strips thermal strain from the total deformation.
Solved with Newton–Raphson + backtracking line search.

---

## What I extract from the simulation

- 🔹 Temperature at tube centre vs time
- 🔹 Von Mises stress at tube centre vs time
- 🔹 Temperature field at final time step
- 🔹 Cauchy stress field (projected onto DG0 space)

All fields saved as XDMF and viewed in ParaView.

---

## Quick convergence sanity check

Simulation repeated on a finer mesh:

| Mesh  | T_centre [K] | σ_vm,centre [Pa] |
|-------|-------------|-----------------|
| Fine  | 332.30      | 1.039 × 10⁷     |
| Finer | 332.31      | 1.029 × 10⁷     |

Temperature difference < 0.5%. Von Mises difference ≈ 1%.
Temperature converges faster than stress (expected for P1 elements —
stress involves spatial gradients and converges one order slower).
The chosen mesh is sufficient for the qualitative objectives.

---

## Repo structure

```
src/
  coupling_dynamic_disp.py   ← main coupled solver
mesh/
  tube.xdmf
  tube_facets_linear.xdmf
Figures/                     ← ParaView output snapshots
physics.pdf                  ← governing equations and weak form
Report.pdf                   ← full simulation report with results
```

---

## Dependencies

```
dolfinx >= 0.7
mpi4py
petsc4py
numpy
```

---

## References

- Logg, Mardal, Wells. *Automated Solution of Differential Equations by the FEM (FEniCS Book).*
  Springer, 2012. — [fenicsproject.org/book](https://fenicsproject.org/book) (open access)
  — variational formulation, UFL, FEniCSx assembly patterns

- Dokken. *The FEniCSx Tutorial*, 2023. — [jsdokken.com/dolfinx-tutorial](https://jsdokken.com/dolfinx-tutorial)
  — nonlinear mechanics, heat equation, Newton solver in FEniCSx

- Bonet, Wood. *Nonlinear Continuum Mechanics for Finite Element Analysis*, 2nd ed. Cambridge, 2008.
  — Neo-Hookean energy, multiplicative F=FeFth decomposition, Cauchy stress

- Dokken, Bleyer et al. *FEniCSx numerical tours* — [bleyerj.github.io/fenics-tutorial](https://bleyerj.github.io/fenics-tutorial)
  — thermo-mechanical coupling examples in FEniCSx

