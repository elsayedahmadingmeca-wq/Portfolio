# Computational Mechanics Portfolio — FEniCSx / Python

Personal research projects in nonlinear finite element mechanics,
implemented from scratch in FEniCSx. Both projects share the same
hyperelastic tube geometry and Neo-Hookean material model, allowing
direct comparison between static and dynamic stress fields.

---

## Projects

### [`explicit_dynamics/`](explicit_dynamics/)
**Explicit Nonlinear Dynamics — Hyperelastic Tube**

3-D explicit central-difference solver for transient structural dynamics
under prescribed shock kinematics. Focus on numerical scheme correctness:
CFL stability, lumped mass, kinematic BC consistency, and inline
energy/momentum validation.

`FEniCSx · Python · MPI · Newmark β=0 · Neo-Hookean · CFL`

---

### [`thermo_mechanical/`](thermo_mechanical/)
**Transient Thermo-Mechanical FEM — Hyperelastic Tube**

Sequential one-way coupling: transient heat equation (implicit Euler)
driving a quasi-static large-deformation mechanical solve. Thermal
expansion decomposed as F = F_e · F_th⁻¹. Newton solver with
backtracking line-search and hypre preconditioner.

`FEniCSx · Python · Newton · Neo-Hookean · Thermo-mechanical coupling`

---

## Common geometry

Both projects use the same tube mesh generated with Gmsh
(`explicit_dynamics/mesh/generate_mesh.py`).
Material: E = 5 MPa, ν = 0.49, ρ = 1250 kg/m³ — soft
nearly-incompressible hyperelastic solid representative of
biological tissue or elastomeric components.

---

## Skills demonstrated

| Topic | Project |
|---|---|
| Explicit time integration, CFL, mass lumping | `explicit_dynamics` |
| Implicit time integration, Newton line-search | `thermo_mechanical` |
| Total Lagrangian formulation, UFL autodiff | both |
| Prescribed kinematic BCs, moving Dirichlet | `explicit_dynamics` |
| Thermal expansion, multiplicative decomposition F=F_e·F_th⁻¹ | `thermo_mechanical` |
| Energy balance, reaction force derivation | `explicit_dynamics` |
| MPI parallel assembly, ghost DOF communication | both |
| Stress post-processing (DG0 projection) | `thermo_mechanical` |

---

## Dependencies

```
dolfinx >= 0.7
mpi4py
petsc4py
numpy
gmsh
```

---

## References

- Carlberg, Tuminaro, Boggs. *Preserving Lagrangian structure in nonlinear model reduction.*
  SIAM J. Sci. Comput., 37(2):B153–B184, 2015. — [arXiv:1401.8044](https://arxiv.org/abs/1401.8044)
- Bonet, Wood. *Nonlinear Continuum Mechanics for Finite Element Analysis*, 2nd ed. Cambridge, 2008.
- Logg, Mardal, Wells. *Automated Solution of Differential Equations by the FEM.*
  Springer, 2012. — [fenicsproject.org/book](https://fenicsproject.org/book)
- Dokken. *The FEniCSx Tutorial*, 2023. — [jsdokken.com/dolfinx-tutorial](https://jsdokken.com/dolfinx-tutorial)
