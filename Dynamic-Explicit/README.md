# Explicit Nonlinear Dynamics: Hyperelastic Tube (FEniCSx)

A high-performance **3D explicit finite element solver** for the transient dynamics of a hyperelastic tube.  
Built with **FEniCSx**, this solver targets **high-frequency dynamic loading** where traditional implicit
methods become prohibitively expensive due to ill-conditioning.

> [!TIP]
> This project is part of a two-project portfolio. See the companion  
> **thermo_mechanical/** repository for quasi-static comparisons using the same geometry.

---

# Physics & Methodology

## Constitutive Model

The material is modeled as a **compressible Neo-Hookean solid** with Poisson ratio

$$
\nu = 0.49
$$

In this **near-incompressible regime**, the bulk modulus \(K\) is significantly larger than the shear modulus \(\mu\).  
This introduces stiffness in the volumetric response and requires **strict timestep control** for explicit dynamics.

The strain energy density function is

$$
\psi(\mathbf{F}) =
\frac{\mu}{2}(I_c - 3)
- \mu \ln J
+ \frac{\lambda}{2}(\ln J)^2
$$

where

- \( \mathbf{F} \) is the deformation gradient
- \( I_c = \mathrm{tr}(\mathbf{F}^T\mathbf{F}) \)
- \( J = \det(\mathbf{F}) \)

### Automatic Stress Computation

Using **UFL automatic differentiation**, the first Piola–Kirchhoff stress tensor is computed symbolically

$$
\mathbf{P} =
\frac{\partial \psi}{\partial \mathbf{F}}
$$

### Internal Restoring Forces

The internal nodal restoring force vector is obtained from the variation of the potential energy

$$
\mathbf{f}_{\text{int}}
=
-\frac{\partial E_{\text{pot}}}{\partial \mathbf{U}}
$$

which yields the semi-discrete equation of motion

$$
\mathbf{M}\ddot{\mathbf{U}} = \mathbf{f}_{\text{int}}
$$

---

# Explicit Time Integration

## Explicit Newmark Scheme

The solver uses the **central difference method**, corresponding to the explicit Newmark parameters

$$
\beta = 0, \qquad \gamma = \frac{1}{2}
$$

A **row-sum lumped mass matrix** \( \mathbf{M}_L \) allows an \( \mathcal{O}(N) \) update per timestep, avoiding global linear solvers (PETSc KSP).

---

## Time Integration Cycle

| Step | Operation |
|-----|------|
| Predictor | \( \mathbf{U}^{n+1,*} = \mathbf{U}^n + \Delta t\,\dot{\mathbf{U}}^n + \frac{\Delta t^2}{2}\ddot{\mathbf{U}}^n \) |
| Force assembly | Evaluate \( \mathbf{f}_{\text{int}}(\mathbf{U}^{n+1,*}) \) |
| Acceleration | \( \ddot{\mathbf{U}}^{n+1} = \mathbf{M}_L^{-1}\mathbf{f}_{\text{int}}(\mathbf{U}^{n+1,*}) \) |
| Velocity corrector | \( \dot{\mathbf{U}}^{n+1} = \dot{\mathbf{U}}^n + \frac{\Delta t}{2}(\ddot{\mathbf{U}}^n + \ddot{\mathbf{U}}^{n+1}) \) |

Because the mass matrix is diagonal, computing \( \mathbf{M}_L^{-1} \) reduces to **element-wise division**.

---

# Critical Numerical Implementation

## 1. Kinematic Consistency (Boundary Drift Fix)

For simulations with **\(10^5+\)** timesteps, Dirichlet DOFs can accumulate spurious velocity residuals.

To prevent drift, analytical kinematics are enforced **after the velocity corrector**.

```python
# Enforce prescribed kinematics at boundary DOFs
v_new[dofs_bot_z] = vz_bot
a_new[dofs_bot_z] = az_bot
```

> [!WARNING]  
> Setting \( \ddot{\mathbf{U}}_{BC} = 0 \) instead of the analytical value breaks the Hamiltonian energy balance and introduces silent solver errors.

---

## 2. CFL Stability Condition

The timestep is controlled by the **dilatational wave speed**

$$
c_{\text{mech}} =
\sqrt{\frac{\lambda + 2\mu}{\rho}}
$$

The explicit timestep is therefore

$$
\Delta t =
\alpha_s
\frac{h_{\min}}{c_{\text{mech}}}
$$

| Parameter | Value |
|------|------|
| Wave speed \(c_{\text{mech}}\) | ≈ 115 m/s |
| Timestep \( \Delta t \) | ≈ \(5.7\times10^{-7}\) s |
| Safety factor \( \alpha_s \) | 0.3 |

---

## 3. MPI Ghost Communication

Parallel force assembly requires two ghost communications per timestep.

```python
u.x.scatter_reverse(...)   # sum ghost contributions → owner
u.x.scatter_forward(...)   # broadcast owner values → ghosts
```

> [!CAUTION]  
> Omitting the **REVERSE scatter** produces artificial stress concentrations along MPI partition boundaries.

---

# Verification

## Hamiltonian Conservation

For an undamped system, the Hamiltonian

$$
\mathcal{H}
=
E_{\text{kin}}
+
E_{\text{pot}}
-
W_{\text{ext}}
$$

should remain constant.

Reaction forces at Dirichlet nodes are computed as

$$
R_i =
(M_L)_{ii} a_{\text{prescribed},i}
-
(f_{\text{int}})_i
$$

and the external work increment is

$$
\Delta W = \sum_i R_i \Delta u_i
$$

---

# Results

| Check | Metric | Value |
|------|------|------|
| Energy balance | Relative Hamiltonian drift | 0.1218 % |
| Momentum residual | Max relative residual | \(2.69\times10^{-9}\) |

### Energy Evolution

![Energy and Hamiltonian](energy_hamiltonian.png)

*Near-constant Hamiltonian despite large kinetic ↔ potential energy exchange.*

### Momentum Residual

![Momentum residual](momentum_residual.png)

*Relative momentum residual near \(10^{-9}\), confirming parallel assembly correctness.*

---

# Why Not Implicit?

Implicit Newmark \( \beta = 1/4 \) requires a nonlinear solve every timestep, which is unattractive here because:

1. **Ill-conditioning**  
   Near-incompressible tangent stiffness (\( \kappa \approx 83 \)) degrades Krylov solver convergence.

2. **No timestep advantage**  
   Resolving the ~200 Hz shock still requires \( \Delta t \sim 10^{-6} \) s.

3. **Computational cost**  
   Implicit methods would require ~87,000 nonlinear tangent assemblies versus a single explicit update loop.

---

# References

- J. Bonet, R. D. Wood — *Nonlinear Continuum Mechanics for Finite Element Analysis*. Cambridge University Press, 2008.
- A. Logg et al. — *Automated Solution of Differential Equations by the Finite Element Method*. Springer, 2012.
- J. S. Dokken — *The FEniCSx Tutorial*, 2023.
