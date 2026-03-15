# Explicit Nonlinear Dynamics: Hyperelastic Tube (FEniCSx)

A high-performance **3D explicit finite element solver** for the transient dynamics of a hyperelastic tube.
Built with **FEniCSx**, this solver handles high-frequency kinematic loading where traditional implicit
methods become prohibitively expensive due to ill-conditioning.

> [!TIP]
> This project is part of a two-project portfolio. See the companion
> [thermo_mechanical/](../thermo_mechanical/) repository for quasi-static comparisons using the same geometry.

---

## Physics & Methodology

### Constitutive Model

The material is modeled as a **compressible Neo-Hookean** solid with $\nu = 0.49$. In this
near-incompressible regime, the bulk modulus $K$ is significantly larger than the shear modulus $\mu$,
requiring strict time-step control.

The strain energy density is:

$$\psi(\mathbf{F}) = \frac{\mu}{2}(I_c - 3) - \mu \ln J + \frac{\lambda}{2}(\ln J)^2$$

- **UFL Automatic Differentiation:** The first Piola–Kirchhoff stress is computed symbolically:

$$\mathbf{P} = \frac{\partial \psi}{\partial \mathbf{F}}$$

- **Restoring Forces:** The internal force vector $\mathbf{f}_{int} = -\partial E_{pot}/\partial \mathbf{U}$
  is consistent with the equation of motion $\mathbf{M}\ddot{\mathbf{U}} = \mathbf{f}_{int}$.

---

### Explicit Newmark Scheme ($\beta = 0,\ \gamma = 1/2$)

The solver uses the **central difference** method. A **row-sum lumped mass matrix** $\mathbf{M}_L$
yields an $\mathcal{O}(N)$ update per step, bypassing global linear solvers (KSP).

#### Update Cycle ($n \to n+1$)

| Step | Operation |
|------|-----------|
| 1. Predictor | $\mathbf{U}^{n+1,\*} = \mathbf{U}^n + \Delta t\,\dot{\mathbf{U}}^n + \frac{\Delta t^2}{2}\ddot{\mathbf{U}}^n$ |
| 2. Force assembly | Evaluate $\mathbf{f}_{int}(\mathbf{U}^{n+1,\*})$ with MPI ghost scatters |
| 3. Acceleration | $$\ddot{\mathbf{U}}^{n+1} = \mathbf{M}_L^{-1}\,\mathbf{f}_{int}(\mathbf{U}^{n+1,*})$$ |
| 4. Velocity corrector | $\dot{\mathbf{U}}^{n+1} = \dot{\mathbf{U}}^n + \frac{\Delta t}{2}\!\left(\ddot{\mathbf{U}}^n + \ddot{\mathbf{U}}^{n+1}\right)$ |

---

## Critical Numerical Implementation

### 1. Kinematic Consistency (Drift Fix)

Over $10^5+$ steps, Dirichlet DOFs accumulate spurious velocity residuals. We enforce analytical
kinematics post-corrector to prevent boundary drift:
```python
# Enforce prescribed kinematics at boundary DOFs
v_new[dofs_bot_z] = vz_bot   # Prescribed velocity
a_new[dofs_bot_z] = az_bot   # Prescribed acceleration — critical for energy balance
```

> [!WARNING]
> Setting $\ddot{\mathbf{U}}_{BC} = 0$ instead of the analytical value breaks the Hamiltonian
> energy balance and is a common source of silent solver error.

### 2. CFL Stability

The timestep is governed by the dilatational wave speed $c_{mech}$:

$$\Delta t = \alpha_s \frac{h_{\min}}{c_{mech}}, \qquad c_{mech} = \sqrt{\frac{\lambda + 2\mu}{\rho}}$$

| Parameter | Value |
|-----------|-------|
| Wave speed $c_{mech}$ | $\approx 115\ \text{m/s}$ |
| Timestep $\Delta t$ | $\approx 5.7 \times 10^{-7}\ \text{s}$ |
| Safety factor $\alpha_s$ | $0.3$ |

### 3. MPI Ghost Communication

Parallel assembly requires two scatter operations per step:
```python
u.x.scatter_reverse(...)   # REVERSE: sum ghost → owner
u.x.scatter_forward(...)   # FORWARD: broadcast owner → ghosts
```

> [!CAUTION]
> Omitting `REVERSE` produces spurious stress concentrations at partition boundaries
> that are difficult to detect without a momentum residual check.

---

## Verification

### Hamiltonian Conservation

For an undamped system, the Hamiltonian $\mathcal{H} = E_{kin} + E_{pot} - W_{ext}$ must remain constant.
Reaction forces at Dirichlet nodes are used to compute external work:

$$R_i = (M_L)_{ii}\,a_{\text{prescribed},\,i} - (f_{int})_i, \qquad \Delta W = \sum_i R_i\,\Delta u_i$$

### Results

| Check | Metric | Value |
|-------|--------|-------|
| Energy balance | Relative Hamiltonian drift | $0.1218\%$ |
| Momentum residual | Max relative residual | $2.69 \times 10^{-9}$ |

![Energy and Hamiltonian](energy_hamiltonian.png)
*Near-constant Hamiltonian (blue) despite large kinetic ↔ potential energy exchanges.*

![Momentum Residual](momentum_residual.png)
*Relative momentum residual near $10^{-9}$, confirming parallel assembly correctness.*

---

## Why Not Implicit?

Implicit Newmark ($\beta = 1/4$) requires a nonlinear solve every step — unattractive here because:

1. **Ill-conditioning:** Near-incompressible tangent stiffness ($\kappa \approx 83$) degrades KSP convergence.
2. **No timestep advantage:** Resolving the 200 Hz shock still requires $\Delta t \sim 10^{-6}\ \text{s}$.
3. **Cost:** $87{,}000$ tangent assemblies vs. a single explicit loop.

---

## References

- J. Bonet, R. D. Wood. *Nonlinear Continuum Mechanics for Finite Element Analysis*. Cambridge, 2008.
- A. Logg et al. *Automated Solution of Differential Equations by the FEM*. Springer, 2012.
- J. S. Dokken. *The FEniCSx Tutorial*, 2023.
