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
| 3. Acceleration | $\ddot{\mathbf{U}}^{n+1} = \mathbf{M}_L^{-1}\,\mathbf{f}_{int}(\mathbf{U}^{n+1,\*})$ |
| 4. Velocity corrector | $\dot{\mathbf{U}}^{n+1} = \dot{\mathbf{U}}^n + \frac{\Delta t}{2}\!\left(\ddot{\mathbf{U}}^n + \ddot{\mathbf{U}}^{n+1}\right)$ |

---

## Critical Numerical Implementation

### 1. Kinematic Consistency (Drift Fix)

Over $10^5+$ steps, Dirichlet DOFs accumulate spurious velocity residuals. We enforce analytical
kinematics post-corrector to prevent boundary drift:
```python
