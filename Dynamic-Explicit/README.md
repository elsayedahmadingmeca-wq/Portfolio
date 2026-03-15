# Explicit Nonlinear Dynamics: Hyperelastic Tube (FEniCSx)

A high-performance **3D explicit finite element solver** for the transient dynamics of a hyperelastic tube. Built with **FEniCSx**, this solver handles high-frequency kinematic loading where traditional implicit methods become prohibitively expensive due to ill-conditioning.

> [!TIP]
> This project is part of a two-project portfolio. See the companion [thermo_mechanical/](../thermo_mechanical/) repository for quasi-static comparisons using the same geometry.

## 🚀 Physics & Methodology

### Constitutive Model
The material is modeled as a **compressible Neo-Hookean** solid with $\nu = 0.49$. In this near-incompressible regime, the bulk modulus $K$ is significantly larger than the shear modulus $\mu$, requiring strict time-step control.

* **UFL Automatic Differentiation:** We leverage the Unified Form Language (UFL) to differentiate the strain energy density $\psi$ symbolically:
    $$\mathbf{P} = \frac{\partial \psi}{\partial \mathbf{F}}$$
* **Restoring Forces:** The internal force vector $\mathbf{f}_{int} = -\frac{\partial E_{pot}}{\partial \mathbf{U}}$ is computed to remain consistent with the equation of motion $\mathbf{M}\ddot{\mathbf{U}} = \mathbf{f}_{int}$.

### Explicit Newmark Scheme ($\beta=0, \gamma=1/2$)
The solver utilizes the **central difference** method. By implementing a **row-sum lumped mass matrix** ($\mathbf{M}_L$), we achieve an $\mathcal{O}(N)$ update per step, bypassing the need for global linear solvers (KSP).

#### The Update Cycle ($n \to n+1$):
1.  **Predictor:** $\mathbf{U}_{n+1,*} = \mathbf{U}_n + \Delta t \dot{\mathbf{U}}_n + \frac{\Delta t^2}{2} \ddot{\mathbf{U}}_n$
2.  **Internal Force Assembly:** Evaluated at $\mathbf{U}_{n+1,*}$ with MPI ghost scatters.
3.  **Acceleration Update:** $\ddot{\mathbf{U}}_{n+1} = \mathbf{M}_L^{-1} \mathbf{f}_{int}(\mathbf{U}_{n+1,*})$
4.  **Velocity Corrector:** $\dot{\mathbf{U}}_{n+1} = \dot{\mathbf{U}}_n + \frac{\Delta t}{2}(\ddot{\mathbf{U}}_n + \ddot{\mathbf{U}}_{n+1})$

---

## 🛠 Critical Numerical Implementation

### 1. Kinematic Consistency (The "Drift" Fix)
In explicit integration over $10^5+$ steps, Dirichlet DOFs can accumulate spurious velocity residuals. To prevent boundary drift, we enforce analytical kinematics post-correction:

```python
# Analytical enforcement at boundary DOFs
v_new[dofs_bot_z] = vz_bot   # Prescribed velocity
a_new[dofs_bot_z] = az_bot   # Prescribed acceleration (Critical for Energy Balance!)
