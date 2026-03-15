# Explicit Nonlinear Dynamics: Hyperelastic Tube (FEniCSx)

A high-performance **3-D explicit finite element solver** for the transient dynamics of a hyperelastic tube. This solver is built from the ground up in **FEniCSx** to handle high-frequency kinematic loading where traditional implicit methods fail or become prohibitively expensive.

This project is part of a two-project portfolio. See the companion [thermo_mechanical/](../thermo_mechanical/) repository for quasi-static comparisons using the same geometry.

## 🚀 Physics & Methodology

### Constitutive Model
The material is modeled as a **compressible Neo-Hookean** solid with $\nu = 0.49$. In this near-incompressible regime, the bulk modulus $K$ is roughly 83 times the shear modulus $\mu$, creating a challenging numerical environment.

* **UFL Automatic Differentiation:** Rather than deriving the first Piola–Kirchhoff stress $\mathbf{P} = \partial\psi/\partial\mathbf{F}$ by hand, we leverage UFL to differentiate the strain energy density $\psi$ symbolically.
* **Restoring Forces:** The internal force $f_{int} = -\partial E_{pot}/\partial U$ is computed such that it remains consistent with the equation of motion $M\ddot{U} = f_{int}$.

### Explicit Newmark Scheme ($\beta=0, \gamma=1/2$)
The solver utilizes the **central difference** family of Newmark integrators. Unlike implicit versions, this approach uses a **row-sum lumped mass matrix** ($M_L$), enabling an $\mathcal{O}(N)$ update without any global linear solves.

#### The Update Cycle ($n \to n+1$):
1.  **Predictor:** $U_{n+1,*} = U_n + \Delta t \dot{U}_n + \frac{\Delta t^2}{2} \ddot{U}_n$
2.  **Internal Force Assembly:** Evaluated at $U_{n+1,*}$ with MPI ghost scatters.
3.  **Acceleration Update:** $\ddot{U}_{n+1} = M_L^{-1} f_{int}(U_{n+1,*})$
4.  **Velocity Corrector:** $\dot{U}_{n+1} = \dot{U}_n + \frac{\Delta t}{2}(\ddot{U}_n + \ddot{U}_{n+1})$

---

## 🛠 Critical Numerical Corrections

### 1. Kinematic Consistency (The "Drift" Fix)
In explicit integration over $O(10^5)$ steps, Dirichlet DOFs often accumulate spurious velocity residuals. To prevent the boundary from drifting away from the analytical trajectory, we explicitly overwrite the boundary kinematics after the corrector:

```python
# Overwrite boundary velocities and accelerations analytically
v_new[dofs_bot_z] = vz_bot   # Analytical dz/dt
v_new[dofs_bot_y] = vy_bot
a_new[dofs_bot_z] = az_bot   # Prescribed acceleration (Critical for Energy!)
a_new[dofs_bot_y] = ay_bot
