# Thermo–Hyperelastic Tube (FEniCSx)

3-D thermo-mechanical simulation of a cylindrical tube using **FEniCSx**.  
The model couples transient heat conduction with finite-strain hyperelasticity and applies realistic thermal and mechanical boundary conditions.

---

## Model overview

**Physics**

- Transient heat conduction (implicit)
- Convection on the bottom surface with ramped ambient temperature
- Prescribed temperature on the top surface
- Hyperelastic (Neo-Hookean) mechanics with thermal expansion
- Top face fixed, bottom face: ramp + sinusoidal displacement

Coupling is **thermal → mechanical** (temperature drives expansion and stress).

---

## What I extract from the simulation

- 🔹 **Temperature at tube center vs time**
- 🔹 **Von Mises stress at tube center vs time**
- 🔹 **Temperature field at the final time step**

All fields are saved as XDMF and viewed in ParaView.

---

## Quick convergence sanity check

I repeated the simulation on a finer mesh:

- center temperature difference: **< 0.5%**
- center von Mises stress difference: **≈ 1–2%**

Temperature converges faster than stress (expected), and the chosen mesh is acceptable for the study.

---

## How to run

Generate the mesh:

```bash
python make_tube_mesh.py
