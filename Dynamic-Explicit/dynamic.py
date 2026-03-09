"""
Explicit central-difference solver — hyperelastic tube.
Neo-Hookean material, prescribed shock kinematics at the bottom face.

Validation checks (undamped, rayleigh_alpha = 0):
  - Energy balance  : E_kin + E_pot - W_ext = const  (< 1 % drift)
  - Momentum balance: ||M_L*a - f_int - R|| / ||M_L*a||  ~ 1e-8

References:
  [1] Dokken, The FEniCSx Tutorial, 2023
  [2] Logg, Mardal, Wells, Automated FEM, Springer 2012
  [3] Carlberg, Tuminaro, Boggs, SIAM J. Sci. Comput. 37(2), 2015
  [4] Bonet & Wood, Nonlinear Continuum Mechanics for FEA, 2nd ed., 2008
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.fem import petsc
import ufl
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

comm = MPI.COMM_WORLD
rank = comm.rank

# ─────────────────────────────────────────────────────────────────────
# 1.  Mesh and facet tags
# ─────────────────────────────────────────────────────────────────────
with io.XDMFFile(comm, "tube.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)

with io.XDMFFile(comm, "tube_facets_linear.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(mesh, name="Grid")

# Integration measure — must be bound to the mesh explicitly
dx = ufl.Measure("dx", domain=mesh)

# ─────────────────────────────────────────────────────────────────────
# 2.  Function space  (P1 vectorial, 3 DOFs per node)
# ─────────────────────────────────────────────────────────────────────
V = fem.functionspace(mesh, ("Lagrange", 1, (3,)))

u   = fem.Function(V, name="u")    # predictor  u^{n+1,*}
u_n = fem.Function(V, name="u_n")  # state at n
v_n = fem.Function(V, name="v_n")
a_n = fem.Function(V, name="a_n")

# ─────────────────────────────────────────────────────────────────────
# 3.  Material constants
# ─────────────────────────────────────────────────────────────────────
E_mod, nu, rho = 5.0e6, 0.49, 1250.0

mu  = fem.Constant(mesh, E_mod / (2.0 * (1.0 + nu)))
lam = fem.Constant(mesh, E_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

# 0.0  →  undamped (energy conservation check)
# > 0  →  physical simulation
rayleigh_alpha = 0.0

# ─────────────────────────────────────────────────────────────────────
# 4.  CFL time step
# ─────────────────────────────────────────────────────────────────────
mesh.topology.create_connectivity(tdim, 0)
coords = mesh.geometry.x
c_to_v = mesh.topology.connectivity(tdim, 0)


h_min_local = 1.0e10
for i in range(mesh.topology.index_map(tdim).size_local):
    verts = c_to_v.links(i)
    if len(verts) < 4:
        continue
    for a, b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
        h_min_local = min(h_min_local,
                          np.linalg.norm(coords[verts[a]] - coords[verts[b]]))

h_min  = comm.allreduce(h_min_local, op=MPI.MIN)
c_mech = np.sqrt((lam.value + 2.0 * mu.value) / rho)

CFL       = 0.3
dt        = CFL * h_min / c_mech
T_final   = 0.05
num_steps = int(np.ceil(T_final / dt))

if rank == 0:
    print(f"h_min  = {h_min:.4e} m  |  c_mech = {c_mech:.2f} m/s")
    print(f"dt     = {dt:.4e} s  (CFL = {CFL})")
    print(f"steps  = {num_steps}")

# ─────────────────────────────────────────────────────────────────────
# 5.  Lumped mass vector
#     M_L_ii = sum_j M_ij   <=>   M_L = M_consistent * 1
# ─────────────────────────────────────────────────────────────────────
m_form = fem.form(
    rho * ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * dx
)
M_mat = petsc.assemble_matrix(m_form)
M_mat.assemble()

m_lumped     = fem.Function(V)
m_lumped_inv = fem.Function(V)
ones = fem.Function(V)
ones.x.array[:] = 1.0

M_mat.mult(ones.x.petsc_vec, m_lumped.x.petsc_vec)
m_lumped_inv.x.array[:] = 1.0 / (m_lumped.x.array + 1e-15)

# ─────────────────────────────────────────────────────────────────────
# 6.  Neo-Hookean strain energy and internal force
#     psi = mu/2*(Ic-3) - mu*ln J + lam/2*(ln J)^2
#     f_int = -d(∫psi dV)/du   via UFL automatic differentiation
# ─────────────────────────────────────────────────────────────────────
I_ufl = ufl.Identity(3)
F_def = I_ufl + ufl.grad(u)
J_det = ufl.det(F_def)
Ic    = ufl.tr(F_def.T * F_def)

psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J_det) + (lam / 2) * (ufl.ln(J_det))**2

f_int_form = fem.form(
    -ufl.derivative(psi * dx, u, ufl.TestFunction(V))
)
f_int_vec  = petsc.create_vector(f_int_form)
E_pot_form = fem.form(psi * dx)

# ─────────────────────────────────────────────────────────────────────
# 7.  Boundary DOF index arrays
#     top (101) : fully clamped   u = 0
#     bot (102) : x pinned,  y and z prescribed
# ─────────────────────────────────────────────────────────────────────
dofs_top = fem.locate_dofs_topological(V, fdim, facet_tags.find(101))
dofs_bot = fem.locate_dofs_topological(V, fdim, facet_tags.find(102))

dofs_top_all = np.concatenate([dofs_top*3, dofs_top*3+1, dofs_top*3+2])
dofs_bot_x   = dofs_bot * 3
dofs_bot_y   = dofs_bot * 3 + 1
dofs_bot_z   = dofs_bot * 3 + 2

all_bc_dofs = np.concatenate([dofs_top_all, dofs_bot_x,
                               dofs_bot_y,  dofs_bot_z])

local_size = V.dofmap.index_map.size_local * 3

# Boolean mask: True on free DOFs, used for momentum check
free_mask = np.ones(local_size, dtype=bool)
free_mask[all_bc_dofs] = False

# ─────────────────────────────────────────────────────────────────────
# 8.  Prescribed kinematics — bottom face
# ─────────────────────────────────────────────────────────────────────
_omega = 2.0 * np.pi * 200.0

def prescribed_bot(t):
    if t < 0.005:
        z, vz, az = -0.015 * (t / 0.005), -0.015 / 0.005, 0.0
        y, vy, ay = 0.0, 0.0, 0.0
    else:
        tau = t - 0.005
        z   = -0.015 - 0.005 * np.sin(_omega * tau)
        vz  = -0.005 * _omega * np.cos(_omega * tau)
        az  =  0.005 * _omega**2 * np.sin(_omega * tau)
        y, vy, ay = 0.5*(0.015+z), 0.5*vz, 0.5*az
    return z, y, vz, vy, az, ay

# ─────────────────────────────────────────────────────────────────────
# 9.  Output and tracking
# ─────────────────────────────────────────────────────────────────────
xdmf_out = io.XDMFFile(comm, "tube_result.xdmf", "w")
xdmf_out.write_mesh(mesh)

W_ext    = 0.0
u_prev_z = np.zeros(len(dofs_bot_z))
u_prev_y = np.zeros(len(dofs_bot_y))

log_t       = []
log_E_kin   = []
log_E_pot   = []
log_W_ext   = []
log_balance = []
log_mom_rel = []

if rank == 0:
    print("\nStarting time loop...")

# ─────────────────────────────────────────────────────────────────────
# 10.  Explicit central-difference loop
# ─────────────────────────────────────────────────────────────────────
for n in range(num_steps):
    t_next = (n + 1) * dt
    z_bot, y_bot, vz_bot, vy_bot, az_bot, ay_bot = prescribed_bot(t_next)

    # ── Predictor ────────────────────────────────────────────────────
    u.x.array[:] = (u_n.x.array
                    + dt        * v_n.x.array
                    + 0.5*dt**2 * a_n.x.array)

    # ── Dirichlet BCs on predictor ───────────────────────────────────
    u.x.array[dofs_top_all] = 0.0
    u.x.array[dofs_bot_x]   = 0.0
    u.x.array[dofs_bot_y]   = y_bot
    u.x.array[dofs_bot_z]   = z_bot
    u.x.scatter_forward()

    # ── Internal force assembly ──────────────────────────────────────
    f_int_vec.set(0.0)
    petsc.assemble_vector(f_int_vec, f_int_form)
    f_int_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
    f_int_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)

    # ── Acceleration  A = M_L^{-1} * f_int - alpha * V_n ─────────────
    a_new = (f_int_vec.array[:local_size] * m_lumped_inv.x.array[:local_size]
             - rayleigh_alpha * v_n.x.array[:local_size])

    # ── Velocity corrector  V = V_n + 0.5*dt*(A_n + A) ───────────────
    v_new = (v_n.x.array[:local_size]
             + 0.5 * dt * (a_n.x.array[:local_size] + a_new))

    # ── Kinematic correction at BC DOFs ──────────────────────────────
    v_new[dofs_top_all] = 0.0;  a_new[dofs_top_all] = 0.0
    v_new[dofs_bot_x]   = 0.0;  a_new[dofs_bot_x]   = 0.0
    v_new[dofs_bot_y]   = vy_bot; a_new[dofs_bot_y]  = ay_bot
    v_new[dofs_bot_z]   = vz_bot; a_new[dofs_bot_z]  = az_bot

    # ── Validation checks every 1 step ────────────────────────────
    if n % 1 == 0:

        # Kinetic energy
        E_k = 0.5 * comm.allreduce(
            np.dot(v_new, m_lumped.x.array[:local_size] * v_new),
            op=MPI.SUM)

        # Potential energy
        E_p = comm.allreduce(
            fem.assemble_scalar(E_pot_form), op=MPI.SUM)

        # Reaction force at moving BCs: R = M_L * a_bc - f_int
        R_z = (m_lumped.x.array[dofs_bot_z] * az_bot
               - f_int_vec.array[dofs_bot_z])
        R_y = (m_lumped.x.array[dofs_bot_y] * ay_bot
               - f_int_vec.array[dofs_bot_y])

        # External work increment: dW = R . du
        dW = comm.allreduce(
            np.dot(R_z, u.x.array[dofs_bot_z] - u_prev_z)
          + np.dot(R_y, u.x.array[dofs_bot_y] - u_prev_y),
            op=MPI.SUM)
        W_ext += dW

        balance = E_k + E_p - W_ext

        # Momentum residual on free DOFs: r = M_L*a - f_int - R
        R_vec = np.zeros(local_size)
        R_vec[dofs_bot_z]   =  R_z
        R_vec[dofs_bot_y]   =  R_y
        R_vec[dofs_bot_x]   = -f_int_vec.array[dofs_bot_x]
        R_vec[dofs_top_all] = -f_int_vec.array[dofs_top_all]

        residual = (m_lumped.x.array[:local_size] * a_new
                    - f_int_vec.array[:local_size]
                    - R_vec)

        mom_rel = comm.allreduce(np.linalg.norm(residual[free_mask]),
                                 op=MPI.SUM) / (
                  comm.allreduce(np.linalg.norm(
                      m_lumped.x.array[:local_size][free_mask] * a_new[free_mask]),
                      op=MPI.SUM) + 1e-15)


        if rank == 0:
            print(f"  t={t_next:.5f}s | "
                  f"E_kin={E_k:.3e} | E_pot={E_p:.3e} | "
                  f"W_ext={W_ext:.3e} | balance={balance:.3e} | "
                  f"mom_rel={mom_rel:.2e}")
        
    # W_ext accumulation needs u_prev updated every step
    u_prev_z = u.x.array[dofs_bot_z].copy()
    u_prev_y = u.x.array[dofs_bot_y].copy()

    # XDMF snapshot every 500 steps
    if n % 500 == 0:
        xdmf_out.write_function(u, t_next)

    # State update
    u_n.x.array[:local_size] = u.x.array[:local_size]
    v_n.x.array[:local_size] = v_new
    a_n.x.array[:local_size] = a_new

xdmf_out.close()

# ─────────────────────────────────────────────────────────────────────
# 11.  Final summary
# ─────────────────────────────────────────────────────────────────────
if rank == 0:
    drift     = abs(log_balance[-1] - log_balance[0])
    rel_drift = drift / (abs(log_W_ext[-1]) + 1e-15) * 100
    mom_max   = float(np.max(np.abs(log_mom_rel)))

    print("\n── Energy balance ───────────────────────────────────")
    print(f"   Drift : {drift:.4e} J  ({rel_drift:.4f} %)")
    print("   ✅ OK" if rel_drift < 1.0 else "   ⚠️  > 1 %")

    print("\n── Momentum residual ────────────────────────────────")
    print(f"   Max   : {mom_max:.4e}")
    print("   ✅ OK" if mom_max < 1e-6 else "   ⚠️  > 1e-6")

    np.save("log_energy.npy",   np.column_stack([
        log_t, log_E_kin, log_E_pot, log_W_ext, log_balance]))
    np.save("log_momentum.npy", np.column_stack([log_t, log_mom_rel]))
    print("\n   tube_result.xdmf   →  displacement snapshots")
    print("   validation_log.csv →  energy + momentum log")