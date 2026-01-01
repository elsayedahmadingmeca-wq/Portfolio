# coupling_dynamic_disp.py — 3-D thermo-hyperelastic tube (sinusoidal displacement)
# FEniCSx 0.9

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl, numpy as np
from basix.ufl import element as basix_element


# ---------------------------------------------------------------------
# MPI communicator (allows parallel run)
# ---------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.rank
if rank == 0:
    print("Running coupled thermo-hyperelastic tube")

# ---------------------------------------------------------------------
# Load mesh + boundary tags (generated previously in make_tube_mesh.py)
# ---------------------------------------------------------------------
with io.XDMFFile(comm, "tube.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

# Needed so boundary facets are available for BCs
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

with io.XDMFFile(comm, "tube_facets_linear.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(mesh, name="Grid")

# Geometric helpers
dim = mesh.geometry.dim
I = ufl.Identity(dim)
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# Boundary identifiers from gmsh
TOP, BOTTOM, LATERAL = 101, 102, 103

# ---------------------------------------------------------------------
# Material parameters (compressible Neo-Hookean + heat conduction)
# ---------------------------------------------------------------------
E = 10.0e6              # Young's modulus [Pa]
nu = 0.49               # nearly incompressible
alpha_th = 1.2e-4       # thermal expansion [1/K]
rho, cp, k = 1250.0, 1500.0, 200.0   # density, heat capacity, conductivity

# Convert to Lame parameters
mu = E/(2*(1+nu))
lam = E*nu/((1+nu)*(1-2*nu))

# Store as FEniCS Constants
mu_c     = fem.Constant(mesh, mu)
lam_c    = fem.Constant(mesh, lam)
alpha_c  = fem.Constant(mesh, alpha_th)
rho_cp_c = fem.Constant(mesh, rho*cp)
k_c      = fem.Constant(mesh, k)

# ---------------------------------------------------------------------
# Thermal boundary data (time-varying)
# ---------------------------------------------------------------------
T_amb_start = 60 + 273.15          # initial ambient temperature
T_amb_end   = 90 + 273.15          # ambient after ramp
T_amb_c     = fem.Constant(mesh, T_amb_start)

h_conv = 100.0                     # convection coefficient [W/m2/K]

# ---------------------------------------------------------------------
# Mechanical loading — ramp + sinusoid on bottom surface
# ---------------------------------------------------------------------
A_disp   = 0.005      # oscillation amplitude
A_disp_0 = 0.04       # preload amplitude
f_disp   = 2.0        # frequency [Hz]
omega    = 2*np.pi*f_disp

# ---------------------------------------------------------------------
# Time discretisation
# ---------------------------------------------------------------------
dt_value = 0.1
dt = fem.Constant(mesh, dt_value)

# simulate ~5 oscillation periods + ramp buffer
t_final = 1/f_disp*5 + 8*dt_value
num_steps = int(np.ceil(t_final/dt_value))
num_steps_ramp = 8                  # number of ramp steps

if rank == 0:
    print(f"dt={dt_value}, steps={num_steps}, t_final={t_final}")

# Temperature prescribed at the top boundary
T_top_start = 50 + 273.15
T_top_end   = 60 + 273.15

# ---------------------------------------------------------------------
# Function spaces (P1 for both)
# ---------------------------------------------------------------------
cell = mesh.topology.cell_type.name
T_el = basix_element("Lagrange", cell, 1)
U_el = basix_element("Lagrange", cell, 1, shape=(dim,))

V_T = fem.functionspace(mesh, T_el)
V_U = fem.functionspace(mesh, U_el)

# Unknown fields
u = fem.Function(V_U, name="u")
T = fem.Function(V_T, name="T")

# Time stepping temperature
T_old = fem.Function(V_T)
T_old.x.array[:] = T_amb_start
T.x.array[:] = T_old.x.array

# ---------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------
# --- mechanical ---
u_zero = fem.Function(V_U)                 # top fixed
dofs_top = fem.locate_dofs_topological(V_U, facet_tags.dim, facet_tags.find(TOP))

u_bottom = fem.Function(V_U)               # bottom prescribed motion
dofs_bottom = fem.locate_dofs_topological(V_U, facet_tags.dim, facet_tags.find(BOTTOM))

bcs_U = [fem.dirichletbc(u_zero, dofs_top),
         fem.dirichletbc(u_bottom, dofs_bottom)]

# --- thermal ---
T_top = fem.Function(V_T)
T_top.x.array[:] = T_top_start
dofs_Ttop = fem.locate_dofs_topological(V_T, facet_tags.dim, facet_tags.find(TOP))
bcs_T = [fem.dirichletbc(T_top, dofs_Ttop)]

# ---------------------------------------------------------------------
# Weak form — transient heat equation (Backward Euler)
# ---------------------------------------------------------------------
T_trial = ufl.TrialFunction(V_T)
T_test  = ufl.TestFunction(V_T)

a_T = ((rho_cp_c/dt)*T_trial*T_test*dx
       + ufl.inner(k_c*ufl.grad(T_trial), ufl.grad(T_test))*dx
       + h_conv*T_trial*T_test*ds(BOTTOM))

L_T = (rho_cp_c/dt)*T_old*T_test*dx + h_conv*T_amb_c*T_test*ds(BOTTOM)

# Solver: CG + Hypre preconditioner
solver_T = PETSc.KSP().create(comm)
solver_T.setType("cg")
solver_T.getPC().setType("hypre")

# ---------------------------------------------------------------------
# Mechanical problem — hyperelastic with thermal expansion
# ---------------------------------------------------------------------
w  = ufl.TestFunction(V_U)
du = ufl.TrialFunction(V_U)

F = I + ufl.grad(u)
T_ref = fem.Constant(mesh, T_amb_start)

# multiplicative thermal decomposition
Fth_inv = (1.0/(1.0 + alpha_c*(T - T_ref))) * I
Fe = F * Fth_inv
Be = Fe*Fe.T
Je = ufl.det(Fe)

# Neo-Hookean strain energy
psi = (mu_c/2)*(ufl.tr(Be) - 3) - mu_c*ufl.ln(Je) + (lam_c/2)*(ufl.ln(Je))**2

# residual & Jacobian
R_u = ufl.derivative(psi*dx, u, w)
J_u = ufl.derivative(R_u, u, du)

problem_u = NonlinearProblem(R_u, u, bcs=bcs_U, J=J_u)
solver_U = NewtonSolver(comm, problem_u)
solver_U.line_search = "bt"
solver_U.max_it = 40

# ---------------------------------------------------------------------
# Helper for applying displacement only in z direction
# ---------------------------------------------------------------------
def fill_bottom_disp(val):
    def f(x):
        return np.vstack((0*x[0], 0*x[0], val*np.ones_like(x[0])))
    return f

# ---------------------------------------------------------------------
# Output file and projection spaces
# ---------------------------------------------------------------------
with io.XDMFFile(comm, "tube.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)

    V_tensor = fem.functionspace(mesh, ("DG", 0, (dim, dim)))
    V_scalar = fem.functionspace(mesh, ("DG", 0))

    stress_out = fem.Function(V_tensor, name="stress")
    vonmises_out = fem.Function(V_scalar, name="vonMises")

    # generic projection routine
    def project(expr, V, target):
        utrial, vtest = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = ufl.inner(utrial, vtest)*dx
        L = ufl.inner(expr, vtest)*dx
        fem.petsc.LinearProblem(a, L, u=target).solve()

    # -----------------------------------------------------------------
    # TIME LOOP
    # -----------------------------------------------------------------
    for n in range(num_steps):

        t = n*dt_value

        # --- update thermal BCs (ramp in first num_steps_ramp) ---
        if n < num_steps_ramp:
            Tamb_now = T_amb_start + n/num_steps_ramp*(T_amb_end - T_amb_start)
            T_amb_c.value = Tamb_now
            Ttop_now = T_top_start + n/num_steps_ramp*(T_top_end - T_top_start)

        T_top.x.array[:] = Ttop_now

        # --- update mechanical displacement ---
        ubot = -A_disp_0*(n/num_steps_ramp)
        if n >= num_steps_ramp:
            ubot = -A_disp_0 - A_disp*np.sin(omega*(t-num_steps_ramp*dt_value))

        u_bottom.interpolate(fill_bottom_disp(ubot))

        # --- solve heat equation ---
        A = fem.petsc.assemble_matrix(fem.form(a_T), bcs=bcs_T); A.assemble()
        b = fem.petsc.assemble_vector(fem.form(L_T))
        fem.petsc.apply_lifting(b, [fem.form(a_T)], [bcs_T])
        fem.petsc.set_bc(b, bcs_T)
        solver_T.setOperators(A)
        solver_T.solve(b, T.x.petsc_vec)

        # --- solve mechanical equilibrium ---
        solver_U.solve(u)

        # --- compute stresses ---
        F = I + ufl.grad(u)
        Fth_inv = (1.0/(1.0 + alpha_c*(T - T_ref))) * I
        Fe = F*Fth_inv
        Be = Fe*Fe.T
        Je = ufl.det(Fe)

        sigma = (mu_c/Je)*(Be - I) + (lam_c*ufl.ln(Je)/Je)*I
        dev = sigma - (ufl.tr(sigma)/3)*I
        von = ufl.sqrt(1.5*ufl.inner(dev, dev))

        project(sigma, V_tensor, stress_out)
        project(von, V_scalar, vonmises_out)

        # save every step
        xdmf.write_function(T, t)
        xdmf.write_function(u, t)
        xdmf.write_function(stress_out, t)
        xdmf.write_function(vonmises_out, t)

        # advance in time
        T_old.x.array[:] = T.x.array


if rank == 0:
    print("Simulation finished.")
