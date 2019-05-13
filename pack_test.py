"""Transient flow over a backward-facing step. Incompressible Navier-Stokes
equations are solved using Newton/Picard iterative method. Linear solver is
based on field split PCD preconditioning."""

# Begin demo
from dolfin import *
#from matplotlib import pyplot
import numpy as np

from fenapack import PCDKrylovSolver
from fenapack import PCDAssembler
from fenapack import PCDNewtonSolver, PCDNonlinearProblem
from fenapack import StabilizationParameterSD

from mpi4py import MPI as pmp
import argparse, sys, os, gc
import time

commmpi = pmp.COMM_WORLD
# Parse input arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", type=int, dest="level", default=0,
                    help="level of mesh refinement")
parser.add_argument("--nu", type=float, dest="viscosity", default=0.02,
                    help="kinematic viscosity")
#parser.add_argument("--pcd", type=str, dest="pcd_variant", default="BRM1",
#                    choices=["BRM1", "BRM2"], help="PCD variant")
parser.add_argument("--nls", type=str, dest="nls", default="newton",
                    choices=["picard", "newton"], help="nonlinear solver")
parser.add_argument("--ls", type=str, dest="ls", default="iterative",
                    choices=["direct", "iterative"], help="linear solvers")
parser.add_argument("--dm", action='store_true', dest="mumps_debug",
                    help="debug MUMPS")
parser.add_argument("--dt", type=float, dest="dt", default=0.2,
                    help="time step")
parser.add_argument("--t_end", type=float, dest="t_end", default=30.0,
                    help="termination time")
parser.add_argument("--ts_per_out", type=int, dest="ts_per_out", default=1,
                    help="number of ts per output file")
parser.add_argument("--mesh_file", type=str, dest="mesh_file", default="./mesh.xml",
                    help="path and file name of the mesh")
parser.add_argument("--out_folder", type=str, dest="out_folder", default="./result",
                    help="output folder name")
args = parser.parse_args(sys.argv[1:])

class rmtursAssembler(object):
    def __init__(self, a, L, bcs):
        self.assembler = SystemAssembler(J, F, bcs)
        self._bcs = bcs
    def rhs_vector(self, b, x=None):
        if x is not None:
            self.assembler.assemble(b, x)
        else:
            self.assembler.assemble(b)
    def system_matrix(self, A):
        self.assembler.assemble(A)

class rmtursNonlinearProblem(NonlinearProblem):
    def __init__(self, rmturs_assembler):
        #assert isinstance(rmturs_assembler, rmtursAssembler)
        super(rmtursNonlinearProblem, self).__init__()
        self.rmturs_assembler = rmturs_assembler
    def F(self, b, x):
        self.rmturs_assembler.rhs_vector(b, x)
    def J(self, A, x):
        self.rmturs_assembler.system_matrix(A)

class rmtursNewtonSolver(NewtonSolver):
    def __init__(self, solver):
        comm = solver.ksp().comm.tompi4py()
        factory = PETScFactory.instance()
        super(rmtursNewtonSolver, self).__init__(comm, solver, factory)
        self._solver = solver
    def solve(self, problem, x):
        self._problem = problem
        r = super(rmtursNewtonSolver, self).solve(problem, x)
        del self._problem
        return r

parameters["form_compiler"]["quadrature_degree"] = 3
parameters["std_out_all_processes"] = False
rank = commmpi.Get_rank()
root = 0
# Load mesh from file and refine uniformly
try:
    mesh = Mesh(args.mesh_file)
except:
    try:
        mesh = Mesh()
        fid = HDF5File(commmpi, args.mesh_file, 'r')
        fid.read(mesh, 'mesh', False)
        fid.close()
    except:
        info("No valid mesh to read in.")

for i in range(args.level):
    mesh = refine(mesh)

##################################
#### Boundary & design domain ####
##################################
eps = 1e-6
# No-slip bc
class Gamma0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Inlet bc
class Gamma1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0]<eps

# Oultet bc
class Gamma2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0]>4.0-eps)# or x[0]<-1.0+eps)



boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_markers.set_all(4)        # interior facets
Gamma0().mark(boundary_markers, 0) # no-slip facets
Gamma1().mark(boundary_markers, 1) # inlet facets
Gamma2().mark(boundary_markers, 2) # outlet facet
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)


# Build Taylor-Hood function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P3 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P4 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#W = FunctionSpace(mesh, ((P2*P1)*P3)*P4)
W = FunctionSpace(mesh, MixedElement([P2, P1]))
W_turb = FunctionSpace(mesh, MixedElement([P3, P4]))

u0 = 1.0
u_in = Expression(("u0*(x[1])*(1.0-x[1])*(x[2])*(1.0-x[2])*16.0*(1.0 - exp(-0.1*t))","0.0","0.0"),u0=u0,t=0.0,degree=2)
# Navier-stokes bc
bc00 = DirichletBC(W.sub(0), (0.0, 0.0, 0.0), boundary_markers, 0)

bc1 = DirichletBC(W.sub(0), u_in, boundary_markers, 1)
bcu = [bc00, bc1]

# BC for turbulence models k and e
bc_nsk = DirichletBC(W_turb.sub(0), 0.0, boundary_markers, 0)
bc_nse = DirichletBC(W_turb.sub(1), 0.0, boundary_markers, 0)

Cmu = 0.09
turb_intensity = 0.05
turb_lengthscale = 0.038*1.0
#k_in = 1.5*(u_in_xnormal*turb_intensity)**2
#e_in = Cmu*(k_in**1.5)/turb_lengthscale
k_in = Expression("1.5*pow(u0*(x[1])*(1.0-x[1])*(x[2])*(1.0-x[2])*16.0*(1.0 - exp(-0.1*t))*turb_intensity,2)",u0=u0,t=0.0,turb_intensity=turb_intensity,degree=2)
e_in = Expression("Cmu*pow(1.5*pow(u0*(x[1])*(1.0-x[1])*(x[2])*(1.0-x[2])*16.0*(1.0 - exp(-0.1*t))*turb_intensity,2),1.5)/turb_lengthscale",u0=u0,t=0.0,turb_intensity=turb_intensity,turb_lengthscale=turb_lengthscale,Cmu=Cmu,degree=2)
bc_ink = DirichletBC(W_turb.sub(0), k_in, boundary_markers, 1)
bc_ine = DirichletBC(W_turb.sub(1), e_in, boundary_markers, 1)

# k-e BCs in a package
bcke = [bc_nsk, bc_ink, bc_ine, bc_nse]

# Provide some info about the current problem
info("Reynolds number: Re = %g" % (1.0*u0/args.viscosity))
info("Dimension of the function space: %g" % W.dim())
# Arguments and coefficients of the form
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
w = Function(W)
w = interpolate(Expression(("eps","eps","eps","0.0"),eps=1e-10,degree=1),W)
w0 = Function(W)
(u_, p_) = split(w)
(u0_, p0_) = split(w0)

(k, e) = TrialFunctions(W_turb)
(vk, ve) = TestFunctions(W_turb)
k_e = Function(W_turb)
#k_e = interpolate(Expression(("k_in","0.09*pow(k_in,1.5)/0.038"),k_in=1.5*0.05**2,degree=2),W_turb)
k_e = interpolate(Expression(("eps","eps"),eps=1e-9,degree=2),W_turb)
k_e0 = Function(W_turb)
(k_, e_) = split(k_e)
(k0_, e0_) = split(k_e0)

info("Function space constructed")
#u0_, p0_ = w0.split(True) #split using deepcopy
#nu = Constant(args.viscosity)
final_nu = args.viscosity
#nu = final_nu
nu = Expression("nu",nu=1000*final_nu,degree=2,domain=mesh)
idt = Constant(1.0/args.dt)
h = CellDiameter(mesh)
#ramp_time = 6.0/(u0*0.5)

info("Courant number: Co = %g ~ %g" % (u0*args.dt/mesh.hmax(), u0*args.dt/mesh.hmin()))
#w.interpolate(Constant((0.01,0.01,0.0)))
#vnorm = sqrt(dot(u0_,u0_))
#vnorm = u0_.vector().norm("l2")
vnorm = norm(w.sub(0),"l2")
# SUPG & PSPG stabilization parameters
h_vgn = mesh.hmin()
h_rgn = mesh.hmin()
tau_sugn1 = h_vgn/(2.0)
tau_sugn2 = idt/2.0
tau_sugn3 = h_rgn**2/(4.0*args.viscosity)
tau_supg = (1.0/tau_sugn1**2 + 1.0/tau_sugn2**2 + 1.0/tau_sugn3**2)**(-0.5)
tau_pspg = tau_supg
tau_lsic = tau_supg#*vnorm**2

# Nonlinear equation
small_r = 1e-7
nu_t = Cmu*(k_**2)/e_
#nu_t = (Cmu*(k_**2)/e_)/(1.0+small_r*(Cmu*(k_**2)/e_))
sigma_e = 1.3
Ceps = 0.07
#C1 = 1.44
C1 = 0.126
C2 = 1.92
MomEqn = idt*(u_ - u0_) - div(nu*grad(u_)) + grad(u_)*u_ + grad(p_)
F_stab = (tau_supg*inner(grad(v)*u_,MomEqn) + tau_pspg*inner(grad(q),MomEqn) + tau_lsic*div(v)*div(u_))*dx
F = (
      idt*inner(u_ - u0_, v)
    + nu*inner(grad(u_), grad(v))
    + inner(dot(grad(u_), u_), v)
    - (p_)*div(v)
    + q*div(u_)
)*dx
F_k = (idt*(k_ - k0_)*vk + div(k_*u_)*vk + nu_t*dot(grad(k_), grad(vk))\
        - nu_t*(0.5*inner(grad(u_)+grad(u_).T, grad(u_)+grad(u_).T)*vk) + e_*vk)*dx
F_e = (idt*(e_ - e0_)*ve + div(e_*u_)*ve + (Ceps/Cmu)*nu_t*dot(grad(e_), grad(ve))\
        - C1*k_*(0.5*inner(grad(u_)+grad(u_).T, grad(u_)+grad(u_).T)*ve)\
        #+ C2*((e_**2/k_)/(1.0+small_r*(e_**2/k_))*ve))*dx
        + C2*(e_**2/k_)*ve)*dx
F = F + F_stab
F_ke = F_k + F_e
# Jacobian
if args.nls == "picard":
    J = (
          idt*inner(u, v)
        + nu*inner(grad(u), grad(v))
        + inner(dot(grad(u), u_), v)
        - p*div(v)
        + q*div(u)
    )*dx
elif args.nls == "newton":
    J = derivative(F, w)
    J_ke = derivative(F_ke, k_e)
    #J_pc = None #derivative(F+F_stab, w)

# Add stabilization for AMG 00-block
#J_pc = None
"""
if args.ls == "iterative":
    delta = StabilizationParameterSD(w.sub(0), nu)
    J_pc = J + delta*inner(dot(grad(u), u_), dot(grad(v), u_))*dx
elif args.ls == "direct":
    J_pc = None
"""

"""
# PCD operators
mu = idt*inner(u, v)*dx
mp = 1.0/(nu)*p*q*dx
kp = 1.0/(nu)*(dot(grad(p), u_) + idt*p)*q*dx
ap = inner(grad(p), grad(q))*dx
if args.pcd_variant == "BRM2":
    n = FacetNormal(mesh)
    ds = Measure("ds", subdomain_data=boundary_markers)
    # TODO: What about the reaction term? Does it appear here?
    kp -= 1.0/nu*dot(u_, n)*p*q*ds(1)
    #kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(0)  # TODO: Is this beneficial?
"""
# Collect forms to define nonlinear problem
#pcd_assembler = PCDAssembler(J, F, bcu,
#                             J_pc, ap=ap, kp=kp, mp=mp, bcs_pcd=bc_pcd)
#assert pcd_assembler.get_pcd_form("gp").phantom # pressure grad obtained from J
#problem = PCDNonlinearProblem(pcd_assembler)

NS_assembler = rmtursAssembler(J, F, bcu)
problem = rmtursNonlinearProblem(NS_assembler)
 
ke_assembler = rmtursAssembler(J_ke, F_ke, bcke)
ke_problem = rmtursNonlinearProblem(ke_assembler)

# Set up linear solver (GMRES with right preconditioning using Schur fact)
PETScOptions.clear()
linear_solver = PETScKrylovSolver()
linear_solver.parameters["relative_tolerance"] = 1e-4
linear_solver.parameters["absolute_tolerance"] = 1e-6
linear_solver.parameters['error_on_nonconvergence'] = False
PETScOptions.set("ksp_monitor")
if args.ls == "iterative":
    PETScOptions.set("ksp_type", "fgmres")
    PETScOptions.set("ksp_gmres_restart", 30)
    PETScOptions.set("ksp_max_it", 100)
    PETScOptions.set("preconditioner", "jacobi")
    #PETScOptions.set("nonzero_initial_guess", True)
linear_solver.set_from_options()

PETScOptions.clear()
ke_linear_solver = PETScKrylovSolver()
ke_linear_solver.parameters["relative_tolerance"] = 1e-4
ke_linear_solver.parameters["absolute_tolerance"] = 1e-6
ke_linear_solver.parameters['error_on_nonconvergence'] = False
PETScOptions.set("ksp_monitor")
if args.ls == "iterative":
    PETScOptions.set("ksp_type", "fgmres")
    PETScOptions.set("ksp_gmres_restart", 30)
    PETScOptions.set("ksp_max_it", 100)
    PETScOptions.set("preconditioner", "jacobi")
    #PETScOptions.set("nonzero_initial_guess", True)
ke_linear_solver.set_from_options()


# Set up nonlinear solver
solver = rmtursNewtonSolver(linear_solver)
solver.parameters["relative_tolerance"] = 1e-3
solver.parameters["error_on_nonconvergence"] = False
solver.parameters["maximum_iterations"] = 3

# Set up k-e solver
ke_solver = rmtursNewtonSolver(ke_linear_solver)
solver.parameters["relative_tolerance"] = 1e-3
solver.parameters["error_on_nonconvergence"] = False
solver.parameters["maximum_iterations"] = 3

if rank == 0:
    set_log_level(20) #INFO level, no warnings
else:
    set_log_level(50)
# files
#if rank == 0:
ufile = File(args.out_folder+"/velocity.pvd")
pfile = File(args.out_folder+"/pressure.pvd")

# Solve problem
t = 0.0
time_iters = 0
krylov_iters = 0
solution_time = 0.0

while t < args.t_end and not near(t, args.t_end, 0.1*args.dt):
    # Move to current time level
    t += args.dt
    time_iters += 1
    
    # move from ramping to computing
    """
    if t < ramp_time:
        if ramp_time/2.0 - t > 0:
            nu.nu = (np.exp(-t) - np.exp(-ramp_time/2.0))*999.0*args.viscosity + args.viscosity

        else:
            nu.nu = args.viscosity
    """
    nu.nu = final_nu
    info("Viscosity: %g" % nu.nu)
    # Update boundary conditions
    u_in.t = t
    k_in.t = t
    e_in.t = t
    norm_k = norm(k_e.sub(0),'l2')
    norm_e = norm(k_e.sub(1),'l2')
    info("Eddy momt is: %g" %(norm_k))
    info("Eddy ener is: %g" %(norm_e))

    # Solve the nonlinear problem
    info("t = {:g}, step = {:g}, dt = {:g}".format(t, time_iters, args.dt))
    with Timer("Solve") as t_solve:
        newton_iters, converged = solver.solve(problem, w.vector())
        ke_solver.solve(ke_problem, k_e.vector())
    krylov_iters += solver.krylov_iterations()
    solution_time += t_solve.stop()

    if (time_iters % args.ts_per_out==0)or(time_iters == 1):
        u_out, p_out = w.split()
        ufile << u_out
        pfile << p_out

    # Update variables at previous time level
    w0.assign(w)
    k_e0.assign(k_e)


# Report timings
list_timings(TimingClear.clear, [TimingType.wall, TimingType.user])

# Get iteration counts
result = {
    "ndof": W.dim(), "time": solution_time, "steps": time_iters,
    "lin_its": krylov_iters, "lin_its_avg": float(krylov_iters)/time_iters}
tab = "{:^15} | {:^15} | {:^15} | {:^19} | {:^15}\n".format(
    "No. of DOF", "Steps", "Krylov its", "Krylov its (p.t.s.)", "Time (s)")
tab += "{ndof:>9}       | {steps:^15} | {lin_its:^15} | " \
       "{lin_its_avg:^19.1f} | {time:^15.2f}\n".format(**result)
print("\nSummary of iteration counts:")
print(tab)
#with open("table_pcdr_{}.txt".format(args.pcd_variant), "w") as f:
#    f.write(tab)

# Plot solution
#u, p = w.split()
#size = MPI.size(mesh.mpi_comm())
#rank = MPI.rank(mesh.mpi_comm())
"""
pyplot.figure()
pyplot.subplot(2, 1, 1)
plot(u, title="velocity")
pyplot.subplot(2, 1, 2)
plot(p, title="pressure")
pyplot.savefig("figure_v_p_size{}_rank{}.pdf".format(size, rank))
pyplot.figure()
plot(p, title="pressure", mode="warp")
pyplot.savefig("figure_warp_size{}_rank{}.pdf".format(size, rank))
pyplot.show()
"""

