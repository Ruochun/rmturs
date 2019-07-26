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
parser.add_argument("--ramp_type", type=str, dest="ramp_type", default="none",
                    choices=["viscosity", "velocity", "none"], help="choose to ramp viscosity or velocity")
parser.add_argument("--ramp_ts", type=int, dest="ramp_ts", default=1000,
                    help="number of ramping time steps")
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
        self.assembler = SystemAssembler(a, L, bcs)
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

if args.ramp_type == "viscosity":
    ramp_token = 1
elif args.ramp_type == "velocity":
    ramp_token = 2
else:
    ramp_token = 0


#commmpi.Barrier()
##################################
#### Boundary & design domain ####
##################################
eps = 1e-7
"""
# The cube in the middle of channel example BCs
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
"""
# bump example BCs
class Gamma0(SubDomain):  #no-slip
    def inside(self, x, on_boundary):
        return on_boundary 

class Gamma1(SubDomain):  #bump
    def inside(self, x, on_boundary):
        return on_boundary and (x[2]>=eps) and (x[2]<=0.2-eps) and (x[1]>0.1) and (x[1]<0.6) and (x[0]>=0.1) and (x[0]<=0.6) 

class Gamma2(SubDomain): #slip
    def inside(self, x, on_boundary):
        return on_boundary and ((x[1]<eps)or(x[1]>0.7-eps))

# Inlet bc
class Gamma3(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0]<eps #and x[1]>=eps and (x[1]<=0.7-eps)

# Oultet bc
class Gamma4(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0]>1.5-eps)

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_markers.set_all(5)        # interior facets
Gamma0().mark(boundary_markers, 0) # side no-slip facets
Gamma1().mark(boundary_markers, 1) # bump facets
Gamma2().mark(boundary_markers, 2) # slip facet
Gamma3().mark(boundary_markers, 3) # inlet facet
Gamma4().mark(boundary_markers, 4) # outlet facet
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)


# Build Taylor-Hood function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#W = FunctionSpace(mesh, ((P2*P1)*P3)*P4)
W = FunctionSpace(mesh, MixedElement([P2, P1]))
#vel_space = FunctionSpace(mesh, P2)

n = FacetNormal(mesh)
flow_direction = Constant((1.0,0.0,0.0))
u0 = 1.0
if ramp_token == 2:
    #ramp_time = 10.0
    ramp_time = args.dt*args.ramp_ts
    #u_in = Expression(("u0*(x[1])*(1.0-x[1])*(x[2])*(1.0-x[2])*16.0*(1.0 - exp(-0.1*t))","0.0","0.0"),u0=u0,t=0.0,degree=2)
    u_in = Expression(("u0*(t/ramp_time)","0.0","0.0"),u0=u0,t=0.0,ramp_time=ramp_time,degree=2)
    nu = Expression("nu",nu=args.viscosity,degree=1,domain=mesh)
elif ramp_token == 1:
    if args.viscosity > 1.0:
        info("Warning: the viscosity is larger than 1, the solver will ramp up the viscosity, instead of ramp down.")
    ramp_time = args.dt*args.ramp_ts
    u_in = Expression(("u0","0.0","0.0"),u0=u0,degree=1)
    nu = Expression("(nu-1.0)/ramp_time*t+1.0",nu=args.viscosity,t=0.0,ramp_time=ramp_time,degree=2,domain=mesh)
else:
    u_in = Expression(("u0","0.0","0.0"),u0=u0,degree=1)
    nu = Expression("nu",nu=args.viscosity,degree=1,domain=mesh)

# Navier-stokes bc
bc00 = DirichletBC(W.sub(0), (0.0, 0.0, 0.0), boundary_markers, 0)
bc01 = DirichletBC(W.sub(0), (0.0, 0.0, 0.0), boundary_markers, 1)
bc_slip = DirichletBC(W.sub(0).sub(1), 0.0, boundary_markers, 2)
bc_in = DirichletBC(W.sub(0), u_in, boundary_markers, 3)
bcu = [bc00, bc01, bc_slip, bc_in]

# Provide some info about the current problem
info("Reynolds number: Re = %g" % (0.1*u0/args.viscosity))
info("Dimension of the function space: %g" % W.dim())
# Arguments and coefficients of the form
init_ufield = 0.1
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
w = Function(W)
w = interpolate(Expression(("eps","eps","eps","0.0"),eps=init_ufield,degree=1),W)
w0 = Function(W)
w0 = interpolate(Expression(("eps","eps","eps","0.0"),eps=init_ufield,degree=1),W)
(u_, p_) = split(w)
(u0_, p0_) = split(w0)

#u0_, p0_ = w0.split(True) #split using deepcopy
idt = Constant(1.0/args.dt)
h = CellDiameter(mesh)

info("Courant number: Co = %g ~ %g" % (u0*args.dt/mesh.hmax(), u0*args.dt/mesh.hmin()))
# vnorm = norm(w.sub(0),"l2")
# SUPG & PSPG stabilization parameters
h_vgn = mesh.hmin()
h_rgn = mesh.hmin()
#u0_norm = sqrt(dot(u0, u0))
u0_norm2 = dot(u0_, u0_)
#tau_sugn1 = h_vgn/(2.0*u0_norm)
tau_sugn1 = h_vgn/(2.0)
tau_sugn2 = idt/2.0
tau_sugn3 = h_rgn**2/(4.0*args.viscosity)
tau_supg = (1.0/tau_sugn1**2 + 1.0/tau_sugn2**2 + 1.0/tau_sugn3**2)**(-0.5)
#tau_supg = 1.0/sqrt(4.0*u0_norm2/h_vgn**2 + 1.0/tau_sugn2**2 + 1.0/tau_sugn3**2)
tau_pspg = tau_supg
tau_lsic = tau_supg*u0_norm2

# Nonlinear equation
MomEqn = idt*(u_ - u0_) - div(nu*grad(u_)) + grad(u_)*u_ + grad(p_)
u_prime = -tau_supg*MomEqn
p_prime = -tau_lsic*div(u_)
F_stab = (tau_supg*inner(grad(v)*u_,MomEqn) + tau_pspg*inner(grad(q),MomEqn) + tau_lsic*div(v)*div(u_))*dx
F = (
      idt*inner(u_ - u0_, v)
    + nu*inner(grad(u_), grad(v))
    + inner(dot(grad(u_), u_), v)
    - (p_)*div(v)
    + q*div(u_)
)*dx
F_VMS = (dot(v, dot(u_prime, grad(u_))) - dot(dot(u_prime, grad(v)), u_prime))*dx
F = F + F_stab + F_VMS

I = Identity(3)
drag = dot(flow_direction, dot(n, p_*I - nu*(grad(u_)+grad(u_).T)))*ds(1)
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

NS_assembler = rmtursAssembler(J, F, bcu)
problem = rmtursNonlinearProblem(NS_assembler)


# Set up linear solver (GMRES with right preconditioning using Schur fact)
PETScOptions.clear()
linear_solver = PETScKrylovSolver()
linear_solver.parameters["relative_tolerance"] = 1e-4
linear_solver.parameters["absolute_tolerance"] = 1e-12
linear_solver.parameters['error_on_nonconvergence'] = False
PETScOptions.set("ksp_monitor")

# Set up subsolvers
if args.ls == "iterative":
    PETScOptions.set("ksp_type", "fgmres")
    PETScOptions.set("ksp_gmres_restart", 10)
    PETScOptions.set("ksp_max_it", 100)
    PETScOptions.set("preconditioner", "default")
    #PETScOptions.set("nonzero_initial_guess", True)


# Apply options
linear_solver.set_from_options()

# Set up nonlinear solver
solver = rmtursNewtonSolver(linear_solver)
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
    
    # nu.nu = args.viscosity
    # info("Viscosity: %g" % nu.nu)
    # Update boundary conditions
    if ramp_token == 2:
        if t<ramp_time:
            u_in.t = t
        else: 
            u_in.t = ramp_time
    elif ramp_token == 1:
        if t<ramp_time:
            nu.t = t
        else:
            nu.t = ramp_time

    # Solve the nonlinear problem
    info("t = {:g}, step = {:g}, dt = {:g}".format(t, time_iters, args.dt))
    with Timer("Solve") as t_solve:
        newton_iters, converged = solver.solve(problem, w.vector())
    krylov_iters += solver.krylov_iterations()
    solution_time += t_solve.stop()

    if (time_iters % args.ts_per_out==0)or(time_iters == 1):
        u_out, p_out = w.split()
        ufile << u_out
        pfile << p_out

    # Update variables at previous time level
    w0.assign(w)

    #k_e0.assign(k_e)
    F_d = assemble(drag)
    info("Drag coef = %g" % (2.0*F_d/0.01))


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
