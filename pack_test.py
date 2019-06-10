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

def getDistance(W, markers):
    w = Function(W)
    v = TestFunction(W)
    u = TrialFunction(W)
    bc0 = DirichletBC(W, 0.0, markers, 0)
    bcu = [bc0]
    f = Constant(1.0)
    F = inner(grad(u), grad(v))*dx - f*v*dx
    a, L = lhs(F), rhs(F)
    solve(a == L, w, bcs=bcu)
    return w


parameters["form_compiler"]["quadrature_degree"] = 2
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
        return on_boundary and x[0]<-1.0+eps

# Oultet bc
class Gamma2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0]>5.0-eps)# or x[0]<-1.0+eps)



boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_markers.set_all(4)        # interior facets
Gamma0().mark(boundary_markers, 0) # no-slip facets
Gamma1().mark(boundary_markers, 1) # inlet facets
Gamma2().mark(boundary_markers, 2) # outlet facet
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

#distance_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1) # the marker for distance computation
#distance_markers.set_all(4)
#Gamma0().mark(distance_markers, 0)

# Build Taylor-Hood function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P3 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P4 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#W = FunctionSpace(mesh, ((P2*P1)*P3)*P4)
W = FunctionSpace(mesh, MixedElement([P2, P1]))
W_vel = FunctionSpace(mesh, P2)
W_turb = FunctionSpace(mesh, MixedElement([P3, P4]))
W_scalar = FunctionSpace(mesh, P3)

ramp_time = 3.0
u0 = 1.0
u_in = Expression(("u0*(x[1])*(1.0-x[1])*4.0*(t/ramp_time)","0.0"),u0=u0,t=0.0,ramp_time=ramp_time,degree=2)
# Navier-stokes bc
bc00 = DirichletBC(W.sub(0), (0.0, 0.0), boundary_markers, 0)

bc1 = DirichletBC(W.sub(0), u_in, boundary_markers, 1)
bcu = [bc00, bc1]

# BC for turbulence models k and e
bc_nsk = DirichletBC(W_scalar, 0.0, boundary_markers, 0)
bc_nse = DirichletBC(W_scalar, 0.0, boundary_markers, 0)

Cmu = 0.09
turb_intensity = 0.05
turb_lengthscale = 0.038*1.0
k_in = Expression("1.5*pow(u0*(t/ramp_time)*turb_intensity,2)",u0=u0,t=0.0,ramp_time=ramp_time,turb_intensity=turb_intensity,degree=1)
e_in = Expression("Cmu*pow(1.5*pow(u0*(t/ramp_time)*turb_intensity,2),1.5)/turb_lengthscale",u0=u0,t=0.0,ramp_time=ramp_time,turb_intensity=turb_intensity,turb_lengthscale=turb_lengthscale,Cmu=Cmu,degree=1)
#k_in = Expression("1.5*pow(u0*(x[1])*(1.0-x[1])*4.0*(t/ramp_time)*turb_intensity,2)",u0=u0,t=0.0,ramp_time=ramp_time,turb_intensity=turb_intensity,degree=2)
#e_in = Expression("Cmu*pow(1.5*pow(u0*(x[1])*(1.0-x[1])*4.0*(t/ramp_time)*turb_intensity,2),1.5)/turb_lengthscale",u0=u0,t=0.0,ramp_time=ramp_time,turb_intensity=turb_intensity,turb_lengthscale=turb_lengthscale,Cmu=Cmu,degree=2)
bc_ink = DirichletBC(W_scalar, k_in, boundary_markers, 1)
bc_ine = DirichletBC(W_scalar, e_in, boundary_markers, 1)

# k-e BCs in a package
bcke = [bc_nsk, bc_ink, bc_ine, bc_nse]
#bcke = [bc_nsk, bc_ink, bc_ine]
#bcke = [bc_ine, bc_ink]
#bcke = [bc_nsk, bc_nse]
bck = [bc_nsk, bc_ink]
bce = [bc_nse, bc_ine]
bcL = []

# Provide some info about the current problem
info("Reynolds number: Re = %g" % (1.0*u0/args.viscosity))
info("Dimension of the function space: %g" % W.dim())
# Arguments and coefficients of the form
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
w = Function(W)
w = interpolate(Expression(("eps","eps","0.0"),eps=1e-10,degree=1),W)
w0 = Function(W)
#w0 = interpolate(Expression(("eps","eps","eps","0.0"),eps=1e-10,degree=1),W)
(u_, p_) = split(w)
(u0_, p0_) = split(w0)
L = TrialFunction(W_vel)
L_ = Function(W_vel)
vL = TestFunction(W_vel)

ke_ini = 1e-2
k = TrialFunction(W_scalar)
e = TrialFunction(W_scalar)
vk = TestFunction(W_scalar)
ve = TestFunction(W_scalar)
k_ = Function(W_scalar)
e_ = Function(W_scalar)
k0_ = Function(W_scalar)
e0_ = Function(W_scalar)
k_ = interpolate(Expression("eps",eps=ke_ini,degree=1),W_scalar)
k0_ = interpolate(Expression("eps",eps=ke_ini,degree=1),W_scalar)
e_ = interpolate(Expression("eps",eps=ke_ini,degree=1),W_scalar)
e0_ = interpolate(Expression("eps",eps=ke_ini,degree=1),W_scalar)

#dist2bnd = Function(W_scalar)
dist2bnd = getDistance(W_scalar, boundary_markers)
n = FacetNormal(mesh)
tangent = as_vector([n[1], -n[0]])

info("Function space constructed")
#u0_, p0_ = w0.split(True) #split using deepcopy
#nu = Constant(args.viscosity)
#final_nu = args.viscosity
#nu = final_nu
nu = Expression("final_nu*(1.0 + exp(-1.0*(t-ramp_time)))",ramp_time=ramp_time/2.0,final_nu=args.viscosity,t=99999.9,degree=2,domain=mesh)
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
tau_sugn3 = h_rgn**2/(4.0*nu)
tau_supg = (1.0/tau_sugn1**2 + 1.0/tau_sugn2**2 + 1.0/tau_sugn3**2)**(-0.5)
tau_pspg = tau_supg
tau_lsic = tau_supg#*vnorm**2

# Nonlinear equation
omega_k = 0.5
omega_e = 0.5
ReT = k_**2/(nu*e_)
f_mu = exp(-3.4/(1.0+ReT/50.0)**2)
f_2 = 1.0 - 0.3*exp(-(ReT**2))

#Cmu = Cmu #* f_mu
nu_t = f_mu*Cmu*(k_**2)/e_
#nu_t = Cmu*k0_
#nu_t = conditional( lt(abs(e0_), deno_tol), project(Constant(0.0), W_scalar), Cmu*(k0_**2)/e0_)
#nu_t = (Cmu*(k0_**2)/e0_)/(1.0+small_r*(Cmu*(k0_**2)/e0_))
sigma_k = 1.0
sigma_e = 1.3
#Ceps = 0.07
C1 = 1.44
#C1 = 0.126 #* f_1
C2 = 1.92 #* f_2
MomEqn = idt*(u_ - u0_) - div((nu+nu_t)*(grad(u_)+grad(u_).T)) + grad(u_)*u_ + grad(p_+2.0/3.0*k_)
MomEqn_base = idt*(u_ - u0_) - div(nu*(grad(u_)+grad(u_).T)) + grad(u_)*u_ + grad(p_)
F_stab = (tau_supg*inner(grad(v)*u_,MomEqn) + tau_pspg*inner(grad(q),MomEqn) + tau_lsic*div(v)*div(u_))*dx
F_stab_base = (tau_supg*inner(grad(v)*u_,MomEqn_base) + tau_pspg*inner(grad(q),MomEqn_base) + tau_lsic*div(v)*div(u_))*dx
F = (
      idt*inner(u_ - u0_, v)
    #+ (nu+nu_t)*inner(grad(u_), grad(v)) + inner(outer(grad(nu_t), v), grad(u_)) + dot(v, grad(u_)*grad(nu_t))
    + inner(grad(v), (nu+nu_t)*(grad(u_)+grad(u_).T))
    #+ inner(grad(v), nu*(grad(u_)+grad(u_).T))
    + inner(dot(grad(u_), u_), v)
    - (p_ + 2.0/3.0*k_)*div(v)
    #- p_*div(v)
    + q*div(u_)
)*dx
F_base = (
      idt*inner(u_ - u0_, v)
    #+ (nu+nu_t)*inner(grad(u_), grad(v)) + inner(outer(grad(nu_t), v), grad(u_)) + dot(v, grad(u_)*grad(nu_t))
    #+ inner(grad(v), (nu+nu_t)*(grad(u_)+grad(u_).T))
    + inner(grad(v), nu*(grad(u_)+grad(u_).T))
    + inner(dot(grad(u_), u_), v)
    #- (p_ + 2.0/3.0*k_)*div(v)
    - p_*div(v)
    + q*div(u_)
)*dx
F_k = (\
	idt*(k - k_)*vk \
	+ 0.5*dot(u_, grad(k))*vk\
    + 0.5*dot(u_, grad(k_))*vk\
    - 0.5*(nu+nu_t/sigma_k)*dot(grad(k), grad(vk))\
    - 0.5*(nu+nu_t/sigma_k)*dot(grad(k_), grad(vk))\
    - p_*vk\
    + e_*k/k_*vk\
    - 2.0*nu*(grad(k_**0.5))**2*vk\
    )*dx
F_e = (\
	idt*(e - e_)*ve \
	+ 0.5*dot(u_, grad(e))*ve\
    + 0.5*dot(u_, grad(e_))*ve\
    - 0.5*(nu+nu_t/sigma_e)*dot(grad(e), grad(ve))\
    - 0.5*(nu+nu_t/sigma_e)*dot(grad(e_), grad(ve))\
    - (C1*p_ - f_2*C2*e)*e_/k_*ve\
    - 2*nu*nu_t*L_**2*ve\
    )*dx
F_L = dot(L, vL)*dx + inner(grad(u_), grad(vL))*dx
F = F + F_stab
F_base = F_base + F_stab_base
#F_ke = F_k + F_e
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
    J_base = derivative(F_base, w)
    a_k, L_k = lhs(F_k), rhs(F_k)
    a_e, L_e = lhs(F_e), rhs(F_e)
    a_L, L_L = lhs(F_L), rhs(F_L)
    #J_ke = lhs(F_ke)
    #F_ke = rhs(F_ke)

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

NS_assembler = rmtursAssembler(J_base, F_base, bcu)
problem = rmtursNonlinearProblem(NS_assembler)
 
# Set up linear solver (GMRES with right preconditioning using Schur fact)
PETScOptions.clear()
linear_solver = PETScKrylovSolver()
linear_solver.parameters["relative_tolerance"] = 1e-4
linear_solver.parameters["absolute_tolerance"] = 1e-10
linear_solver.parameters['error_on_nonconvergence'] = False
PETScOptions.set("ksp_monitor")
if args.ls == "iterative":
    PETScOptions.set("ksp_type", "fgmres")
    PETScOptions.set("ksp_gmres_restart", 10)
    PETScOptions.set("ksp_max_it", 50)
    PETScOptions.set("preconditioner", "default")
    #PETScOptions.set("nonzero_initial_guess", True)
linear_solver.set_from_options()

# Set up nonlinear solver
solver = rmtursNewtonSolver(linear_solver)
solver.parameters["relative_tolerance"] = 1e-3
solver.parameters["error_on_nonconvergence"] = False
solver.parameters["maximum_iterations"] = 5

if rank == 0:
    set_log_level(20) #INFO level, no warnings
else:
    set_log_level(50)
# files
#if rank == 0:
ufile = File(args.out_folder+"/velocity.pvd")
pfile = File(args.out_folder+"/pressure.pvd")
kfile = File(args.out_folder+"/k.pvd")
efile = File(args.out_folder+"/epsilon.pvd")
Lfile = File(args.out_folder+"/L.pvd")
# Solve problem
t = 0.0
time_iters = 0
krylov_iters = 0
solution_time = 0.0
NS_changed = False
NS_change_time = 5.0

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
    # Update viscosity
    #nu.t = t
    #info("Viscosity: %g" %(args.viscosity*(1.0 + np.exp(-1.0*(t-ramp_time/2.0)))) )
    # Update boundary conditions
    if t<=ramp_time:
        u_in.t = t
        k_in.t = t
        e_in.t = t
    else:
        u_in.t = ramp_time
        k_in.t = ramp_time
        e_in.t = ramp_time
    if (not(NS_changed))and(t>=NS_change_time):
        NS_assembler = rmtursAssembler(J, F, bcu)
        problem = rmtursNonlinearProblem(NS_assembler)
        NS_changed = True
    
    #norm_k = norm(k_,'l2')
    #norm_e = norm(e_,'l2')
    #info("Eddy momentum is: %g" %(norm_k))
    #info("Eddy energy is: %g" %(norm_e))
    #info("Eddy viscosity is: %g" %(norm_nut))

    # Solve the nonlinear problem
    info("t = {:g}, step = {:g}, dt = {:g}".format(t, time_iters, args.dt))
    with Timer("Solve") as t_solve:
        info("Solving N-S problem:")
        newton_iters, converged = solver.solve(problem, w.vector())
        L_problem = LinearVariationalProblem(a_L, L_L, L_, bcL)
        L_solver = LinearVariationalSolver(L_problem)
        L_solver.solve()
        if NS_changed:
            info("Solving k-e problem:")
            for j in range(2):
                k_vec = k_.vector()[:]
                e_vec = e_.vector()[:]
                np.place(e_vec, e_vec<1e-50*args.viscosity, 1e-50*args.viscosity)
                e_.vector()[:] = e_vec

                k_problem = LinearVariationalProblem(a_k, L_k, k_, bck)
                k_solver = LinearVariationalSolver(k_problem)
                k_solver.solve()
                k_.assign(k_*omega_k + k0_*(1.0-omega_k))
                e_problem = LinearVariationalProblem(a_e, L_e, e_, bce)
                e_solver = LinearVariationalSolver(e_problem)
                e_solver.solve()
                e_.assign(e_*omega_e + e0_*(1.0-omega_e))
                k0_.assign(k_)
                e0_.assign(e_)
            
    krylov_iters += solver.krylov_iterations()
    solution_time += t_solve.stop()

    if (time_iters % args.ts_per_out==0)or(time_iters == 1):
        u_out, p_out = w.split()
        #k_out, e_out = k_e.split()
        ufile << u_out
        pfile << p_out
        kfile << k_
        efile << e_
        Lfile << L_

    # Update variables at previous time level
    w0.assign(w)
    #k0_.assign(k_e)
    #nu_t = conditional( gt(abs(nu_t),1e100*args.viscosity), project(Constant(1e100*args.viscosity),W_scalar), nu_t)
    #e_.assign(Cmu*k_**2/nu_t)
    k_vec = k_.vector()[:]
    e_vec = e_.vector()[:]
    #nu_vec = np.divide(Cmu*np.square(k_vec), e_vec)
    #np.place(nu_vec, nu_vec>1e100*args.viscosity, 1e100*args.viscosity)
    np.place(e_vec, e_vec<1e-50*args.viscosity, 1e-50*args.viscosity)
    #e_vec = np.divide(Cmu*np.square(k_vec), nu_vec)
    e_.vector()[:] = e_vec
    
    


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

