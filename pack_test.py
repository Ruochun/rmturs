"""Transient flow over a backward-facing step. Incompressible Navier-Stokes
equations are solved using Newton/Picard iterative method. Linear solver is
based on field split PCDR preconditioning."""

# Copyright (C) 2015-2018 Martin Rehor, Jan Blechta
#
# This file is part of FENaPack.
#
# FENaPack is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FENaPack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FENaPack.  If not, see <http://www.gnu.org/licenses/>.

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

commmpi = pmp.COMM_WORLD
# Parse input arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", type=int, dest="level", default=4,
                    help="level of mesh refinement")
parser.add_argument("--nu", type=float, dest="viscosity", default=0.02,
                    help="kinematic viscosity")
parser.add_argument("--pcd", type=str, dest="pcd_variant", default="BRM1",
                    choices=["BRM1", "BRM2"], help="PCD variant")
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
args = parser.parse_args(sys.argv[1:])

parameters["form_compiler"]["quadrature_degree"] = 3
parameters["std_out_all_processes"] = False
# Load mesh from file and refine uniformly
#mesh = Mesh("mesh_lshape.xml")
mesh = Mesh()
fid = HDF5File(commmpi, 'mesh.h5', 'r')
fid.read(mesh, 'mesh', False)
fid.close()
for i in range(args.level):
    mesh = refine(mesh)

# Define and mark boundaries
class Gamma0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
class Gamma1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -1.0)
class Gamma2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 5.0)
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_markers.set_all(3)        # interior facets
Gamma0().mark(boundary_markers, 0) # no-slip facets
Gamma1().mark(boundary_markers, 1) # inlet facets
Gamma2().mark(boundary_markers, 2) # outlet facets

# Build Taylor-Hood function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P2*P1)

# No-slip BC
bc0 = DirichletBC(W.sub(0), (0.0, 0.0), boundary_markers, 0)

# Parabolic inflow BC
inlet_velocity = 1.0
inflow = Expression(("4.0*inlet*(1.0 - exp(-5.0*t))*x[1]*(1.0 - x[1])", "0.0"),
                     inlet=inlet_velocity, t=1.0e6, degree=2)
bc1 = DirichletBC(W.sub(0), inflow, boundary_markers, 1)

# Artificial BC for PCD preconditioner
if args.pcd_variant == "BRM1":
    bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 1)
elif args.pcd_variant == "BRM2":
    bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 2)

# Provide some info about the current problem
info("Reynolds number: Re = %g" % (2.0*inlet_velocity/args.viscosity))
info("Dimension of the function space: %g" % W.dim())
# Arguments and coefficients of the form
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
w = Function(W)
w0 = Function(W)
u_, p_ = split(w)
u0_, p0_ = split(w0)
#u0_, p0_ = w0.split(True) #split using deepcopy
#nu = Constant(args.viscosity)
final_nu = args.viscosity
#nu = final_nu
nu = Expression("nu_t",nu_t=final_nu,degree=2,domain=mesh)
idt = Constant(1.0/args.dt)
h = CellDiameter(mesh)
ramp_time = 6.0/(inlet_velocity*0.5)

info("Courant number: Co = %g" % (inlet_velocity*args.dt/h))
#w.interpolate(Constant((0.01,0.01,0.0)))
#vnorm = sqrt(dot(u0_,u0_))
#vnorm = u0_.vector().norm("l2")
vnorm = norm(w.sub(0),"l2")
# SUPG & PSPG stabilization parameters
#tau_supg = h/(2.0*vnorm)
#tau_pspg = h/(2.0*vnorm)
#tau_lsic = vnorm*h/2.0
tau_supg = h/2.0
tau_pspg = h/2.0
tau_lsic = h/2.0

# Nonlinear equation
MomEqn = idt*(u_ - u0_) - div(nu*grad(u_)) + grad(u_)*u_ + grad(p_)
F_stab = (tau_supg*inner(grad(v)*u_,MomEqn) + tau_pspg*inner(grad(q),MomEqn) + tau_lsic*div(v)*div(u_))*dx
F = (
      idt*inner(u_ - u0_, v)
    + nu*inner(grad(u_), grad(v))
    + inner(dot(grad(u_), u_), v)
    - p_*div(v)
    + q*div(u_)
    #+ (tau_supg*inner(grad(v)*u_,MomEqn))
    #+ (tau_pspg*inner(grad(q),MomEqn))
)*dx
F = F + F_stab
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
    J_pc = None #derivative(F+F_stab, w)

# Add stabilization for AMG 00-block
#J_pc = None
"""
if args.ls == "iterative":
    delta = StabilizationParameterSD(w.sub(0), nu)
    J_pc = J + delta*inner(dot(grad(u), u_), dot(grad(v), u_))*dx
elif args.ls == "direct":
    J_pc = None
"""

# PCD operators
mu = idt*inner(u, v)*dx
mp = 1.0/nu*p*q*dx
kp = (1.0/nu)*( dot(grad(p), u_) + idt*p )*q*dx
ap = inner(grad(p), grad(q))*dx
if args.pcd_variant == "BRM2":
    n = FacetNormal(mesh)
    ds = Measure("ds", subdomain_data=boundary_markers)
    # TODO: What about the reaction term? Does it appear here?
    kp -= 1.0/nu*dot(u_, n)*p*q*ds(1)
    #kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(0)  # TODO: Is this beneficial?

# Collect forms to define nonlinear problem
pcd_assembler = PCDAssembler(J, F, [bc0, bc1],
                             #J_pc, ap=ap, kp=kp, mp=mp, mu=mu, bcs_pcd=bc_pcd)
                            J_pc, ap=ap, kp=kp, mp=mp, bcs_pcd=bc_pcd)
assert pcd_assembler.get_pcd_form("gp").phantom # pressure grad obtained from J
problem = PCDNonlinearProblem(pcd_assembler)

# Set up linear solver (GMRES with right preconditioning using Schur fact)
PETScOptions.clear()
linear_solver = PCDKrylovSolver(comm=mesh.mpi_comm())
linear_solver.parameters["relative_tolerance"] = 1e-4
linear_solver.parameters["absolute_tolerance"] = 1e-8
#PETScOptions.set("ksp_monitor")

# Set up subsolvers
#PETScOptions.set("fieldsplit_p_pc_python_type", "fenapack.PCDRPC_" + args.pcd_variant)
PETScOptions.set("fieldsplit_p_pc_python_type", "fenapack.PCDPC_" + args.pcd_variant)
if args.ls == "iterative":

    PETScOptions.set("ksp_type", "fgmres")
    #PETScOptions.set("fieldsplit_u_ksp_monitor")
    #PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_monitor")
    #PETScOptions.set("fieldsplit_p_PCD_Rp_ksp_monitor")
    #PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_monitor")
    PETScOptions.set("fieldsplit_u_ksp_rtol", 1e-4)
    PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_rtol", 1e-4)
    PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_rtol", 1e-4)
    #PETScOptions.set("fieldsplit_p_PCD_Rp_ksp_rtol", 1e-4)

    PETScOptions.set("ksp_gmres_restart", 100)

    
    PETScOptions.set("fieldsplit_u_ksp_type", "gmres")
    PETScOptions.set("fieldsplit_u_pc_type", "hypre")
    PETScOptions.set("fieldsplit_u_pc_hypre_type", "boomeramg")
    PETScOptions.set("fieldsplit_u_pc_hypre_boomeramg_coarsen_type", "hmis")
    PETScOptions.set("fieldsplit_u_pc_hypre_boomeramg_interp_type", "ext+i")
    PETScOptions.set("fieldsplit_u_pc_hypre_boomeramg_p_max", 4)
    PETScOptions.set("fieldsplit_u_hypre_boomeramg_agg_nl", 1)
    
    #PETScOptions.set("fieldsplit_p_PCD_Rp_ksp_type", "gmres")
    #PETScOptions.set("fieldsplit_p_PCD_Rp_pc_type", "hypre")
    #PETScOptions.set("fieldsplit_p_PCD_Rp_pc_hypre_type", "boomeramg")
    #PETScOptions.set("fieldsplit_p_PCD_Rp_pc_hypre_boomeramg_coarsen_type", "hmis")
    #PETScOptions.set("fieldsplit_p_PCD_Rp_pc_hypre_boomeramg_interp_type", "ext+i")
    #PETScOptions.set("fieldsplit_p_PCD_Rp_pc_hypre_boomeramg_p_max", 4)
    #PETScOptions.set("fieldsplit_p_PCD_Rp_pc_hypre_boomeramg_agg_nl", 1)
    #PETScOptions.set("fieldsplit_p_PCD_Rp_ksp_type", "richardson")
    #PETScOptions.set("fieldsplit_p_PCD_Rp_ksp_max_it", 1)
    #PETScOptions.set("fieldsplit_p_PCD_Rp_pc_type", "hypre")
    #PETScOptions.set("fieldsplit_p_PCD_Rp_pc_hypre_type", "boomeramg")

    PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_type", "cg")
    PETScOptions.set("fieldsplit_p_PCD_Ap_pc_type", "hypre")
    PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_type", "boomeramg")
    PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_boomeramg_coarsen_type", "hmis")
    PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_boomeramg_interp_type", "ext+i")
    PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_boomeramg_p_max", 4)
    PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_boomeramg_agg_nl", 1)

    PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_type", "cg")
    PETScOptions.set("fieldsplit_p_PCD_Mp_pc_type", "jacobi")

    """
    PETScOptions.set("fieldsplit_u_ksp_type", "richardson")
    PETScOptions.set("fieldsplit_u_ksp_max_it", 1)
    PETScOptions.set("fieldsplit_u_pc_type", "hypre")
    PETScOptions.set("fieldsplit_u_pc_hypre_type", "boomeramg")
    
    PETScOptions.set("fieldsplit_p_PCD_Rp_ksp_type", "richardson")
    PETScOptions.set("fieldsplit_p_PCD_Rp_ksp_max_it", 1)
    PETScOptions.set("fieldsplit_p_PCD_Rp_pc_type", "hypre")
    PETScOptions.set("fieldsplit_p_PCD_Rp_pc_hypre_type", "boomeramg")
    PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_type", "richardson")
    PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_max_it", 2)
    PETScOptions.set("fieldsplit_p_PCD_Ap_pc_type", "hypre")
    PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_type", "boomeramg")
    PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_type", "chebyshev")
    PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_max_it", 5)
    PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues", "0.5, 2.0")
    #PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_chebyshev_esteig", "1,0,0,1")  # FIXME: What does it do?
    PETScOptions.set("fieldsplit_p_PCD_Mp_pc_type", "jacobi")
    """
elif args.ls == "direct" and args.mumps_debug:
    # Debugging MUMPS
    PETScOptions.set("fieldsplit_u_mat_mumps_icntl_4", 2)
    PETScOptions.set("fieldsplit_p_PCD_Ap_mat_mumps_icntl_4", 2)
    PETScOptions.set("fieldsplit_p_PCD_Mp_mat_mumps_icntl_4", 2)

# Apply options
linear_solver.set_from_options()

# Set up nonlinear solver
solver = PCDNewtonSolver(linear_solver)
solver.parameters["relative_tolerance"] = 3e-3

# files
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

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
            nu.nu_t = (np.exp(-t) - np.exp(-ramp_time/2.0))*999.0*args.viscosity + args.viscosity

        else:
            nu.nu_t = args.viscosity
    """
    nu.nu_t = final_nu
    info("Viscosity: %g" % nu.nu_t)
    # Update boundary conditions
    inflow.t = t

    # Solve the nonlinear problem
    info("t = {:g}, step = {:g}, dt = {:g}".format(t, time_iters, args.dt))
    with Timer("Solve") as t_solve:
        newton_iters, converged = solver.solve(problem, w.vector())
    krylov_iters += solver.krylov_iterations()
    solution_time += t_solve.stop()

    if time_iters%args.ts_per_out==0:
        u_out, p_out = w.split()
        ufile << u_out
        pfile << p_out

    # Update variables at previous time level
    w0.assign(w)


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
u, p = w.split()
size = MPI.size(mesh.mpi_comm())
rank = MPI.rank(mesh.mpi_comm())
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

