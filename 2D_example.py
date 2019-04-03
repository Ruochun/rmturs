import sys
from dolfin import *
import time

import numpy as np
#from mma import mmasub

from mpi4py import MPI as PYMPI

import argparse, os, gc
import NavierStokesSystems as NSS

parameters["form_compiler"]["quadrature_degree"] = 3
parameters["mesh_partitioner"] = "SCOTCH"
parameters["std_out_all_processes"] = False
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

commmpi = PYMPI.COMM_WORLD
rank = commmpi.Get_rank()
size = commmpi.Get_size()
root = 0
##############
#### Path ####
##############
path = 'ex04/'


###################
#### Load mesh ####
###################
mesh = Mesh('2dgeo3.xml')


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
        return on_boundary and x[1]>1.5-eps

# Oultet bc
class Gamma2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0]>2.0-eps)# or x[0]<-1.0+eps)

# HeatSource bc
class Gamma3(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1]<0.0+eps and between(x[0],(-0.4,0.1))

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_markers.set_all(4)        # interior facets
Gamma0().mark(boundary_markers, 0) # no-slip facets
Gamma1().mark(boundary_markers, 1) # inlet facets
Gamma2().mark(boundary_markers, 2) # outlet facet
Gamma3().mark(boundary_markers, 3) # heat source bc
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# Design domain
class DesignDomain(SubDomain):
	def inside(self, x, on_boundary):
		return  x[0]<=0.2 and x[1]<=0.7

design_domain = DesignDomain()
domains = MeshFunction("size_t",mesh,mesh.topology().dim())
domains.set_all(0)
design_domain.mark(domains, 1)
dx = Measure("dx")(subdomain_data=domains)


##############################
#### Build function space ####
##############################
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
P2 = VectorElement("CG", mesh.ufl_cell(), 1)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P2*P1)
# Tell the degree of freedoms
if rank == 0:
    info("Dimension of the function space: %g" % W.dim())
    info("DOF of velocity: %g" % W.sub(0).dim())
    info("DOF of pressure: %g" % W.sub(1).dim())


###################################
#### Parameters & coefficients ####
###################################
# Darcy's number
Da = 1e-4
# Reynolds number
Re = 10000.
# Kinematic viscosity
nu = 0.01
nu_air = 1.7894e-5
# Hydraulic diameter
l = 1.0
# Maximum velocity at inlet
u0 = 2.0*Re*nu/l
# Thermal conductivity & diffusivity & density & specific heat
k_s = 110.0
k_f = 0.0242
#alpha_s = 9.7e-5
#alpha_f = 1.43e-7
rho_s = 2.719e3
rho_f = 1.225
c_s = 871.0
c_f = 1006.43
h = CellDiameter(mesh)
#mu_water = 0.001003
#### Boundary condition ####
############################
# Inlet velocity
u_in = Expression(("0.0","-u0*(0.5+x[0])*(0.5-x[0])/pow(0.5,2)"),u0=0.0001*u0,degree=2)
# Navier-stokes bc
bc00 = DirichletBC(W.sub(0), (0.0, 0.0), boundary_markers, 0)
bc01 = DirichletBC(W.sub(0), (0.0, 0.0), boundary_markers, 3)
bc1 = DirichletBC(W.sub(0), u_in, boundary_markers, 1)
bcu = [bc00, bc01, bc1]
# Artificial bc for PCD preconditioner


####################
#### Initialize ####
####################
iteration = 0
change = 1
w_  = Function(W)
gamma_h = Function(Q)


###############
#### Start ####
###############
ramp_ts = 200
problem = NSS.NavierStokesSystem(W, w_, bcs = [], nu=nu)
problem.setBCs(bcu)

while iteration <= ramp_ts:
    iteration += 1
    info("Ramping iter %d" % iteration)
    current_u0 = (iteration/ramp_ts)*u0
    u_in.u0 = current_u0
    w_ = NSS.NS(w_, W, bcu, h, nu, rho_f, problem)
    #u1, p1 = w1.split()
    #assign(uc_, w1.sub(0))



iteration = 0
u_in.u0 = u0

while iteration < 200:
    # Clean the memory
    gc.collect()
    iteration += 1

    w_ = NS(w_, W, bcu, h, nu, rho_f)
    u1, p1 = w_.split()
