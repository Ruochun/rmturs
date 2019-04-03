import sys
from dolfin import *
import time

import numpy as np

from mpi4py import MPI as PYMPI

import argparse, os, gc

parameters["form_compiler"]["quadrature_degree"] = 3
parameters["mesh_partitioner"] = "SCOTCH"
parameters["std_out_all_processes"] = False
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

class NavierStokesSolver(NonlinearVariationalSolver):
    def __init__(self, problem=None):
        self.problem = problem
        self.storedParas = {'newton_solver':{'absolute_tolerance':1e-8,
                                             'relative_tolerance':1e-7,
                                             'maximum_iterations':25,
                                             'relaxation_parameter':1.0}
                           }
                                              
    def defProblem(self, problem):
        self.problem = problem
            
    def solve(self, NS=None):
        if NS!=None:
            problem = NonlinearVariationalProblem(NS.F, NS.d, NS.bcs, NS.J)
        elif self.problem!=None:
            problem = NonlinearVariationalProblem(self.problem.F, self.problem.d, self.problem.bcs, self.problem.J)
        else:
            error("Please specify the non-linear problem to be solved!")
        NonlinearVariationalSolver.__init__(self, problem)
        #self.applyStoredParas()
        NonlinearVariationalSolver.solve(self)
    
    def applyStoredParas(self):
        prm = self.parameters
        for item in self.storedParas:
            prm[item] = self.storedParas[item]
    """
    def restoreDefault(self):
        prm = self.parameters
        prm['absolute_tolerance'] = 1E-8
        prm['relative_tolerance'] = 1E-7
        prm['maximum_iterations'] = 25
        prm['relaxation_parameter'] = 1.0
    """
class NavierStokesSystem:
    def __init__(self, W=None, d=None, rho=1.0, nu=0.01, bcs=[], isTransient=True, stepping=["Euler", 0.5], form=True):
        #self.FunctionSpace = W
        #self.Function = d
        self.rho = rho
        self.nu = nu
        self.bcs = bcs
        if (W!=None)and(d!=None):
            self.canForm = True
            v, q = TestFunctions(W)
            u, p = TrialFunctions(W)
            if form:
                F = ( nu*inner(grad(u), grad(v)) + inner(dot(grad(u), u), v) - p*div(v) + q*div(u) ) * dx
                self.d = d
                self.constructJacobi(F, self.d)
        else:
            self.canForm = False
    #def setFunctionSpace(self, W, form=False):
    def constructJacobi(self, F, d):
        self.F = action(F, d)
        self.J = derivative(self.F, d)

    def setBCs(self, bcs):
        self.bcs = bcs
    
    #def formProblem(self):
        #if (self.__FunctionSpace==None):
        #    sys.exit("The function space on which the Navier--Stokes problem lies is not given, unable to form problem.")
        #self.F = ( nu*inner(grad(u), grad(v)) + inner(dot(grad(u), u), v) - p*div(v) + q*div(u) ) * dx
            
        

def NS(w_, W, bcu, h, nu, rho, problem):
    u_, p_ = w_.split()
    #u, p = TrialFunctions(W)
    vnorm = sqrt(dot(u_,u_))
    # SUPG & PSPG stabilization parameters
    tau_supg = ( (2.0*vnorm/h)**2 + 9*(4.0*nu/h**2)**2 )**(-0.5)
    tau_pspg = h**2/2
    # Left hand side
    """
    F = ( nu*inner(grad(u_), grad(v))\
         + inner(dot(grad(u_), u_), v)\
         - p_*div(v)\
         + q*div(u_)\
         ) *dx
    # Add SUPG/PSPG stabilization
    F += tau_supg*inner(grad(v)*u_,grad(u_)*u_+grad(p_)-div(nu*grad(u_)))*dx\
          + tau_pspg*inner(grad(q),grad(u_)*u_+grad(p_)-div(nu*grad(u_)))*dx\
    #F = action(F, w_)
    J = derivative(F, w_)
    """
    #problem = NavierStokesSystem(W, w_, bcs = [], nu=nu)
    #problem.setBCs(bcu)
    """
    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "lu"
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6
    """
    #solver  = NonlinearVariationalSolver(problem)
    solver = NavierStokesSolver(problem)
    #solver.restoreDefault()
    solver.solve()
    """
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-8
    prm['newton_solver']['relative_tolerance'] = 1E-7
    prm['newton_solver']['maximum_iterations'] = 25
    prm['newton_solver']['relaxation_parameter'] = 1.0
    """
    return w_

if __name__ == "__main__":
    print('something')
