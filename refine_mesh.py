from dolfin import *
from mpi4py import MPI

comm = MPI.COMM_WORLD

class Area1(SubDomain):
    def inside(self, x, on_boundary):
        #return (x[0]>=0.2)and(x[0]<=1.0)and(x[1]>=0.25)and(x[1]<=0.75)and(x[2]>=0.25)and(x[2]<=0.75)
        return (x[0]>=0.15)and(x[0]<=0.7)and(x[1]>=0.25)and(x[1]<=0.45)and(x[2]<=0.15)

class Area2(SubDomain):
    def inside(self, x, on_boundary):
        #return (x[0]<=1.5)
        return (x[0]<=1.0)and(x[1]>=0.15)and(x[1]<=0.55)

class Area99(SubDomain):
    def inside(self, x, on_boundary):
        return True

mesh = Mesh("bump_channel.xml")

for i in range(2):
    Coarse_marker = MeshFunction("bool", mesh, 1)#mesh.topology().dim()-1)
    Coarse_marker.set_all(False)
    Area99().mark(Coarse_marker, True)
    mesh = refine(mesh, Coarse_marker)

for i in range(1):
    Mid_marker = MeshFunction("bool", mesh, 1)
    Mid_marker.set_all(False)
    Area2().mark(Mid_marker, True)
    mesh = refine(mesh, Mid_marker)

for i in range(0):
    Fine_marker = MeshFunction("bool", mesh, 1)
    Fine_marker.set_all(False)
    Area1().mark(Fine_marker, True)
    mesh = refine(mesh, Fine_marker)

fid = HDF5File(comm,'bump_local_refined.h5','w')
fid.write(mesh,'mesh')
fid.close()
