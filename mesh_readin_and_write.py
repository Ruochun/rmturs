from dolfin import *
from mpi4py import MPI as pmp
commmpi = pmp.COMM_WORLD
"""
if not has_cgal():
    print("DOLFIN must be compiled with CGAL to run this demo.")
    exit(0)
box_1 = Sphere()
#box_1 = Box(0,0,0,1,1,1)
#box_1 = Box(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0))
box_2 = Box(Point(1.0, 1.0, 1.0), Point(2.0, 0.0, 0.0)) 
box_tot = box_1 + box_2
mesh_tot = generate_mesh(box_tot, 32)
"""
mesh = Mesh('bump.xml')
#mesh = Mesh()
#fid = HDF5File(commmpi, 'bump.xml', 'r')
#fid.read(mesh, 'mesh', False)
#fid.close()
File("./mesh_temp/mesh.pvd") << mesh
