# generate import FEniCS libary
# podman run --volume $(pwd):/home/fenics/shared -it quay.io/fenicsproject/stable
from fenics import *
from pylab import plt

import numpy as np
import os
import sys

# Paramters supplied via command line options, otherwise use parameters in  parameters.py
if len(sys.argv)==2:
    parfile = str(sys.argv[1])
    print('Import parameters from ',parfile)
    os.system('cp ' + parfile + ' parameters.py')

from parameters import *
from phasefield_custom import *

print("Creating initial data for simulation")
print("OUTPUT FOLDER : ",filepath)
print("INCOMPRESSIBLE: ",INCOMPRESSIBLE)
print("MOBILITY      : ",MOBILITY)
print("DEG FEu       : ",FEu_DEG)
print("DEG FEp       : ",FEp_DEG)
print("")

os.system('mkdir '+filepath)
os.system('cp parameters.py ' + filepath)

class SolidMarker(SubDomain):
    def __init__(self,eps):
        self.eps = eps
        SubDomain.__init__(self) # Call base class constructor!
    def inside(self, x, on_boundary):
        return (x[1] <= 1-eps/2)

class ContactLineMarker(SubDomain):
    def __init__(self,delta,eps,h1):
        self.delta = delta
        self.eps   = eps
        self.h1    = h1
        SubDomain.__init__(self) # Call base class constructor!
    def inside(self, x, on_boundary):
        return ((pow(x[0]-self.h1,2) + pow(x[1]-1.0,2)) < pow(self.delta,2))
        # return ((x[1] <= 1-self.eps/2) and ((pow(x[0]-self.h1,2) + pow(x[1]-1.0,2)) < pow(self.delta,2)))

def refine_loop(mesh, old_q, psi11, psi22): 
    mesh2 = mesh
    delta = DELTA0
    for k in range(N_refine):
        V           = get_scalar(mesh2)
        marker_1    = mark_isolevel(mesh2,interpolate(psi11, V),ceps=ceps)
        marker_2    = mark_isolevel(mesh2,interpolate(psi22, V),ceps=ceps)
        # original ohne
        if k<N_refine_solid:
            solid = SolidMarker(eps)
            solid.mark(marker_1,True)

        if k<N_refine_cl:
            cl    = ContactLineMarker(delta,eps,h1)
            cl.mark(marker_1,True)
            delta *= DELTAFAC

        marker1      = mark_collect(marker_1,marker_2,mesh2)
        mesh1,_      = refine_and_adapt(marker1,mesh2)   
        mesh2        = mesh1
        q            = interpolate(initial, get_space(mesh2))

    if MOBILITY:
        Psi1  = interpolate(psi11, get_scalar(mesh2))
        assign(q.sub(1), Psi1)
        Psi2  = interpolate(psi22, get_scalar(mesh2))
        assign(q.sub(2), Psi2)

    return mesh2, q

print("interpolate...")
psi11, psi22, initial = get_initial_data(a, b, eps, h0, h1, H1)
mesh0                 = RectangleMesh(Point(0,0),Point(H1,H2),Nx,Ny)
old_q                 = interpolate(initial,  get_space(mesh0))


print("refine...")
mesh, old_q           = refine_loop(mesh0, old_q, psi11, psi22)

if REFINE: 
    mesh1                 = refine(mesh)
    mesh = mesh1
    old_q = interpolate(initial, get_space(mesh))
    Psi1  = interpolate(psi11, get_scalar(mesh))
    assign(old_q.sub(1), Psi1)
    Psi2  = interpolate(psi22, get_scalar(mesh))
    assign(old_q.sub(2), Psi2)

    #old_q                 = project(old_q, get_space(mesh))  
    
    print("refine... again")
if REFINE2: 
    mesh1                 = refine(mesh)
    mesh = mesh1
    old_q = interpolate(initial, get_space(mesh))
    Psi1  = interpolate(psi11, get_scalar(mesh))
    assign(old_q.sub(1), Psi1)
    Psi2  = interpolate(psi22, get_scalar(mesh))
    assign(old_q.sub(2), Psi2)

    #old_q                 = project(old_q, get_space(mesh)) 
    
    #mesh2                  = refine(mesh)
    #mesh = mesh2
    #old_q                  = interpolate(initial,  get_space(mesh))
    
    print("refine... again and again")

print(" -> mesh vertices = ",mesh.num_vertices())
print(" -> mesh cells    = ",mesh.num_cells())

print("write to file...")
print(" -> output PV")
file1 = File(filepath + "initial.pvd")
output_PV(file1, mesh, old_q, 0)

print(" -> output h5 to ",filepath + 'initial.h5')
output_state(mesh,old_q,filepath + 'initial',0.0)

# print("plot mesh...")
# psi1 = project(psi11, get_scalar(mesh))
# psi2 = project(psi22, get_scalar(mesh))
# psi3 = project( -1-psi1-psi2, get_scalar(mesh))
# psi4 = project( (1+psi2)/2 - (1+psi3)/2, get_scalar(mesh))
# plot(psi4,extend="both")
# plot(mesh,linewidth=0.02,color='w')
# plt.savefig(filepath + 'mesh.png',dpi=2400)