#!/usr/bin/env python
# coding: utf-8

# <h2> General utility functions </h2>

# In[4]:


from dolfin import *
from mshr import *
from ufl import cofac
import matplotlib.pyplot as plt
import numpy as np
from os import mkdir

#fpath_DI = './Diffusi/'

### DEFINE PARAMETERS ###
#nuni           = 2
taufac         = 1
FEu_DEG        = 2

mu             = 1.00
dt             = 0.005 / taufac
dt0            = 0.050
n_steps        = 250 * taufac
FE_quad        = 5
RADIAL         = True

F = 'E3MInf_G10_S/'

# Mobility false für M0, und True für MInf und sonst

parsD           = {'FEu_DEG':2      ,'FEp_DEG':2,'FEs_DEG':2,'INCOMPRESSIBLE':False,'MOBILITY':True }

filepath1_h5    = F +'output/'
filebase_h5     = 'state'

outnum =  23
t0p    =  0

fout = 'POSTPROCESSING_' + F 

def get_space(mesh,pars):
    FEu_DEG = pars['FEu_DEG']
    FEp_DEG = pars['FEp_DEG']
    FEs_DEG = pars['FEs_DEG']
    INCOMPRESSIBLE = pars['INCOMPRESSIBLE'] 
    MOBILITY = pars['MOBILITY']
    if MOBILITY:
        if FEu_DEG>0: # Standard element
            Pv = VectorElement("P", mesh.ufl_cell(), FEu_DEG)
        else: # Mini element
            Pk = FiniteElement("Lagrange", mesh.ufl_cell(), -FEu_DEG)
            B  = FiniteElement("Bubble",   mesh.ufl_cell(), 3)
            Pv = VectorElement(NodalEnrichedElement(Pk, B))    
        if FEp_DEG>0:
            R  = FiniteElement("P", mesh.ufl_cell(), FEp_DEG)
        else:
            R  = FiniteElement("DG", mesh.ufl_cell(), -FEp_DEG)
        Ps = FiniteElement("P", mesh.ufl_cell(), FEs_DEG)
        if INCOMPRESSIBLE:
            V   = FunctionSpace(mesh, MixedElement([Pv,Ps,Ps,Ps,Ps,R])) 
        else:
            V   = FunctionSpace(mesh,MixedElement([Pv,Ps,Ps,Ps,Ps]))
    else:
        if FEu_DEG>0: # Standard element
            Pv = VectorElement("P", mesh.ufl_cell(), FEu_DEG)
        else: # Mini element
            Pk = FiniteElement("Lagrange", mesh.ufl_cell(), -FEu_DEG)
            B  = FiniteElement("Bubble",   mesh.ufl_cell(), 3)
            Pv = VectorElement(NodalEnrichedElement(Pk, B))    
        if FEp_DEG>0:
            R  = FiniteElement("P", mesh.ufl_cell(), FEp_DEG)
        else:
            R  = FiniteElement("DG", mesh.ufl_cell(), -FEp_DEG)
        Ps = FiniteElement("P", mesh.ufl_cell(), FEs_DEG)

        if INCOMPRESSIBLE:
            V   = FunctionSpace(mesh, MixedElement([Pv, R])) 
        else:
            V   = FunctionSpace(mesh,Pv)
    return V


def get_scalar(mesh,pars):
  FEu_DEG = pars['FEu_DEG']
  FEp_DEG = pars['FEp_DEG']
  FEs_DEG = pars['FEs_DEG']
  INCOMPRESSIBLE = pars['INCOMPRESSIBLE']
  MOBILITY = pars['MOBILITY']
    
  Ps = FiniteElement("P", mesh.ufl_cell(), FEs_DEG)
  S = FunctionSpace(mesh, Ps)
  return S

def get_vector(mesh,pars): 
  FEu_DEG = pars['FEu_DEG']
  FEp_DEG = pars['FEp_DEG']
  FEs_DEG = pars['FEs_DEG']
  INCOMPRESSIBLE = pars['INCOMPRESSIBLE']
  MOBILITY = pars['MOBILITY']
  if FEu_DEG>0: # Standard element
      Pv = VectorElement("P", mesh.ufl_cell(), FEu_DEG)
  else: # Mini element
      Pk = FiniteElement("Lagrange", mesh.ufl_cell(), -FEu_DEG)
      B  = FiniteElement("Bubble",   mesh.ufl_cell(), 3)
      Pv = VectorElement(NodalEnrichedElement(Pk, B))     
  V = FunctionSpace(mesh, Pv)
  return V



def read_diffuse(fname,pars):
    INCOMPRESSIBLE = pars['INCOMPRESSIBLE']
    MOBILITY = pars['MOBILITY']

    f = open(fname+'.time','r')
    t = float(f.read())
    f.close()

    mesh = Mesh()
    f=HDF5File(mesh.mpi_comm(),fname+'.h5', 'r')
    f.read(mesh,"mesh",False)   
    
    W    = get_space(mesh,pars)
    q    = Function(W)
    f.read(q,"q")

    if MOBILITY:
        if INCOMPRESSIBLE:
            v, psi1, psi2, eta1, eta2,lambda1 = q.split()
        else:
            v,psi1,psi2,eta1,eta2 = q.split()
    else:
        if INCOMPRESSIBLE:
            v, lambda1 = q.split()
        else:
            v = q
        psi1 = Function(get_scalar(mesh,pars))
        psi2 = Function(get_scalar(mesh,pars))
        f.read(psi1,"psi1")   
        f.read(psi2,"psi2")   
    f.close()
    psi4 = project(psi1-psi2,get_scalar(mesh,pars))
    return v, t, mesh,psi4

def get_interfaces(u_ALE):
    h     = 1.0
    r     = 0.5
    H1    = 1.0    
    delta = 1e-4
    
    m1 = RectangleMesh(Point(0,h-delta/2), Point(r,h+delta/2),128,1)
    m2 = RectangleMesh(Point(r,h-delta/2), Point(H1,h+delta/2),128,1)
    m3 = RectangleMesh(Point(0,-delta/2), Point(pi/2,+delta/2),128,1)
    
    coords = m3.coordinates()    
    for i in range(np.size(coords,0)):
        phi = coords[i,0]
        dr  = coords[i,1]
        coords[i,0] =   (r+dr)*np.cos(phi)
        coords[i,1] = h+(r+dr)*np.sin(phi)    
    
    V1 = VectorFunctionSpace(m1,'CG',2)
    u1 = project(u_ALE,V1)
    ALE.move(m1,u1)
    
    V2 = VectorFunctionSpace(m2,'CG',2)
    u2 = project(u_ALE,V2)
    ALE.move(m2,u2)
    
    V3 = VectorFunctionSpace(m3,'CG',2)
    u3 = project(u_ALE,V3)
    ALE.move(m3,u3)
    
    return m1, m2, m3



f0  = File(fout + "diffuse.pvd")
f1  = File(fout + "diffuse_SL.pvd")
f2  = File(fout + "diffuse_SA.pvd")
f3  = File(fout + "diffuse_LA.pvd")
ff0 = File(fout + "bdiffuse.pvd")

for m in range(0,outnum+1): 
    print(m)
    if (m==0):
        fname1 = filepath1_h5 + 'initial'
    else:
        fname1 = filepath1_h5 + filebase_h5 + '_' + str(m)
    
    v1,t1,mesh1,id1 = read_diffuse(fname1,parsD)    
    
    m0 = RectangleMesh(Point(0,0),Point(1,1),32,32) #,63,63) 
    W0 = VectorFunctionSpace(m0, "CG", 2)
    ALE.move(m0,project(v1,W0))
    
    m1,m2,m3 = get_interfaces(v1)
    
    f1 << m1
    f2 << m2
    f3 << m3
    
    id1.rename("id1","id1")
    ALE.move(mesh1,v1)
    
    f0  << id1
    ff0 << m0





