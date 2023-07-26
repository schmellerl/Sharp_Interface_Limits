from operator import truediv
from fenics import *
from parameters import *

import numpy as np
import os
import sys

interface_factor = Constant(1.0/sqrt(2))
noslip   = Constant((0, 0)) 
noslip1d = Constant(0)

# fileX = File(filepath + "phasefield.pvd")

def get_initial_data(a, b, eps, h0, h1, H1):
    interface_factor =  Constant(1.0/sqrt(2))
    psi11   = Expression(("tanh(ff*(h0-x[1])/eps)"),eps=eps,h0=h0,h1=h1,ff=interface_factor,degree=2 )
    #psi22   = Expression(("-1 + 2*(1+tanh(ff*(x[1]-h0)/eps))*(1+tanh(ff/eps*(h1-pow(pow(x[0]-H1/2,2.0)/(a)+pow(x[1]-h0,2.0)/(b),0.5))))/4"),a=a,b=b,eps=eps,h0=h0,h1=h1,ff=interface_factor,H1=0,degree=2)
    psi22   = Expression(("-1 + 2*(1+tanh(ff*(x[1]-h0)/eps))*(1+tanh(ff/eps*(h1-pow(pow((x[0]-H1/2)/a,2.0) +pow((x[1]-h0)/b,2.0),0.5))))/4"),a=a,b=b,eps=eps,h0=h0,h1=h1,ff=interface_factor,H1=0,degree=2)
    if INCOMPRESSIBLE:
        if MOBILITY:
            initial = Expression(("0","0","0","0","0","0","0"),a=a,b=b,eps=eps,h0=h0,h1=h1,ff=interface_factor,H1=0,degree=2)
        else:
            initial = Expression(("0","0","0"),a=a,b=b,eps=eps,h0=h0,h1=h1,ff=interface_factor,H1=0,degree=2)
    else:
        if MOBILITY:
            initial = Expression(("0","0","0","0","0","0"),a=a,b=b,eps=eps,h0=h0,h1=h1,ff=interface_factor,H1=0,degree=2)
        else:
            initial = Expression(("0","0"),a=a,b=b,eps=eps,h0=h0,h1=h1,ff=interface_factor,H1=0,degree=2)
    return psi11, psi22, initial


def GshearF(psi1, psi2):
    psi3 = -1-psi1-psi2
    xi1 = (1+psi1)/2
    xi2 = (1+psi2)/2
    xi3 = (1+psi3)/2
    return xi1*G1 + xi2*G2 + xi3*G3

def mu_phase(psi1, psi2):
    psi3 = -1-psi1-psi2
    xi1 = (1+psi1)/2
    xi2 = (1+psi2)/2
    xi3 = (1+psi3)/2
    return xi1*mu1 + xi2*mu2 + xi3*mu3

def get_BC(W):   
    if INCOMPRESSIBLE:
        bc1   = DirichletBC(W.sub(0).sub(0), noslip1d, boundary_bot1)
        bc2   = DirichletBC(W.sub(0),        noslip, boundary_bot2)
        bc3   = DirichletBC(W.sub(0).sub(0), noslip1d, boundary_bot3)
        bc4   = DirichletBC(W.sub(0),        noslip, boundary_bot4)
    else:
        if MOBILITY:
            bc1   = DirichletBC(W.sub(0).sub(0), noslip1d, boundary_bot1)
            bc2   = DirichletBC(W.sub(0)       , noslip  , boundary_bot2)
            bc3   = DirichletBC(W.sub(0).sub(0), noslip1d, boundary_bot3)
            bc4   = DirichletBC(W.sub(0)       , noslip  , boundary_bot4)  
        else:
            bc1   = DirichletBC(W.sub(0), noslip1d, boundary_bot1)
            bc2   = DirichletBC(W       , noslip  , boundary_bot2)
            bc3   = DirichletBC(W.sub(0), noslip1d, boundary_bot3)
            bc4   = DirichletBC(W       , noslip  , boundary_bot4)      
    bc = [bc1,bc2,bc3,bc4]
    return bc

def boundary_bot1(x, on_boundary):
    return on_boundary and near(x[0], 0, tol)
def boundary_bot2(x, on_boundary):
    return on_boundary and near(x[1], 0, tol)
def boundary_bot3(x, on_boundary):
    return on_boundary and near(x[0], H1, tol)
def boundary_bot4(x, on_boundary):
    return on_boundary and near(x[1], H2, tol)

#def output_state(mesh,q,psi11,psi22,fname, t=0):
def output_state(mesh,q,fname, t=0):
    ff = open(fname + '.time','w')
    ff.write(str(t)+'\n')
    ff.close()

    f=HDF5File(mesh.mpi_comm(),fname + '.h5', 'w')
    f.write(mesh,"mesh")   
    f.write(q,"q",t)
    if MOBILITY:
        v,psi1,psi2,*_  = split(q) 
    else: 
        psi11, psi22, _   = get_initial_data(a, b, eps, h0, h1, H1)
        psi1 = project(psi11, get_scalar(mesh))
        psi2 = project(psi22, get_scalar(mesh))

    if MOBILITY:
        print("TODO")
    else:
        f.write(psi1,"psi1")
        f.write(psi2,"psi2")
    f.close()
    # filepath + 'state_'+str(m)+'.h5'

def output_PV(file1, mesh, q, t=0):
    if MOBILITY:
        v,psi1,psi2,*_  = split(q) 
        #psi1.rename('psi1','Phase field')
        #psi2.rename('psi2','Phase field')
    else: 
        psi11, psi22, _   = get_initial_data(a, b, eps, h0, h1, H1)
        psi1 = project(psi11, get_scalar(mesh))
        psi2 = project(psi22, get_scalar(mesh))

    
    psi3 = project( -1-psi1-psi2, get_scalar(mesh))
    psi4 = project( (1+psi2)/2 - (1+psi3)/2, get_scalar(mesh))

    psi4.rename('psi4','Phase field')
    tmp = Function(get_vector(mesh))
    tmp.assign(-q)
    
    # visualization on deformed mesh 
    if INCOMPRESSIBLE:
        ALE.move(mesh,q.sub(0))
    else:
        if MOBILITY:
            ALE.move(mesh,q.sub(0))
        else:
            ALE.move(mesh,q)
    
    file1 << (psi4, t) 

    if INCOMPRESSIBLE:
            ALE.move(mesh,tmp.sub(0))
    else:
        if MOBILITY:
            ALE.move(mesh,tmp.sub(0))
        else:
            ALE.move(mesh,tmp)
    
def get_space(mesh):
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

def get_scalar(mesh):
    Ps = FiniteElement("P", mesh.ufl_cell(), FEs_DEG)
    S = FunctionSpace(mesh, Ps)
    return S

def get_vector(mesh): 
    if FEu_DEG>0: # Standard element
        Pv = VectorElement("P", mesh.ufl_cell(), FEu_DEG)
    else: # Mini element
        Pk = FiniteElement("Lagrange", mesh.ufl_cell(), -FEu_DEG)
        B  = FiniteElement("Bubble",   mesh.ufl_cell(), 3)
        Pv = VectorElement(NodalEnrichedElement(Pk, B))     
    V = FunctionSpace(mesh, Pv)
    return V

def mark_isolevel(mesh,u,iso_level=0,ceps=1e-6):
    cell_markers = MeshFunction("bool",mesh,mesh.geometric_dimension())
    cell_markers.set_all(False)
    dm = u.function_space().dofmap()
    uv = u.vector()
    for cell in cells(mesh):
        cell_index = cell.index()
        cell_dofs = dm.cell_dofs(cell_index)
        umin = uv[cell_dofs].min()
        umax = uv[cell_dofs].max()
        if (umin<(iso_level+ceps)) and (umax>(iso_level-ceps)):
            cell_markers[cell] = True            
    return cell_markers

def refine_and_adapt(marker,mesh,level=None):
    level_coarse = MeshFunction("size_t", mesh, mesh.geometric_dimension())
    # level == None -> level_coarse = 1 on refined and 0 otherwise
    # level exists -> level_coarse = level + 1 on refined and level otherwise
    if (level==None):
      for i in range(mesh.num_cells()):
        if (marker[i] == True):
          level_coarse[i] = 1
        else:
          level_coarse[i] = 0
    else:
      for i in range(mesh.num_cells()):
        if (marker[i] == True):
            level_coarse[i] = level[i] +1
        else:
            level_coarse[i] = level[i]
  
    mesh_fine  = refine(mesh, marker)
    level_fine = adapt(level_coarse,mesh_fine)
    # u_fine     = project(u,get_space(mesh_fine))
    return mesh_fine,level_fine

def mark_collect(marker1,marker2,mesh):
  out_marker = MeshFunction("bool",mesh,mesh.geometric_dimension())
  for i in range(marker1.size()):
      out_marker[i] = (marker1[i] or marker2[i])
  return out_marker
