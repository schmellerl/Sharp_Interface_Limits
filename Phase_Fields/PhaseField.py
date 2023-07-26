from fenics import *
from pylab import plt

import numpy as np
import os
import sys

from parameters import *
from phasefield_custom import *

print("Starting simulation")
print("OUTPUT FOLDER : ",filepath)
print("INCOMPRESSIBLE: ",INCOMPRESSIBLE)
print("MOBILITY      : ",MOBILITY)
print("DEG FEu       : ",FEu_DEG)
print("DEG FEp       : ",FEp_DEG)
print("")

file1 = File(filepath + "phasefield.pvd")

def scalar_grad_axi(u):
    if RADIAL:
        return as_vector([u.dx(0), 0, u.dx(1)])
    else:
        return grad(u)
  
def vector_grad_axi(v):
    if RADIAL:
        return as_tensor([[v[0].dx(0), 0, v[0].dx(1)], [0, v[0]/r[0], 0],
                                    [v[1].dx(0), 0, v[1].dx(1)]])
    else:
        return grad(v)

def incremental_minimization(old_q, tau):
    W = old_q.function_space()
    q, dq = Function(W), TestFunction(W)


    if MOBILITY:
        if INCOMPRESSIBLE:
            v, psi1, psi2, eta1, eta2,lambda1   = split(q)              # Current solution
            old_v,old_psi1,old_psi2,*_          = split(old_q)          # Old solution
            dv,dps1,dpsi2,dfeta1,dfeta2,_       = split(dq)           # Test functions
        else:
            v,psi1,psi2,eta1,eta2               = split(q)
            old_v,old_psi1,old_psi2,*_          = split(old_q)  
            dv,dpsi1,dpsi2,dfeta1,dfeta2        = split(dq)  

    else:
        if INCOMPRESSIBLE:
            v, lambda1      = split(q)              # Current solution
            old_v, _        = split(old_q)  # Old solution
            dv,_            = split(dq)           # Test functions
        else:
            v       = q
            dv      = dq
            old_v   = old_q
        psi1 = project(psi11, get_scalar(mesh))
        psi2 = project(psi22, get_scalar(mesh))
 
    if RADIAL: 
        d = 3
    else:
        d = 2

    I       = Identity(d)
    F       = I + vector_grad_axi(v)
    old_F   = I + vector_grad_axi(old_v)

    C = F.T*F        # (right) Cauchy-Green tensor

    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["quadrature_degree"] = FE_quad

    psi3 = -1-psi1-psi2

    gradpsi1 = (inv(F).T)*scalar_grad_axi(psi1)
    gradpsi2 = (inv(F).T)*scalar_grad_axi(psi2)
    gradpsi3 = (inv(F).T)*scalar_grad_axi(psi3)

    # Stored strain energy density (compressible neo-Hookean model)
    e_elastic = GshearF(psi1, psi2)/2*tr(C - I)

    if INCOMPRESSIBLE:
        e_comp = lambda1*(det(F)-1)
    else:
        e_comp = kappa*(det(F)-1)**2
   
    
    e_phase1 = 3/(2*sqrt(2))*sigma1*(1/(4*eps)*(1-psi1**2)**2 + eps/2*inner(gradpsi1, gradpsi1))*det(F)
    e_phase2 = 3/(2*sqrt(2))*sigma2*(1/(4*eps)*(1-psi2**2)**2 + eps/2*inner(gradpsi2, gradpsi2))*det(F)
    e_phase3 = 3/(2*sqrt(2))*sigma3*(1/(4*eps)*(1-psi3**2)**2 + eps/2*inner(gradpsi3, gradpsi3))*det(F)
    
    e_phase = (e_phase1 + e_phase2 + e_phase3)
    E = (e_elastic + e_phase + e_comp)*r[0]*dx
        
    # Finite difference time derivatives
    dot_v = (v-old_v) / tau
    if MOBILITY:
        dot_psi1 = (psi1-old_psi1) / tau
        dot_psi2 = (psi2-old_psi2) / tau

    er = Constant((1,0))

    # Build residual:y
    Res  = derivative(E, q, dq)
    Res += mu*inner(vector_grad_axi(dot_v)*inv(old_F), vector_grad_axi(dv)*inv(old_F))*det(old_F)*r[0]*dx # mu = mu(psii) = 0/sehr klein in der Luft

    print("TYPE ", type(Res))


    if MOBILITY:
        Res -= eta1*dpsi1*r[0]*dx 
        Res -= eta2*dpsi2*r[0]*dx
        Res += inner(dot_psi1, dfeta1)*r[0]*dx + m_CH*inner(inv(old_F).T*scalar_grad_axi(eta1), inv(old_F).T*scalar_grad_axi(dfeta1) )*r[0]*dx
        Res += inner(dot_psi2, dfeta2)*r[0]*dx + m_CH*inner(inv(old_F).T*scalar_grad_axi(eta2), inv(old_F).T*scalar_grad_axi(dfeta2) )*r[0]*dx


    # Solve
    bc      = get_BC(W)
    q.assign(old_q)


    dRes        = derivative(Res, q)    
    pde         = NonlinearVariationalProblem(Res, q, bc,dRes)
    solver      = NonlinearVariationalSolver(pde) 
    parameters["form_compiler"]["cpp_optimize"] = True
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-8
    prm['newton_solver']['relative_tolerance'] = 1E-7
    prm['newton_solver']['maximum_iterations'] = 25
    prm['newton_solver']['relaxation_parameter'] = 1.
    prm['newton_solver']["linear_solver"] = "mumps"
    prm         = solver.parameters
 
    solver.solve()

    E1 = assemble(e_phase*r[0]*dx)
    E2 = assemble(e_elastic*r[0]*dx)
    E3 = assemble(e_comp*r[0]*dx)
    E4 = E1+E2+E3
    return q,E1,E2,E3,E4


# run from initial data
t        = 0.0
m        = 0
psi11, psi22, initial   = get_initial_data(a, b, eps, h0, h1, H1)

n_0   = 0
mesh = Mesh()
f=HDF5File(mesh.mpi_comm(), filepath + 'initial.h5', 'r')
f.read(mesh,"mesh",False)  
W     = get_space(mesh)
old_q = Function(W)
f.read(old_q,"q")

if RADIAL:
    r = SpatialCoordinate(mesh)
else:
    r = Constant((1.0,1.0))

tau_initial = 0.5e-7

tau_adapt   = tau_initial
print("Start mini step")
q,E1,E2,E3,E4 = incremental_minimization(old_q, tau_initial)
t = t + tau_adapt

old_q.assign(q)



print("done mini step")
for n in range(n_0,n_steps+1):
    if (n < n_start):  
        tau_adapt = tau / 10
    else:
        tau_adapt = tau   

    q,E1,E2,E3,E4 = incremental_minimization(old_q, tau_adapt)
    t += tau_adapt
    old_q.assign(q)

    if near(t%tau0,0,tau/100):
        m += 1
        output_state(mesh,old_q,filepath + 'state_'+str(m), t)
        output_PV(file1, mesh, old_q, t)
    print("Iteration: ", n, " of: ", n_steps," time: ", t, " Energy: ", E1,E2,E3,E4)

