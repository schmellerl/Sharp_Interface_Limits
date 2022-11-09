from dolfin import *
from mshr import *
from ufl import cofac
import ufl
import matplotlib.pyplot as plt
parameters["refinement_algorithm"] = "plaza_with_parent_facets"

# Define 2D geometry
domain = Rectangle(Point(0,0), Point(H1,H2))
domain.set_subdomain(1, Circle(Point(0, h), r))
domain.set_subdomain(2, Rectangle(Point(0,0),Point(H1,h)))

# Generate and plot mesh
mesh = generate_mesh(domain, 32)
tol = 0.05 #1e-1 #3

class interface_sl(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], h , tol) and x[0] < (r + tol)

class interface_sa(SubDomain):
    def inside(self, x, on_boundary):  
        return near(x[1], h , tol) and x[0] > (r - tol)

class interface_la(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0]**2+(x[1]-h)**2,r**2, tol) and x[1] > (h - tol)

# Convert subdomains to mesh function for plotting
subdomain_marker = MeshFunction("size_t", mesh,mesh.topology().dim()  ,mesh.domains())
interface_marker = MeshFunction("size_t", mesh,mesh.topology().dim()-1,0)

gamma_sl = interface_sl()
gamma_sa = interface_sa()
gamma_la = interface_la()

gamma_sl.mark(interface_marker, 1)
gamma_sa.mark(interface_marker, 2)
gamma_la.mark(interface_marker, 3)

p = Point(r,h)
R0 = 2.0

# Mark cells for refinement
for k in range(4):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for c in cells(mesh):
        if c.midpoint().distance(p) < R0:
            cell_markers[c] = True    
        else:
            cell_markers[c] = False
    R0 = R0/2
    mesh = refine(mesh, cell_markers)    
    subdomain_marker = adapt(subdomain_marker, mesh)
    interface_marker = adapt(interface_marker, mesh)
    
n  = FacetNormal(mesh)
dS = Measure("dS", domain=mesh, subdomain_data=interface_marker)
dx = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
R  = SpatialCoordinate(mesh)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = FE_quad

def get_space(mesh):
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

noslip   = Constant((0, 0)) 
noslip1d = Constant(0)

def boundary_bot1(x, on_boundary):
    return on_boundary and near(x[0], 0, tol)
def boundary_bot2(x, on_boundary):
    return on_boundary and near(x[1], 0, tol)
def boundary_bot3(x, on_boundary):
    return on_boundary and near(x[0], H1, tol)
def boundary_bot4(x, on_boundary):
    return on_boundary and near(x[1], H2, tol)

def get_BC(W):   
  if INCOMPRESSIBLE:
    bc1 = DirichletBC(W.sub(0).sub(0) , noslip1d , boundary_bot1)
    bc2 = DirichletBC(W.sub(0)        , noslip   , boundary_bot2)
    bc3 = DirichletBC(W.sub(0).sub(0) , noslip1d , boundary_bot3)
    bc4 = DirichletBC(W.sub(0)        , noslip   , boundary_bot4)
  else:
    bc1 = DirichletBC(W.sub(0), noslip1d, boundary_bot1)
    bc2 = DirichletBC(W       , noslip  , boundary_bot2)
    bc3 = DirichletBC(W.sub(0), noslip1d, boundary_bot3)
    bc4 = DirichletBC(W       , noslip  , boundary_bot4)      
  bc = [bc1,bc2,bc3,bc4]
  return bc

def scalar_grad_axi(u):
    if RADIAL:
        return as_vector([u.dx(0), 0, u.dx(1)])
    else:
        return grad(u)
  
def vector_grad_axi(v):
    if RADIAL:
        return as_tensor([[v[0].dx(0), 0, v[0].dx(1)], [0, v[0]/R[0], 0],
                                    [v[1].dx(0), 0, v[1].dx(1)]])
    else:
        return grad(v)
    
def incremental_minimization(old_q, tau):
    W = old_q.function_space()
    q, dq = Function(W), TestFunction(W)

    if INCOMPRESSIBLE:
        v, lambda1 = split(q)      # Current solution
        old_v,_    = split(old_q)  # Old solution
        dv,_       = split(dq)     # Test functions
    else:
        v     = q
        dv    = dq
        old_v = old_q

    if RADIAL: 
        d = 3
    else:
        d = 2

    I       = Identity(d)
    F       = I + vector_grad_axi(v)
    old_F   = I + vector_grad_axi(old_v)
    C       = F.T*F        # (right) Cauchy-Green tensor

    # Stored strain energy density (compressible neo-Hookean model)
    e1 = G1/2 * tr(C-I)*R[0] 
    e2 = G2/2 * tr(C-I)*R[0] 
    e3 = G3/2 * tr(C-I)*R[0] 

    if INCOMPRESSIBLE:
        e_comp = R[0]*lambda1*(det(F)-1)
    else:
        e_comp = R[0]*kappa*(det(F)-1)**2

    #############################
    # DEFINE PROBLEM via RESIDUAL
    #############################
    dn   = cofac(F)('+')*n('+')
    Area = sqrt(inner(dn,dn))
    E  = ( e1*dx(0) + e2*dx(1) + e3*dx(2) + e_comp*dx )
    E += sigma1 * Area * dS(1) 
    E += sigma2 * Area * dS(2) 
    E += sigma3 * Area * dS(3) 

    dot_v = (v-old_v) / tau

    Res  = derivative(E, q, dq)
    Res += mu*inner(vector_grad_axi(dot_v), vector_grad_axi(dv))*R[0]*dx

    # Solve
    bc      = get_BC(W)
    q.assign(old_q)

    solve(Res==0,q,bc)
    return q

# Initial flow conformant with boundary conditions 
t       = 0
dt      = tau
initial = Expression(( "0", "0"), degree=4, t=0.0)
W       = get_space(mesh)
old_q   = interpolate(initial, W)
R       = Constant((1.0,1.0))

old_q.rename("disp","disp")
subdomain_marker.rename("psi","psi")

file1 = File(filepath + "disp.pvd")
file2 = File(filepath + "ind.pvd")

file1 << (old_q,t)
file2 << (subdomain_marker,t)

for i in range(n_steps+1):
    print(i)
    t += dt
    q = incremental_minimization(old_q, dt)    
    old_q.assign(q)
    
    q.rename("disp","disp")
    subdomain_marker.rename("psi","psi")
    
    tmp = Function(q.function_space())
    tmp.assign(-q)
    ALE.move(mesh,q)
    file1 << (q,t)
    file2 << (subdomain_marker,t)
    ALE.move(mesh,tmp)
