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
