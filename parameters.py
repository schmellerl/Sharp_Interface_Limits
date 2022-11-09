#
# DEFINE PARAMETERS
#
r  = 2.0
h  = 5.0
H1 = 10.0
H2 = 10.0

# CONFLICTING THOSE

# Physics
sigma1         = 10.0    # solid/liquid
sigma2         = 20.0    # solid/air
sigma3         = 30.0    # liquid/air
mu             = 1.00 # Stokes Dissipation
kappa          = 1000.0
G0 = 0.01    # air
G1 = 0.01    # liquid
G2 = 200.0   # solid

# Output
n_start        = 20             # number of warm-up steps
n_out          = 10             # frequency of output
filepath       = 'output_Leonie/'

# Solver
NEWTON_tol     = 1e-7
tol            = 1e-9

# Discretization
FEu_DEG        = 2
FEp_DEG        = 1
FEs_DEG        = 2
FE_quad        = 5

tau            = 0.0005  
n_steps        = 250
max_level      = 9
ceps           = 0.85

# Switches
INCOMPRESSIBLE = False
MOBILITY       = False
RADIAL         = False

