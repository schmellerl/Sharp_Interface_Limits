#H              = 10          # scale in m
eps            = 7.5e-03/2   

h1             = 0.5       # Droplet Radius
h0             = 1.0       # Substrat height
H1             = 1.0       # Domain width
H2             = 2.0       # Domain height

a              = 1.0
b              = 1.0

G1             = 10.0      # 10
G2             = 0.0      # 
G3             = 0.0      # 
sigma1         = 0.5      # solig    (divided by G*H) mN/m
sigma2         = 2.5       # liquid   (divided by G*H) mN/m
sigma3         = 3.5       # air      (divided by G*H) mN/m

ALPHA          = 1.0

mu             = 1.00           # Stokes Dissipation
kappa          = 1e+4           # compressibility
m_CH           = eps**(ALPHA) #5.8e-5 #2.5e-6           # Cahn-Hilliard Mobility

# Output
n_start        = 20             # number of warm-up steps
n_out          = 10             # frequency of output
filepath       = 'output/'

# Solver
NEWTON_tol     = 1e-7
tol            = 1e-9

# Discretization
FEu_DEG        = 2
FEp_DEG        = 1
FEs_DEG        = 2
FE_quad        = 4
N_refine       = 5
N_refine_solid = 0
N_refine_cl    = 5
DELTA0         = 0.5 #6.0
DELTAFAC       = 0.6 #0.5
Nx             = 16
Ny             = 16

tau            = 0.005
tau0           = 0.050

n_steps        = 250
max_level      = 9
ceps           = 0.85

# Switches
INCOMPRESSIBLE = False
MOBILITY       = True
RADIAL         = True
REFINE         = False
REFINE2        = False
