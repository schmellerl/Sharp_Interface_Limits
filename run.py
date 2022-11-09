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
