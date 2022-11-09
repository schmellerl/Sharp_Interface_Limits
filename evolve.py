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
