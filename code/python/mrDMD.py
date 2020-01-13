import numpy as np
import numpy.linalg as npla

def sparse_mrDMD(Xraw, dt, r, max_cyc, L):
    T = Xraw.shape[1]*dt
    rho = max_cyc/T
    sub = int(np.ceil(1/rho/8/np.pi/dt))

    # DMD at current level
    Xaug = Xraw[:,np.array(range(0,Xraw.shape[-1],sub))]
    X = Xaug[:,:-1]
    Xp = Xaug[:,1:]
    Xaug = np.vstack([X,Xp])
    X = Xaug[:,:-1]
    Xp = Xaug[:,1:]

    U,S,V = npla.svd(X, full_matrices=False)
    V=V.T
    r = np.min([U.shape[1],r])
    U_r = U[:,:r]
    S_r = S[:r]
    V_r = V[:,:r]

    Atilde = np.dot(np.conj(U_r.T), np.dot(Xp, np.dot(V_r, np.diag(1/S_r))))
    [D,W] = npla.eig(Atilde)
    lam = D
    Phi = np.dot(Xp, np.dot(V_r, np.dot(np.diag(1/S_r), W)))

    # compute power of modes
    Vand = np.zeros((r, X.shape[1]), dtype=np.complex64)
    for k in range(X.shape[1]):
        Vand[:,k] = np.power(lam,k)

    # the next 5 lines follow Jovanovic et al, 2014 code:
    G = np.dot(np.diag(S_r), np.conj(V_r.T))
    P = np.dot(np.conj(W.T),W)*np.conj(np.dot(Vand,np.conj(Vand.T)))
    q = np.conj(np.diag(np.dot(Vand,np.dot(np.conj(G.T),W))))
    Pl = npla.cholesky(P)
    b = np.dot(P.T,npla.inv(Pl/q))

    # consolidate slow modes, abs(omega) < rho
    omega = np.log(lam)/sub/dt/2/np.pi
    mymodes = np.where(np.abs(omega)<=rho)[0]

    ret = {'T':T,'rho':rho,'hit':len(mymodes)>0,'omega':omega,'P':abs(b[mymodes]),'Phi':Phi[:,mymodes]}

    children = []

    if L>1:
        sep = int(np.floor(Xraw.shape[1]/2.))
        children.append( mrDMD(Xraw[:,:sep], dt, r, max_cyc, L-1) )
        children.append( mrDMD(Xraw[:,sep:], dt, r, max_cyc, L-1) )

    ret['children'] = children
    return(ret)


def mrDMD(Xraw, dt, r, max_cyc, L):
    T = Xraw.shape[1]*dt
    rho = max_cyc/T
    sub = int(np.ceil(1/rho/8/np.pi/dt))

    # DMD at current level
    Xaug = Xraw[:,np.array(range(0,Xraw.shape[-1],sub))]
    X = Xaug[:,:-1]
    Xp = Xaug[:,1:]

    U,S,V = npla.svd(X, full_matrices=False)
    V=V.T
    r = np.min([U.shape[1],r])
    U_r = U[:,:r]
    S_r = S[:r]
    V_r = V[:,:r]

    Atilde = np.dot(np.conj(U_r.T), np.dot(Xp, np.dot(V_r, np.diag(1/S_r))))
    [D,W] = npla.eig(Atilde)
    lam = D
    Phi = np.dot(Xp, np.dot(V_r, np.dot(np.diag(1/S_r), W)))

    b = np.dot(npla.pinv(Phi),Xraw)

    # consolidate slow modes, abs(omega) < rho
    omega = np.log(lam)/sub/dt/2/np.pi
    slow_modes = np.where(np.abs(omega)<=rho)[0]
    fast_modes = np.where(np.abs(omega)>rho)[0]

    ret = {'T':T,
           'rho':rho,
           'hit':len(slow_modes)>0,
           'b':b[slow_modes],
           'omega':omega[slow_modes],
           'Phi':Phi[:,slow_modes]}

    children = []

    if L>1:
        print(L)
        Xraw_fast = np.dot(Phi[:,fast_modes], np.dot(np.diag(np.exp(omega[fast_modes])), b[fast_modes,:]))
        sep = int(np.floor(Xraw_fast.shape[1]/2.))
        children.append( mrDMD(Xraw_fast[:,:sep], dt, r, max_cyc, L-1) )
        children.append( mrDMD(Xraw_fast[:,sep:], dt, r, max_cyc, L-1) )

    ret['children'] = children
    return(ret)

