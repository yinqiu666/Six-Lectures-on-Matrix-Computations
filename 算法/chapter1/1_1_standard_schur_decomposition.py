import numpy as np
import math
#alogorithm1.1.1 compute Householder transformation
#given a column vector a(may equal to 0),find a vector w and a number gamma,
# such that H*a=gamma*e1,H=I-w*w_H,w=sqrt(2)in order 2.
def householder(a):
    eps = np.finfo(float).eps
    w=a.copy()
    gamma=np.linalg.norm(a,ord=2)
    if abs(gamma)<eps:
        w[0]=np.sqrt(2)
    if abs(w[0])>eps:
        delta=(w[0].conj())/w[0]
    else:
        delta=1
    w=(delta/gamma)*w
    w[0]=w[0]+1
    w=w/(np.sqrt(w[0]))
    gamma=-delta.conj()*gamma
    return w
#测试
"""
a=np.array([3,4],dtype=float)
w=householder(a)
H=np.eye(2)-np.outer(w,w.conj())
print("w=",w)
print("H=",H)
print("H@a=",H@a)
"""
#输出是
"""
w= [1.26491106 0.63245553]
H= [[-0.6 -0.8]
 [-0.8  0.6]]
H@a= [-5.00000000e+00 -2.22044605e-16]
"""
#==========================================================================
#alogrithm1.1.2 compute Givens transformation
#given 2 complex numbers alpha and beta ,find 3 numbers c,s,eta,such that
#(c s; -s* c*)*(alpha; beta)=(eta;0),where |c|**2+|s|**2=1
def givens(alpha,beta):
    alpha = np.asarray(alpha, dtype=np.complex128)
    beta = np.asarray(beta, dtype=np.complex128)
    eps=np.finfo(float).eps
    if np.abs(beta)<eps:
        c=1
        s=0
        eta=alpha
        return c,s,eta
    if np.abs(alpha)<eps:
        c=0
        s=1
        eta=beta
        return c,s,eta
    mu=alpha/np.sqrt(alpha*alpha.conj())
    tau=np.sqrt(alpha*alpha.conj())+np.sqrt(beta*beta.conj())
    delta=tau*np.sqrt(((alpha/tau)*(alpha/tau).conj())+((beta/tau)*(beta/tau).conj()))
    c=np.sqrt(alpha*alpha.conj())/delta
    s=mu*beta.conj()/delta
    eta=mu*delta
    return c,s,eta
#测试
"""
alpha = 3
beta = 4
c, s, eta = givens(alpha, beta)
G = np.array([[c, s], [-np.conj(s), c]])
print("G @ [alpha,beta]^T =", G @ np.array([alpha,beta]))
print(f"c={c}, s={s}, eta={eta}")
print("|c|^2+|s|^2 =", np.abs(c)**2 + np.abs(s)**2)
"""
#结果是
"""
G @ [alpha,beta]^T = [5.+0.j 0.+0.j]
c=(0.6000000000000001+0j), s=(0.8000000000000002+0j), eta=(4.999999999999999+0j)
|c|^2+|s|^2 = 1.0000000000000004
"""
#=============================================================================
#therom1.1.2:schur decomposition：
#A is a complex square matrix ,whose dimension is n,then there exists a unitary matrix U such that
#U_H*A*U=T=OMEGA+N,where OMEGA is a diagonal matrix whose diagonal elements are the eigenvalues of A,
#N is a strictly upper triangular matrix, and T is called the Schur form of A

#terom1.1.3 real schur decomposition:
#A is a real square matrix, whose dimension is n, then there exists a real orthogonal matrix Q such that
#Q_T*A*Q=T=OMEGA+N,where OMEGA is a block diagonal matrix whose diagonal blocks are either 1*1 or 2*2, 
#when 2*2,the eigenvalues are complex conjugate pairs.

#algorithm1.1.3 compute upper Hessenberg decomposition
#given a matrix A,whose dimension is n,find a unitary matrix U and a upper Hessenberg matrix H
#such that U_H*A*U=H
def upper_hessenberg(A):
    H=A.copy()
    n=H.shape[0]
    U=np.zeros((n,n), dtype=np.complex128)
    for k in range(n-2):
        w=householder(H[k+1:n,k])
        U[k+1:n,k]=w
        vh = np.conj(w).T @ H[k+1:n, k:n]
        H[k+1:n, k:n] = H[k+1:n, k:n] - np.outer(w, vh)
        v = H[:, k+1:n] @ w
        H[:, k+1:n] = H[:, k+1:n] - np.outer(v, np.conj(w).T)
    U[:, n-1] = np.eye(n)[:, n-1]
    U[:, n-2] = np.eye(n)[:, n-2]
    for k in range(n-2, -1, -1):
        w = U[k+1:n, k]
        vh = np.conj(w).T @ U[k+1:n, k+1:n]
        U[k+1:n, k+1:n] = U[k+1:n, k+1:n] - np.outer(w, vh)
        U[:,k] = np.eye(n)[:,k]
    return U, H
#测试
A = np.array([[1,2,3,4],
                  [5,6,7,8],
                  [9,10,11,12],
                  [13,14,15,16]], dtype=complex)
U, H = upper_hessenberg(A)
print("the upper hessenberg decomposition of A is:")
print(np.round(H, 8))
print("\ncompute U^*AU to test Hessenberg")
lhs = np.conj(U).T @ A @ U
print(np.round(lhs,8))
print("\ncompute U^*U to test unitarity")
print(np.round(np.conj(U).T @ U, 8))
