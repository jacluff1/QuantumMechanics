import sympy as sy
from sympy import *
# from sympy import sin,cos,exp,cot,I,Function,diff,simplify,hermite, sqrt
from sympy.physics.quantum.constants import hbar
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

matrix  =   sy.Matrix
x,y,z   =   sy.symbols('x y z', positive=True)
x0      =   sy.symbols('x_0',positive=True)
t       =   sy.symbols('t', positive=True)
omega   =   sy.Symbol('omega', positive=True)
omega0  =   sy.Symbol('omega_0', positive=True)
omega1  =   sy.Symbol('omega_1', positive=True)
pi      =   sy.symbols('pi', positive=True)
tau     =   sy.Symbol('tau', positive=True)
alpha   =   sy.Symbol('alpha', positive=True)
beta    =   sy.Symbol('beta', positive=True)
gamma   =   sy.Symbol('gamma', positive=True)
c1,c2   =   sy.symbols('c_1 c_2', positive=True)
q,m,E   =   sy.symbols('q m E', positive=True)
epsilon =   sy.symbols('epsilon', positive=True)
nf,ni   =   sy.symbols('n_f n_i', positive=True)
ID      =   sy.eye(2)


def herm(M):
    return M.conjugate().T

def expval(M,bra,ket):
    return herm(bra) * M * ket

def omega_fi(H0,psi_f,psi_i):
    return (1/hbar) * ( expval(H0,psi_f,psi_f) - expval(H0,psi_i,psi_i) )

def P_if_perturb(H0,V,psi_f,psi_i):
    omega       =   omega_fi(H0,psi_f,psi_i)
    exponent    =   exp(I*omega*t)
    expectation =   expval(V,psi_f,psi_i)
    integrand   =   expectation * exponent
    integral    =   sy.integrate(integrand,t)
    val         =   (-I/hbar) * integral
    return val.conjugate() * val

def printme(results):
    for key in results:
        print(key, results[key])

def D_n(n,x):
    return 2**(-n/2) * exp(-x**2 / 4) * hermite(n,x/sqrt(2))

# pauli matricies
sigx    =   matrix([ [0,1],[1,0] ])
sigy    =   matrix([ [0,-I],[I,0] ])
sigz    =   matrix([ [1,0],[0,-1] ])

# spin matricies
Sx      =   (hbar/2) * sigx
Sy      =   (hbar/2) * sigy
Sz      =   (hbar/2) * sigz

def problem_1():
    alpha   =   omega * t / 2

    # state vectors
    psif    =   matrix([ [0],[1] ]) # final
    psii    =   matrix([ [1],[0] ]) # initial

    H0      =   sy.simplify(omega0 * Sz)
    V       =   sy.simplify(omega1 * Sx * exp(-t/tau))
    omega   =   sy.simplify(omega_fi(H0,psif,psii))
    P       =   sy.simplify(P_if_perturb(H0,V,psif,psii))

    f       =   sy.lambdify([t,tau,omega0,omega1],P)
    X       =   np.linspace(0,10,1000)
    Y       =   f(X,1,1000,1)

    fig = plt.figure()
    plt.title('$\\tau$ = 1, $\omega_0$ = 1000, $\omega_1$ = 1')
    plt.ylabel('P$_{fi}$(t)')
    plt.xlabel('time')
    plt.plot(X,Y[0,0],linewidth=2)
    fig.savefig('fig1.png')

    results =   {'psif':psif, 'psii':psii, 'H0':H0, 'V':V, 'omega':omega, 'P':P}
    return results

def problem_2():

    a           =   sqrt( m*omega / (pi * hbar) )
    b           =   exp( -(m*omega/(2*hbar)) * ( ( x-x0)**2 -x**2 ) )
    c           =   integrate(b,(x,-oo,oo))
    ans         =   abs(a*c)**2

    result      =   {'a':a, 'b':b, 'c':c, 'ans':ans}
    return result

def problem_3():

    a       =   exp(-(t-alpha)**2 / tau**2)
    b       =   integrate(a,(t,0,oo))

    result  =   {'alpha':alpha, 'a':a, 'b':b}
    return result
