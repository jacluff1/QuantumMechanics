import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy.physics.quantum.constants import hbar
from sympy.solvers import solve
import pdb
from djak.gen import directory_checker

pr  =   sy.pprint

#===============================================================================
""" numerical constants """
#===============================================================================

const           =   {}
const['a']      =   1   # sides of infinite square well
const['mu']     =   1   # mass of particle
const['hbar']   =   1   # reduced planck constant

#===============================================================================
""" CC """
#===============================================================================

X   =   np.linspace(0, const['a'],100)
Y   =   np.linspace(0, const['a'],100)
X,Y =   np.meshgrid(X,Y)

#===============================================================================
""" symbolic constants and variables """
#===============================================================================

# system constants
a,mu    =   sy.symbols('a mu', real=True, positive=True)

# general constants
C,C0    =   sy.symbols('C C_0', real=True, positive=True)

# variables
x,y,t   =   sy.symbols('x y t', real=True, positive=True)

# indecies
n,n1,m,m1   =   sy.symbols('n n_1 m m_1', integer=True, positive=True)

#===============================================================================
""" auxillary functions """
#===============================================================================

def E(n):
    return (n * sy.pi * hbar)**2 / (2*mu*a**2)

def omega(n,m):
    ans =   (E(n) + E(m))/hbar
    ans =   ans.subs({hbar:const['hbar'], mu:const['mu'], a:const['a']})
    return ans.evalf()

def k(n):
    return n * sy.pi / a

def U(n,t):
    return sy.exp(-sy.I * t * E(n) / hbar)

def fX(n,t):
    return sy.sqrt(2/a) * sy.sin( k(n) * x ) * U(n,t)

def fY(m,t):
    return sy.sqrt(2/a) * sy.sin( k(m) * y ) * U(m,t)

def psi_nmt(n,m,t):
    return C * fX(n,t) * fY(m,t)

def solve_boundary(psi0):
    print("normalizing ic function...")
    # normalize and solve for C0
    rhs =   1
    lhs =   psi0.conjugate() * psi0
    lhs =   sy.integrate( lhs , (x,0,a) )
    lhs =   sy.integrate( lhs , (y,0,a) )
    eq  =   sy.Eq( lhs , rhs )
    C01 =   solve( eq , C0 )[0]
    psi0=   psi0.subs( C0 , C01 )
    print("solving boundary...")
    lhs =   psi0
    rhs =   psi_nmt(n,m,0)
    # use Fourier Trick for x and y
    lhs *=  sy.sin(k(n1)*x) * sy.sin(k(m1)*y)
    lhs =   sy.integrate( lhs , (x,0,a) )
    lhs =   sy.integrate( lhs , (y,0,a) )
    rhs *=  sy.sin(k(n1)*x) * sy.sin(k(m1)*y)
    rhs =   sy.integrate( rhs , (x,0,a) )
    rhs =   sy.integrate( rhs , (y,0,a) )
    # solve for C
    eq  =   sy.Eq( lhs , rhs )
    eq  =   eq.subs({m1:m, n1:n})
    C1  =   solve( eq , C )[0]
    return sy.simplify( C1 )

def prob_nmt(C_nm,n1,m1,t1):
    print("making probability function, P(x,y)...")
    psi     =   psi_nmt(n,m,t)
    C_nm    =   C_nm.subs({n:n1, m:m1})
    psi     =   psi.subs( {C:C_nm, n:n1, m:m1, t:t1, a:const['a']} )
    p       =   psi.conjugate() * psi
    f       =   sy.lambdify( (x,y) , p )
    return f(X,Y)

def test_Cnm(Cnm,N):
    total = 0
    for n1 in N:
        for m1 in N:
            c = Cnm.subs({n:n1, m:m1})
            total += c.conjugate() * c
    print(total.evalf())

#===============================================================================
""" plots """
N       =   np.arange(1,6)
N1      =   2*N - 1
N2      =   np.arange(1,3)
psi0_1  =   C0 * x * (x-a) * y * (y-a)
psi0_2  =   C0 * (fX(1,0)*fY(1,0) + fX(2,0)*fY(2,0))
#===============================================================================

def IC1(single=True,multiple=True,color=cm.hot):

    print("\nbeginning initial condition, example 1")
    print("f(x,y,t=0) =")
    pr(psi0_1)

    Cnm     =   solve_boundary(psi0_1)
    parent  =   'IC1/'
    directory_checker(parent)

    def single_frequencies():

        print("\nbeginning single frequencies")
        home    =   parent + 'single_frequencies/'
        directory_checker(home)

        def plot(n1,m1):

            Z   =   prob_nmt(Cnm,n1,m1,0)

            plt.close('all')
            fig =   plt.figure()
            ax  =   fig.gca()
            ax.set_title('n,m: %s,%s' % (n1,m1) )
            ax.set_xlabel('a')
            ax.set_ylabel('a')
            ax.set_aspect(1)
            cf  =   ax.contourf(X,Y,Z, 100, cmap=color)
            fig.colorbar(cf, cmap=color)

            name    =   home + 'n_%s_m_%s.png' % (n1,m1)
            fig.savefig(name)
            print("saved figure to %s" % name)

        for n1 in N1:
            for m1 in N1:
                plot(n1,m1)

    def multiple_frequencies():

        print("\nbeginning multiple frequencies")
        Z   =   np.zeros_like(X)
        for n1 in N1:
            for m1 in N1:
                Z   +=  prob_nmt(Cnm,n1,m1,0)

        plt.close('all')
        fig =   plt.figure()
        ax  =   fig.gca()
        ax.set_title('First 10 Odd n and m')
        ax.set_xlabel('a')
        ax.set_ylabel('a')
        ax.set_aspect(1)
        cf  =   ax.contourf(X,Y,Z, 100, cmap=color)
        fig.colorbar(cf, cmap=color)

        name    =   parent + 'multiple_frequencies.png'
        fig.savefig(name)
        print("saved figure to %s" % name)

    if single: single_frequencies()
    if multiple: multiple_frequencies()

def IC2(single=True,multiple=True,color=cm.hot):

    print("\nbeginning initial condition, example 2")
    print("f(x,y,t=0) =")
    pr(psi0_2)

    Cnm     =   solve_boundary(psi0_2)
    parent  =   'IC2/'
    directory_checker(parent)

    def single_frequencies():

        print("\nbeginning single frequencies")
        home    =   parent + 'single_frequencies/'
        directory_checker(home)

        def plot(n1,m1):

            Z   =   prob_nmt(Cnm,n1,m1,0)

            plt.close('all')
            fig =   plt.figure()
            ax  =   fig.gca()
            ax.set_title('n,m: %s,%s' % (n1,m1) )
            ax.set_xlabel('a')
            ax.set_ylabel('a')
            ax.set_aspect(1)
            cf  =   ax.contourf(X,Y,Z, 100, cmap=color)
            fig.colorbar(cf, cmap=color)

            name    =   home + 'n_%s_m_%s.png' % (n1,m1)
            fig.savefig(name)
            print("saved figure to %s" % name)

        for n1 in N2:
            plot(n1,n1)

    def multiple_frequencies():

        print("\nbeginning multiple frequencies")
        Z   =   np.zeros_like(X)
        for n1 in N2:
            Z   +=  prob_nmt(Cnm,n1,n1,0)

        plt.close('all')
        fig =   plt.figure()
        ax  =   fig.gca()
        ax.set_title('n = m = 1 and n = m = 2')
        ax.set_xlabel('a')
        ax.set_ylabel('a')
        ax.set_aspect(1)
        cf  =   ax.contourf(X,Y,Z, 100, cmap=color)
        fig.colorbar(cf, cmap=color)

        name    =   parent + 'multiple_frequencies.png'
        fig.savefig(name)
        print("saved figure to %s" % name)

    if single: single_frequencies()
    if multiple: multiple_frequencies()
