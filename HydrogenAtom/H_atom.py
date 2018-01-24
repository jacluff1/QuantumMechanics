import numpy as np
import sympy as sy
import scipy as sci
import scipy.special as ss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import pdb
from djak.gen import directory_checker
from sympy.solvers import solve

pr  =   sy.pprint

#===============================================================================
""" constants """
#===============================================================================

const   =   {'a0':5.29177e-11}

#===============================================================================
""" variables """
#===============================================================================

# quantum numbers
n,l =   sy.symbols('n l', integer=True, positive=True)
m   =   sy.symbols('m', integer=True)

# bohr radius
a0  =   sy.symbols('a_0', real=True, positive=True)

# varibales
r,theta,phi     =   sy.symbols('r theta phi', real=True, positive=True)

# dummy variable
x   =   sy.symbols('x', real=True, positive=True)

#===============================================================================
""" radial equation """
#===============================================================================

def laguerre(k,r):
    ans     =   sy.exp(r) * sy.diff( r**k * sy.exp(-r) , r , k)
    return sy.simplify(ans)

def ass_laguerre(k,n,r):
    ans     =   sy.diff( laguerre(k,r) , r, n )
    return sy.simplify(ans)

def R_nl(n,l):
    assert n > l, "n must be greater than l"
    N_nl    =   -(2/(n*a0))**sy.Rational(3,2) * sy.sqrt( sy.factorial(n-l-1) / (2*n*sy.factorial(n+l)**3) )
    R_nl    =   sy.simplify( N_nl * (2*r/(n*a0))**l * sy.exp(-r/(n*a0)) * ass_laguerre(n+l,2*l+1,x) )
    ans     =   R_nl.subs( x , 2*r/(n*a0) )
    return ans

#===============================================================================
""" spherical harmonics """
#===============================================================================

def legendre(l):
    ans =   1/(2**l * sy.factorial(l) ) * sy.diff( (x**2 - 1)**l , x , l )
    # ans =   ans.subs(x, sy.cos(theta) )
    return ans

def ass_legendre(l,m):
    # m   =   abs(m)
    ans =   (1-x**2)**(m/2) * sy.diff( legendre(l), x, m )
    ans =   ans.subs( x , sy.cos(theta) )
    return sy.simplify(ans)

def Y_lm(l,m):
    assert l >= abs(m), "l must be greater than or equal to |m|"
    C_lm    =   (-1)**m * sy.sqrt( (2*l+1)/(4*sy.pi) * sy.factorial(l-m) / sy.factorial(l+m) )
    ans     =   C_lm * ass_legendre(l,m) * sy.exp(sy.I*m*phi)
    return ans

#===============================================================================
""" auxillary functions """
#===============================================================================

def find_exp_val_r(n,l,a01=1):
    # find expectation value
    Rexp    =   R_nl(n,l)
    Rexp    =   Rexp.subs( a0, a01 )
    rval    =   Rexp.conjugate() * r * Rexp * r**2
    rval    =   sy.integrate( rval , (r,0,sy.oo) )
    return np.float(rval)

#===============================================================================
""" ploting functions """
#===============================================================================

def plot_radials(a01=1):

    home    =   'radial_plots/'
    directory_checker(home)

    def plot_radial(n,l):
        R   =   R_nl(n,l)
        R   =   R.subs( a0 , a01)
        f   =   sy.lambdify( r , R )

        rmax    =   3 * find_exp_val_r(n,l)
        X   =   np.linspace(0,rmax,100)
        Y   =   f(X)

        plt.close('all')
        fig = plt.figure()
        plt.title('R$_{%s,%s}$' % (n,l) )
        plt.xlabel('a$_0$')
        plt.xlim([0,rmax])
        plt.plot(X,Y)
        fig.savefig(home + 'n_%s_l_%s.png' % (n,l) )
        plt.close()

    for n in np.arange(1,4):
        for l in np.arange(0,n):
            plot_radial(n,l)

def plot_probabilities_xy(nmax=5,a01=1,theta1=np.pi/2,color=cm.hot):

    home    =   'probability states_xy/'
    directory_checker(home)
    plt.close('all')

    def plot_prob_nlm_z(n,l,m):

        # SPC
        scale   =   2
        size    =   100
        rval    =   find_exp_val_r(n,l,a01=a01)
        R       =   np.linspace(0,rval*scale,size)
        PHI     =   np.linspace(0,2*np.pi,size)
        R,PHI   =   np.meshgrid(R,PHI)

        def prob():
            psi =   R_nl(n,l) * Y_lm(l,m)
            psi =   psi.subs( a0 , a01 )
            psi =   psi.subs( theta , theta1 )
            p   =   psi.conjugate() * psi
            p   =   sy.lambdify( (r,phi) , p )
            ans =   p(R,PHI)
            return  ans.real

        # set up CC
        X       =   R * np.sin(theta1) * np.cos(PHI)
        Y       =   R * np.sin(theta1) * np.sin(PHI)
        Z       =   prob()

        # make figure
        plt.close('all')
        fig =   plt.figure()
        ax  =   fig.gca(projection='polar')
        ax.set_aspect(1)
        ax.set_title('|$\psi_{%s,%s,%s}(r,\\theta=\pi/2,\\varphi)|^2$' % (n,l,m) )
        ax.tick_params(axis='both', labelsize=0, gridOn=False)
        cf  =   ax.contourf(PHI,R,Z, 100, cmap=color)
        fig.colorbar(cf, cmap=color)
        fig.savefig(home+'n_%s_l_%s_m_%s.png' % (n,l,m) )
        return fig,ax

    N   =   np.arange(1,nmax+1)
    for n1 in N:
        L   =   np.arange(0,n1)
        for l1 in L:
            M   =   np.arange(0,l1+1)
            for m1 in M:
                print(n1,l1,m1)
                fig,ax  =   plot_prob_nlm_z(n1,l1,m1)
                ax.tick_params(gridOn=False)
                fig.savefig(home+'n_%s_l_%s_m_%s.png' % (n1,l1,m1) )

def plot_probabilities_yz(nmax=5,a01=1,phi1=np.pi/2,color=cm.hot):

    home    =   'probability states_yz/'
    directory_checker(home)
    plt.close('all')

    def plot_prob_nlm_z(n,l,m):

        # SPC
        scale   =   2
        size    =   100
        rval    =   find_exp_val_r(n,l,a01=a01)
        R       =   np.linspace(0,rval*scale,size)
        THETA   =   np.linspace(0,2*np.pi,size)
        R,THETA =   np.meshgrid(R,THETA)

        def prob():
            psi =   R_nl(n,l) * Y_lm(l,m)
            psi =   psi.subs( a0 , a01 )
            psi =   psi.subs( phi , phi1 )
            p   =   psi.conjugate() * psi
            p   =   sy.lambdify( (r,theta) , p )
            ans =   p(R,THETA)
            return  ans.real

        # set up CC
        Z       =   R * np.sin(phi1) * np.cos(THETA)
        Y       =   R * np.sin(phi1) * np.sin(THETA)
        X       =   prob()

        # make figure
        plt.close('all')
        fig =   plt.figure()
        ax  =   fig.gca(projection='polar')
        ax.set_aspect(1)
        ax.set_title('|$\psi_{%s,%s,%s}(r,\\theta,\\varphi=\pi/2)|^2$' % (n,l,m) )
        ax.tick_params(axis='both', labelsize=0, gridOn=False)
        cf  =   ax.contourf(THETA,R,X, 100, cmap=color)
        fig.colorbar(cf, cmap=color)
        fig.savefig(home+'n_%s_l_%s_m_%s.png' % (n,l,m) )
        return fig,ax

    N   =   np.arange(1,nmax+1)
    for n1 in N:
        L   =   np.arange(0,n1)
        for l1 in L:
            M   =   np.arange(0,l1+1)
            for m1 in M:
                print(n1,l1,m1)
                fig,ax  =   plot_prob_nlm_z(n1,l1,m1)
                ax.tick_params(gridOn=False)
                fig.savefig(home+'n_%s_l_%s_m_%s.png' % (n1,l1,m1) )

def plot_constant_probabilities(prob_const=.5,nmax=5):

    home    =   'constant_probabilities/'
    directory_checker(home)

    plt.close('all')

    THETA1  =   np.linspace(0,np.pi/2,100)
    THETA2  =   np.linspace(np.pi/2,np.pi,100)
    PHI1    =   np.linspace(0,2*np.pi,100)
    PHI2    =   np.array(PHI1,copy=True)

    T1,P1   =   np.meshgrid(THETA1,PHI1)
    T2,P2   =   np.meshgrid(THETA2,PHI2)

    def plot_constant_probability(n1,l1,m1):

        def solve_r():

            psi     =   R_nl(n1,l1) * Y_lm(l1,m1)
            prob    =   psi.conjugate() * psi
            eq      =   sy.Eq( prob_const , prob )
            r1      =   solve( eq , r )[0]

            def func(theta1,phi1):
                # r2  =   r1.subs( {theta:theta1, phi:phi1, a0:1} )
                r2  =   r1.subs(theta,theta1)
                r3  =   r2.subs(phi,phi1)
                r4  =   r3.subs(a0,1)
                pdb.set_trace()
                return r4.evalf()

            print("starting R1...")
            R1  =   func(T1,P1)
            print("starting R2...")
            R2  =   func(T2,P2)

            return R1,R2

        R1,R2   =   solve_r()
        # pdb.set_trace()

        X1  =   R1 * np.sin(T1) * np.cos(P1)
        Y1  =   R1 * np.sin(T1) * np.sin(P1)
        Z1  =   R1 * np.cos(T1)

        X2  =   R2 * np.sin(T2) * np.cos(P2)
        Y2  =   R2 * np.sin(T2) * np.sin(P2)
        Z2  =   R2 * np.cos(T2)

        pdb.set_trace()

        fig     =   plt.figure()
        ax      =   fig.gca(projection='3d')

        ax.plot_surface(X1,Y1,Z1, color='b', alpha = .5)
        ax.plot_surface(X2,Y2,Z2, color='b', alpha = .5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        fig.savefig(home+'n_%s_l_%s_m_%s.png' % (n1,l1,m1) )
        plt.close()
        return

    N   =   np.arange(1,nmax+1)
    for n1 in N:
        L   =   np.arange(1,n1)
        for l1 in L:
            M   =   np.arange(1,l1+1)
            for m1 in M:
                print(n1,l1,m1)
                plot_constant_probability(n1,l1,m1)
