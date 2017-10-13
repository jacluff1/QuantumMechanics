import sympy as sy
import numpy as np
from sympy.physics.quantum.constants import hbar

#===============================================================================
""" add to Hydrogen Atom module """
#===============================================================================

r,theta,phi =   sy.symbols('r theta phi', positive=True)
a0          =   sy.symbols('a_0', positive=True)
alpha       =   sy.symbols('alpha', positive=True)
e,c         =   sy.symbols('e c', positive=True)
E1          =   sy.symbols('E_1', positive=True)

def laguerre(r,k):
    ans =   sy.exp(r) * sy.diff( r**k * sy.exp(-r) , r , k)
    return sy.simplify(ans)

def laguerre_ass(r,k,n):
    ans =   sy.diff( laguerre(r,k) , r, n )
    return sy.simplify(ans)

def R(r,n,l):

    assert n != 0, "n must be greater than 0"
    assert n > l, " n must be larger than l"

    c1      =   2/(n*a0)
    f1      =   -(c1)**(3/2)
    f2      =   sy.sqrt( sy.factorial(n-l-1) / (2*n*sy.factorial(n+l)**3) )
    f3      =   (alpha)**l
    f4      =   sy.exp(-alpha/2)
    f5      =   laguerre_ass( alpha , n+l , 2*l+1)
    ans     =   f1*f2*f3*f4*f5
    return sy.simplify(ans)

#===============================================================================

def prob1():
    # problem 1 part b
    L,x = symbols('L x', positive=True)

    a   =   2/(L * sy.sqrt(8))
    b   =   sy.sin( sy.pi*x / (8*L) ) * sy.sin( sy.pi*x / L )
    c   =   sy.integrate(b,x)
    d   =   sy.integrate(b,(x,0,L))
    e   =   sy.simplify(a*d)
    f   =   abs(e)**2

    return {'a':a, 'b':b, 'c':c, 'd':d, 'e':e, 'f':f}

def prob2():
    from sympy.physics.quantum.constants import hbar
    a,c,m,q,x   =   sy.symbols('a c m q x', positive=True,real=True)
    n1,n2       =   sy.symbols('n_1 n_2', integer=True, positive=True)

    omega       =   sy.pi**2 * hbar / (2 * m * a**2) * (n2**2 - n1**2)
    one1        =   2/a
    one2        =   sy.sin(n2*sy.pi*x/a) * x * sy.sin(n1*sy.pi*x/a)
    one3        =   sy.simplify(sy.integrate(one2,(x,0,a)))
    one         =   sy.simplify(one1 * one3)
    two         =   sy.simplify(sy.Rational(4,3) * omega**3 * q**2 / (hbar * c**3))
    three       =   sy.simplify(one**2 * two)
    return {'a': omega, 'b':one1, 'c':one2, 'd':one3, 'e':one, 'f':two, 'g':three}

def triple_integral(psi):
    psi1    =   sy.integrate(psi,(r,0,sy.oo))
    psi2    =   sy.integrate(psi1,(theta,0,sy.pi))
    psi3    =   sy.integrate(psi2,(phi,0,2*sy.pi))
    if psi1 == 0: print("r integral = 0")
    if psi2 == 0: print("theta integral = 0")
    if psi3 == 0: print("phi integral = 0")
    return sy.simplify(psi3)

R21 =   1/sy.sqrt(6*a0**3) * r/(2*a0) * sy.exp(-r/(2*a0))
R10 =   2/sy.sqrt(a0**3) * sy.exp(-r/a0)

Y00     =   1/sy.sqrt(4*sy.pi)
Y10     =   sy.sqrt(3/(4*sy.pi)) * sy.cos(theta)
Y11     =   -sy.sqrt(3/(8*sy.pi)) * sy.exp(sy.I*phi) * sy.sin(theta)
Y1n1    =   sy.sqrt(3/(8*sy.pi)) * sy.exp(-sy.I*phi) * sy.sin(theta)

dx      =   e*r*sy.sin(theta)*sy.cos(phi)
dy      =   e*r*sy.sin(theta)*sy.sin(phi)
dz      =   e*r*sy.cos(theta)

dV      =   r**2 * sy.sin(theta)

psii1   =   R21 * Y10
psii2   =   R21 * Y11
psii3   =   R21 * Y1n1
psif    =   R10 * Y00

one     =   dV * psii1.conjugate() * psif
onex    =   one * dx
oney    =   one * dy
onez    =   one * dz

two     =   dV * psii2.conjugate() * psif
twox    =   two * dx
twoy    =   two * dy
twoz    =   two * dz

three   =   dV * psii3.conjugate() * psif
threex  =   three * dx
threey  =   three * dy
threez  =   three * dz

d210    =   sy.Rational(128,243) * sy.sqrt(2) * e * a0
d211    =   sy.Rational(128,243) * e * a0 * (-1 + sy.I)
d21n1   =   sy.Rational(128,243) * e * a0 * (1 + sy.I)

omega   =   sy.Rational(3,4) * E1/hbar
factor  =   sy.Rational(4,3) * omega**3 / (hbar * c**3)

W210    =   factor * d210.conjugate() * d210 * alpha/(e**2/(hbar*c))
W211    =   factor * d211.conjugate() * d211 * alpha/(e**2/(hbar*c))
W21n1   =   factor * d21n1.conjugate() * d21n1 * alpha/(e**2/(hbar*c))

E1      =   13.606      # eV
h       =   4.14e-15    # eV s
hbar    =   h/(2*np.pi) # eV s
a0      =   0.529e-10   # m
c       =   2.9979e8    # m/s
alpha   =   1/137

d210    =   sy.Rational(128,243) * sy.sqrt(2) * e * a0
d211    =   sy.Rational(128,243) * e * a0 * (-1 + sy.I)
d21n1   =   sy.Rational(128,243) * e * a0 * (1 + sy.I)

omega   =   sy.Rational(3,4) * E1/hbar
factor  =   sy.Rational(4,3) * omega**3 / (hbar * c**3)

W210    =   sy.simplify( factor * d210.conjugate() * d210 * alpha/(e**2/(hbar*c)) )
W211    =   sy.simplify( factor * d211.conjugate() * d211 * alpha/(e**2/(hbar*c)) )
W21n1   =   sy.simplify( factor * d21n1.conjugate() * d21n1 * alpha/(e**2/(hbar*c)) )
