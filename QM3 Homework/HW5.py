import sympy as sy
import numpy as np
from sympy.physics.quantum.constants import hbar
from sympy.solvers import solve

from sympy import I,exp,sin,sinh

def pl(expr):
    print(sy.latex(expr))

# symbols
E,delta1,delta0,k1,k2,k3,m,r,r0,r1,V0,V1 =   sy.symbols('E delta_1 delta_0 k_1 k_2 k_3 m r r_0 r_1 V_0 V_1', real=True, positive=True)
A1,A2,A3    =   sy.symbols('A_1 A_2 A_3')

# constants
E1,m1,V01,V11   =   2,1,5,3

# wavenumbers^2
k1squared   =   2*m*(E+V1) / hbar**2
k2squared   =   2*m*(V0-E) / hbar**2
k3squared   =   2*m*E / hbar**2

# wave equations
u1  =   A1 * sin(k1*r + delta1)
u2  =   A2 * sinh(k2*r + delta0)
u3  =   A3 * sin(k3*r)

# wave equation derivatives
u1d =   sy.simplify( u1.diff(r) )
u2d =   sy.simplify( u2.diff(r) )
u3d =   sy.simplify( u3.diff(r) )

# boundary conditions
bc1 =   u1d/u1
bc2 =   u2d/u2
bc3 =   u3d/u3

e1  =   sy.Eq( sy.simplify(bc1) , sy.simplify(bc2) )
e2  =   sy.Eq( sy.simplify(bc2) , sy.simplify(bc3) )
