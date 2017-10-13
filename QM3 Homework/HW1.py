import sympy as sy
from sympy import sin,cos,exp,cot,I
from sympy.physics.quantum.constants import hbar
import numpy as np

matrix  =   sy.Matrix
x,y,z   =   sy.symbols('x y z', positive=True)
t       =   sy.symbols('t', positive=True)
omega   =   sy.Symbol('omega', positive=True)
ID      =   sy.eye(2)
alpha   =   omega * t / 2

def herm(M):
    return M.conjugate().T

def comm(M1,M2):
    return M1*M2 - M2*M1

def heisenberg_eom(H,A):
    return (I/hbar) * comm(H,A)

def U_t(sig_i):
    return cos(alpha) * ID - I * sin(alpha) * sig_i


# pauli matricies
sigma_x =   matrix([ [0,1],[1,0] ])
sigma_y =   matrix([ [0,-I],[I,0] ])
sigma_z =   matrix([ [1,0],[0,-1] ])

# spin matricies
S_x     =   (hbar/2) * sigma_x
S_y     =   (hbar/2) * sigma_y
S_z     =   (hbar/2) * sigma_z

# hamiltonian of spin-1/2 particles
H       =   (omega * hbar / 2) * matrix([ [0,-I],[I,0] ])

# time evolution operator II.a
U       =   U_t(sigma_y)
# U       =   matrix([ [cos(alpha),-sin(alpha)],[sin(alpha),cos(alpha)] ])

# state vector II.c
statev  =   S_z.eigenvects()[1][2][0]
statevt =   U * statev

# expectation values of spin operators II.c
expx    =   herm(statevt) * S_x * statevt
expy    =   herm(statevt) * S_y * statevt
expz    =   herm(statevt) * S_z * statevt

# new U(t)
H1      =   (hbar * omega / 2) * matrix([ [1,0],[0,-1] ])
U1      =   sy.simplify(U_t(sigma_z))

# spin operators in Heisenberg picture III
S_xH    =   sy.simplify(herm(U1) * S_x * U1)
S_yH    =   sy.simplify(herm(U1) * S_y * U1)
S_zH    =   sy.simplify(herm(U1) * S_z * U1)

# heisenberg equation of motion III
eomx    =   heisenberg_eom(H1,S_xH)
eomy    =   heisenberg_eom(H1,S_yH)
eomz    =   heisenberg_eom(H1,S_zH)
