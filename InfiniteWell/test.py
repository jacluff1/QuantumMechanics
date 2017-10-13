import sympy as sy
import numpy as np
import InfiniteWell as iw

n,m     =   sy.symbols('n m',integer=True, positive=True)
a,A,k     =   sy.symbols('a A k', real=True,positive=True)
An,kn   =   sy.symbols('A_n k_n', real=True,positive=True)
x,t     =   sy.symbols('x t',real=True, positive=True)

psi_nt  =   A * sy.sin(k*x)
