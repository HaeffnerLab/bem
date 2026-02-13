'''
Functions for performing the spherical
harmonic expansion of the potential
'''
import numpy as np
import math as mt
from scipy.special import sph_harm
from scipy import linalg#, matrix
import scipy
#import pyshtools

# Define multipole names up to 2nd order
NamesUptoOrder2 = ['C', 'Ey', 'Ez', 'Ex', 'U3', 'U4', 'U2', 'U5', 'U1']
PrintNamesUptoOrder2 = [r'$C$', r'$E_y$', r'$E_z$', r'$E_x$', r'$U3=xy$', 
                r'$U4=yz$', r'$U2=z^2-(x^2+y^2)/2$', r'$U5=zx$', r'$U1=x^2-y^2$']

# Normalization factors are taken into account up to 2nd order, see below.
NormsUptoOrder2 = np.array([np.sqrt(1/4/np.pi), np.sqrt(3/4/np.pi), np.sqrt(3/4/np.pi), np.sqrt(3/4/np.pi),
                                 np.sqrt(15/4/np.pi), np.sqrt(15/4/np.pi), np.sqrt(20/16/np.pi), 
                                 np.sqrt(15/4/np.pi), np.sqrt(15/16/np.pi)])

def harmonics_dict(n, theta, phi):
    '''
    like the matlab function, returns an array of all the assosciated legendre functions of degree n
    and order m = 0,1.... n for each element in X
    '''
    r = {}
    for m in range(-n, n+1):
        r[m] = sph_harm(m,n,theta,phi)
    return r

def spher_harm_basis(r0, X, Y, Z, order):
    '''
    Computes spherical harmonics, just re-written matlab code
   
    Returns: Yxx, rnorm
    where Yxx is a 1D array of the spherical harmonic evaluated on the grid
    rnorm is a normalization factor for the spherical harmonics

    The function returns the coefficients in the order:[C00 C10 C11c C11s ]'
    These correspond to the multipoles in cartesian coordinates: 
    ['C','Ey', 'Ez', 'Ex', 'U=xy', 'U4=yz', r'U2=z^2-(x^2+y^2)/2', 'U5=zx', r'U1=x^2-y^2']
     1    2     3     4      5        6               7               8           9  ..
    The normalization factors are dropped up to 2nd order.
    higher order terms ordering: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
     '''

    #initialize grid with expansion point (r0) at 0
    x0,y0,z0 = r0
    
    nx = len(X)
    ny = len(Y)
    nz = len(Z)
    npts = nx*ny*nz

    y, x, z = np.meshgrid(Y-y0,X-x0,Z-z0)
    x, y, z = np.reshape(x,npts), np.reshape(y,npts), np.reshape(z,npts)

    #change variables
    r = np.sqrt(x*x+y*y+z*z)
    r_trans = np.sqrt(x*x+y*y)
    phi = np.arctan2(r_trans,z)
    theta = np.arctan2(y,x)

    # for now normalizing as in matlab code
    dl = Z[1]-Z[0]
    scale = 1#np.sqrt(np.amax(r)*dl)
    rs = r/(scale)

    Q = []

    #real part of spherical harmonics
    for n in range(0,order+1):
        p = harmonics_dict(n, theta, phi)
        for m in range(-n,n+1):
            if m == 0:
                c = (rs**n)*p[0]
                Q.append(c.real)
            elif m < 0:
                c = 1j/np.sqrt(2) * (rs**n) * (p[m] - (-1)**m * p[-m])
                Q.append(c.real)
            elif m > 0:
                c = 1/np.sqrt(2) * (rs**n) * (p[-m] + (-1)**m * p[m])
                Q.append(c.real)

    Q = np.transpose(Q)

    return Q, scale

def spher_harm_expansion(potential_grid, r0, X, Y, Z, order):
    '''
    Compute the least-squares solution for the spherical harmonic expansion on potential_grid.
    Arguments:
    potential_grid: 3D array of potential values
    r0: list [x0, y0, z0] of the expansion point
    X, Y, Z: axis ranges for the potential grid
    order: int, order of the expansion
    '''
    # Convert the 3D DC potential into 1D array.
    # Numerically invert, here the actual expansion takes place and we obtain the expansion coefficients M_{j}.

    nx = len(X)
    ny = len(Y)
    nz = len(Z)
    npts = nx*ny*nz

    W=np.reshape(potential_grid,npts) # 1D array of all potential values
    W=np.array([W]).T # make into column array

    Yj, scale = spher_harm_basis(r0,X,Y,Z,order)
    n_norm_rows = np.min([Yj.shape[1], len(NormsUptoOrder2)])
    for row_ind in range(n_norm_rows):
        Yj[:,row_ind] = Yj[:,row_ind] / NormsUptoOrder2[row_ind]
    #Yj, rnorm = spher_harm_basis_v2(r0, X, Y, Z, order)

    Mj=np.linalg.lstsq(Yj,W,rcond=None) 
    Mj=Mj[0] # array of coefficients

    # rescale to original units
    i = 0
    for n in range(1,order+1):
        for m in range(1,2*n+2):
            i += 1
            Mj[i] = Mj[i]/(scale**n)
    return Mj,Yj,scale

def spher_harm_cmp(Mj,Yj,scale,order):
    '''
    regenerates the potential (V) from the spherical harmonic coefficients. 
    '''
    V = []
    #unnormalize
    i=0
    for n in range(1,order+1):
        for m in range(1,2*n+2):
            i += 1
            Mj[i] = Mj[i]*(scale**n)
    W = np.dot(Yj,Mj)
    return np.real(W)

def nullspace(A,eps=1e-15):
    u,s,vh = np.linalg.svd(A)
    nnz = (s >= eps).sum()
    null_space = vh[nnz:].conj().T
    return null_space




