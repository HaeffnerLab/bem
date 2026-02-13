"""
Functions for locating the rf saddle point
"""

from utils.expansion import spher_harm_expansion
import numpy as np
import scipy.optimize as spo

def exact_saddle(V,X,Y,Z,dim, scale=1, Z0=None):
    """This version finds the approximate saddle point using pseudopotential,
    does a multipole expansion around it, and finds the exact saddle point by
    maximizing the quadrupole terms. Similar to interpolation.
    V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    dim is the dimensionality (2 or 3).
    Z0 is the coordinate where a saddle point will be sought if dim==2.
    
    returns: X, Y, Z coodinate of the potential
    """

    #from all_functions import find_saddle,sum_of_e_field
    if dim==3:
        [I,J,K]=find_saddle(V,X,Y,Z,3, scale) # guess saddle point; Z0 not needed
        r0=[X[I],Y[J],Z[K]]
        if I<2 or I>V.shape[0]-2: 
            print('exact_saddle.py: Saddle point out of bounds in radial direction.')
            return r0
        if J<2 or J>V.shape[1]-2:
            print('exact_saddle.py: Saddle point out of bounds in vertical direction.')
            return r0
        if K<2 or K>V.shape[2]-2:
            print('exact_saddle.py: Saddle point out of bounds in axial direction.')
            return r0
        if V.shape[0]>100:
            Vn = V[I-2:I+3,J-2:J+3,K-2:K+3] # create smaller 5x5x5 grid around the saddle point to speed up optimization
            # note that this does not prevent the optimization function from trying values outside this
            Xn,Yn,Zn=X[I-2:I+3],Y[J-2:J+3],Z[K-2:K+3] # change grid vectors as well
        else:
            Vn,Xn,Yn,Zn = V,X,Y,Z
        #################################### Minimize
        r=spo.minimize(sum_of_e_field,r0,args=(Vn,Xn,Yn,Zn)) 
        r=r.x # unpack for desired values
        Xs,Ys,Zs=r[0],r[1],r[2] 
    #################################################################################################    
    if dim==2: 
        if len(V.shape)==3:
            K=0 # in case there is no saddle
            for i in range(len(Z)):
                if Z[i-1]<Z0 and Z[i]>=Z0:
                    K=i-1
        Vs = V.shape
        if K>=len(Z): # Matlab had Z, not V; also changed from == to >=
            return('The selected coordinate is at the end of range.')
        v1=V[:,:,K-1] # potential to left
        v2=V[:,:,K] # potential to right (actually right at estimate; K+1 to be actually to right)
        V2=v1+(v2-v1)*(Z0-Z[K-1])/(Z[K]-Z[K-1]) # averaged potential around given coordinate
        [I,J,K0]=find_saddle(V,X,Y,Z,2,Z0=Z0) 
        r0=X[I],Y[J]
        if (I<2 or I>V.shape[0]-2): 
            print('exact_saddle.py: Saddle point out of bounds in radial direction.\n')
            return r0
        if (J<2 or J>V.shape[1]-2):
            print('exact_saddle.py: Saddle point out of bounds in vertical direction.\n')
            return r0
        if V.shape[0]>100:
            Vn = V[I-2:I+3,J-2:J+3,K-2:K+3] # create smaller 5x5x5 grid around the saddle point to speed up optimization
            # note that this does not prevent the optimization function from trying values outside this
            Xn,Yn,Zn=X[I-2:I+3],Y[J-2:J+3],Z[K-2:K+3] # Matlab 4, not 2
        else:
            Vn,Xn,Yn,Zn = V,X,Y,Z
        ################################## Minimize
        r=spo.minimize(sum_of_e_field_2d,r0,args=(Z0,Vn,Xn,Yn,Zn)) 
        r=r.x # unpack for desired values
        Xs,Ys,Zs=r[0],r[1],Z0
    return [Xs,Ys,Zs]

def find_saddle(V,X,Y,Z,dim, scale=1, Z0=None):
    """Returns the indices of the local extremum or saddle point of the scalar A as (Is,Js,Ks).
    V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    Z0: Z coordinate for saddle finding in a 2D potential slice
    For dim==2, the values of A are linearly extrapolated from [Z0] and [Z0]+1
    to those corresponding to Z0 and Ks is such that z[Ks]<Z0, z[Ks+1]>=Z0."""

    if (dim==2 and Z0==None):
        return 'z0 needed for evaluation'
    if dim==3:
        if len(V.shape)!=3:
            return('Problem with find_saddle.m dimensionalities.')
        f=V/float(np.amax(V)) # Normalize field
        [Ex,Ey,Ez]=np.gradient(f,abs(X[1]-X[0])/scale,abs(Y[1]-Y[0])/scale,abs(Z[1]-Z[0])/scale) # grid spacing is automatically consistent thanks to BEM-solver
        E=np.sqrt(Ex**2+Ey**2+Ez**2) # magnitude of gradient (E field)
        m=E[0,0,0]
        origin=[0,0,0]
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                for k in range(E.shape[2]):
                    if E[i,j,k]<m:
                        m=E[i,j,k]
                        origin=[i,j,k]          
        if origin[0]==(0 or V.shape[0]):
            print('find_saddle: Saddle out of bounds in  x (i) direction.\n')
            return origin
        if origin[0]==(0 or V.shape[1]):
            print('find_saddle: Saddle out of bounds in  y (j) direction.\n')
            return origin
        if origin[0]==(0 or V.shape[2]): 
            print('find_saddle: Saddle out of bounds in  z (k) direction.\n')
            return origin
    #################################################################################################
    if dim==2: # Extrapolate to the values of A at z0.
        V2=V
        if len(V.shape)==3:
            Ks=0 # in case there is no saddle point
            for i in range(len(Z)):
                if Z[i-1]<Z0 and Z[i]>=Z0:
                    Ks=i-1
                    if Z0<1:
                        Ks+=1
            Vs=V.shape
            if Ks>=len(Z):
                return('The selected coordinate is at the end of range.')
            v1=V[:,:,Ks] 
            v2=V[:,:,Ks+1]
            V2=v1+(v2-v1)*(Z0-Z[Ks])/(Z[Ks+1]-Z[Ks])
        V2s=V2.shape
        if len(V2s)!=2: # Old: What is this supposed to check? Matlab code: (size(size(A2),2) ~= 2)
            return('Problem with find_saddle.py dimensionalities. It is {}.'.format(V2s))
        f=V2/float(np.max(abs(V2)))
        [Ex,Ey]=np.gradient(f,abs(X[1]-X[0]),abs(Y[1]-Y[0]))
        E=np.sqrt(Ex**2+Ey**2)
        m=float(np.min(E))
        mr=E[0,0]
        Is,Js=1,1 # in case there is no saddle
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                if E[i,j]<mr:
                    mr=E[i,j]
                    Is,Js=i,j
        origin=[Is,Js,Ks]
        if Is==1 or Is==V.shape[0]:
            print('find_saddle: Saddle out of bounds in  x (i) direction.\n')
            return origin
        if Js==1 or Js==V.shape[1]:
            print('find_saddle: Saddle out of bounds in  y (j) direction.\n')
            return origin
    return origin

def sum_of_e_field(r,V,X,Y,Z,exact_saddle=True):
    """V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    r: center position for the spherical harmonic expansion
    Finds the weight of high order multipole terms compared to the weight of
    second order multipole terms in matrix V, when the center of the multipoles
    is at x0,y0,z0.
    Used by exact_saddle for 3-d saddle search.
    Note that order of outputs for spher_harm_exp are changed, but 1 to 3 should still be E field."""

    x0,y0,z0=r[0],r[1],r[2]
    c,c1,c2 = spher_harm_expansion(V, [x0, y0, z0], X, Y, Z, 3)
    s=c**2
    f=sum(s[1:4])/sum(s[4:9])
    real_f=np.real(f[0])
    return real_f

def sum_of_e_field_2d(r,z0,V,X,Y,Z,exact_saddle=True):
    """V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    r: center position for the spherical harmonic expansion
    Finds the weight of high order multipole terms compared to the weight of
    second order multipole terms in matrix V, when the center of the multipoles
    is at x0,y0,z0.
    Used by exact_saddle for 3-d saddle search.
    Note that order of outputs for spher_harm_exp are changed, but 1 to 3 should still be E field."""

    x0,y0=r[0],r[1]

    c,c1,c2 = e.spher_harm_expansion(V, [x0, y0, z0], X, Y, Z, 4)
    s=c**2
    f=sum(s[1:4])/sum(s[4:9])
    real_f=np.real(f[0])
    return real_f
