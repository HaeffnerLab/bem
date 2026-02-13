#!/usr/bin/env python
# coding: utf-8

# In[2]:


from time import time
import pickle
import numpy as np
import os
import sys
import vtk
sys.path.append('/home/sqip/Documents/github/bem')
from bem import Electrodes, Sphere, Mesh, Grid, Configuration, Result
import multiprocessing 
# multiprocessing.set_start_method("fork")
from utils.helper_functions import *

area_list = []
for area in area_list:
    radius= 500e-3
    file = 'mit_LL'
    file_in_name = 'inter_results/inter_results_mit/'+file+'_'+str(radius)+'_'+str(area)+'.pkl'
    vtk_out = "inter_results/inter_results_mit/.vtks/"+file
    file_out_name = 'inter_results/inter_results_mit/'+file+'_'+str(radius)+'_'+str(area)+'_simulation'


    #open the mesh that will be used for simulation
    with open(file_in_name,'rb') as f:
        mesh_unit,xl,yl,zl,mesh,electrode_names= pickle.load(f) # import results from mesh processing

    # grid to evalute potential and fields atCreate a grid in unit of scaled length mesh_unit. Only choose the interested region (trap center) to save time.
    s = 1e-3
    # L = 102e-6
    Lx, Ly, Lz = 15*1e-3,15e-3,15*1e-3# in the unit of scaled length mesh_unit
    # xl,yl,zl = -3.75*1e-3,72*1e-3,270*1.0e-3
    xl,yl,zl = 0.0e-3,50.0e-3,0.0e-3
    sx,sy,sz = s,s,s

    # ni is number of grid points, si is step size. To  on i direction you need to fix ni*si.
    nx, ny, nz = [int(Lx/sx),int(Ly/sy),int(Lz/sz)]

    print("Size/l:", Lx, Ly, Lz)
    print("Step/l:", sx, sy, sz)
    print("Shape (grid point numbers):", nx, ny, nz)
    grid = Grid(center=(xl,yl,zl), step=(sx, sy, sz), shape=(nx,ny,nz))


    # Grid center (nx, ny ,nz)/2 is shifted to origin
    print("lowval",grid.indices_to_coordinates([0,0,0]))
    print("Grid center index", grid.indices_to_coordinates([nx/2,ny/2,nz/2]))
    print("gridpts:",nx*ny*nz)

    jobs = list(Configuration.select(mesh,"DC.*","RF"))    # select() picks one electrode each time.
    # run the different electrodes on the parallel pool
    # pmap = multiprocessing.Pool().map # parallel map
    #pmap = map # serial map
    t0 = time()
    # # range(len(jobs))
    # def run_map():
    #     list(pmap(run_job, ((jobs[i], grid, vtk_out,i,len(jobs)) for i in np.arange(len(jobs)))))
    #     print("Computing time: %f s"%(time()-t0))
    #     # run_job casts a word after finishing ea"ch electrode.
    for i in np.arange(len(jobs)):
        run_job((jobs[i], grid, vtk_out,i,len(jobs)))
    write_pickle(vtk_out,file_out_name,grid,electrode_names)

