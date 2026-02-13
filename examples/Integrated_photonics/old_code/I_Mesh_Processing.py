#!/usr/bin/env python
# coding: utf-8

# # `Documention & Tutorial`
# This BEM package is used for for numerical simulation of the electric field of ion traps in python. Some external packages are wrapped into it:
# 
# * fast triangulation (http://www.cs.cmu.edu/~quake/triangle.research.html)
# * fastlap boundary element method (Nabors K S. Efficient three-dimensional capacitance calculation[D]. Massachusetts Institute of Technology, 1993.)
# 
# This document, also as a tutorial, explains how to use these packages to compute the electric field and compute its multipole expansion coefficient starting from scratch with a fusion 360 trap model(take `./f3d/htrap_overhang_9x5mm.f3d` for example). Our document consists of three .ipynb files: 
# 
# * `I_Mesh_Processing.ipynb`
# * `II_Field_Simulation.ipynb`
# * `III_Multipole_Expansion.ipynb`
# 
# We split the document into three files for two reasons:
# 1. The most time consuming part `II_Field_Simulation.ipynb` could be taken out alone and run on a high peformance computer. 
# 2. The intermediate result between files can be checked- 'intermediate files' refer to the inital mesh grid (that corresponds to the charge distribution), the simulation values and the simulation coordinates, all of which are saved in the inter_results folder. So if the voltage solution files for the trap don't make sense, it is possible to go back through saved data to see where things have gone wrong. 
# Here are the three jupyter notebooks and what:
# 1. I_Mesh_Processing- this notebook generates a mesh of the trap, saved in inter_results as a .pkl file
# 
# 
# Going through three files in order and running their python code will help you understand our workflow. Generally our workflow divides into three steps, corresponding to the above three .ipynb files. First, in I_Mesh_Processing, we identify different electrodes of the trap model by different colors. Also, we refine the trap model meshes to prepare for the second step. The result is stored in `mesh_result.pkl`. 
# 
# Secondly, in II_Field_Simulation, we simulate the electric potential with fastlap boundary element method(FMM), the accuracy of this step depends on the mesh size in the first step. The result is stored in `field_result.pkl`. 
# 
# Thirdly, in III_Multipole_Expansion, we analyze the electric field and electric potential. This notebook generates the multipole coefficients of the fields and potentials for each electrode and uses that information to generate a solution file with the function call `write_csv`. If your solution don't make sense, a good place to start is to insure that the mesh you are using is fine enough (generate several field results for different meshes)- if the potentials and fields are converging to specific values, this suggests there is something wrong with the multipole expansion. If you start with multipole expansion, you do not know if it is a problem with the expansion or with the field. 
# 
# Before running codes, you must set up the environment according to instructions in `README`
# 
# notes
# * Our workflow, as indicated by this document, abandons mayavi(with its embeded package tvtk). But we reserve mayavi in our Bem source code, if you want to use mayavi related functions, you must set up environment with files in`./backup/setup_with_mayavi`, just replace corresponding files in the root folder with them and set up.
#     * why we abandon mayavi: because its python interface(with its dependence vtk and PyQt5) requires complex package dependencies and version controls especially for new python versions. This makes it difficult to maintain this project.
#     * why we still keep mayavi in the source code? Because it is rooted in our source code deeply. We can just avoid invoking related functions instead of deleting them, because they may make sense in the future.

# # `I. Mesh Processing` 
# This .pynb file reads a colored model file `./inter_results/htrap.stl`, and assigns a name for each color(thus for each electrode). The result is stored in `./inter_results/mesh_result.pkl`.
# 
# In this file we name different electrodes and refine the meshes. The size of mesh determines the accuracy in next step. You should set the mesh finer and finer until the result converges. Also, you can set the mesh finer and test the convergence.
# 
# ## (1) color your electrodes (in fusion 360) and export stl
# First of all, you have a trap model in fusion 360 project, for example, `./f3d/htrap_overhang_9x5mm.f3d`. Here we use different color to identify different electrodes(STL file stores color for each mesh). In fusion 360, you can assign different appearance for different electrodes. We recommend to use ../../bemCol_lib/bemCol.adsklib for different electrodes. Same color for two electrodes means they are treated as the same electrode. Then we get `./f3d/htrap_overhang_9x5mm_my_colored.f3d`
# 
# Then export to an stl file in fusion360 file>export tab and move the file in `./inter_results`. We export our stl in mm, so we set the mesh unit to mm. 

# Furthurmore, it's our convention to use unit mm in our traps. e.g. [Electric field] = V/mm , [harmonic confining potential] = V/mm^2.so we set 

# In[1]:


# this is the unit for our Mesh structure
# in the following code, this is the default unit for python variables
#units 
mesh_unit = 1e-3    


# Additional notes (from Wenhao):
# * Why use standard color? We define a set of standard color lib `../../../bemCol_lib/bemCol.adsklib` for two reasons: 1. when exporting to .stl file, fusion 360 will compress 0-256 color(RGBA32) into 0-32 color(RGBA16), which means that some similar colors in fusion 360 will become same color after export to .stl file. Our standard color lib avoid this problem. 2.If you use .stl file exported from fusion 360 DIRECTLY, naming the electrodes would be very convinient. If you use other apps such as Inventor or Meshlab to generate .stl file, you can ignore the second reason because color encoding are quite different in different apps
# 
# 

# ## (2) assign a name for each color

# In[2]:


import pickle
import numpy as np
import matplotlib as mpl
import sys
from utils.helper_functions import *
from bem.bemColors_lib.bemColors import bemColors
from bem import Electrodes, Sphere, Mesh, Grid, Configuration, Result, Box
# load stl geometry file
from bem.formats import stl
# stl_path = "inter_results/"+"htrap"+".stl"  

radius= 500e-3
area = 1e-4
file = 'mit_LL'
stl_path = 'inter_results/inter_results_mit/'+file+'.stl'
fout_name = 'inter_results/inter_results_mit/'+file+'_'+str(radius)+'_'+str(area)+'_el7.pkl'
s_nta = stl.read_stl(open(stl_path, "rb"))


# Firstly, we print all the colors appear in the stl file. If a color is of standard colors in `bemCol.adsklib` or named manually by function `bemCol.set_my_color()`, then the color will appear in a straightforward name, such as `bem0`,`bem1` ...... . Otherwise, the color will simply be named as `'_unkCol0'` , `'_unkCol1'`, ... 
# 
# notes
# * try to comment out the line of `ele_col.set_my_color()` in the following block, name `'_unkCol0'` will appear.

# In[3]:


ele_col = bemColors(np.array(list(set(s_nta[2]))),('fusion360','export_stl'))
ele_col.set_my_color(value = (178,178,178),cl_format = ('fusion360','export_stl','RGBA64'),name = 'self_defined')
ele_col.print_stl_colors()


# Next, you need to assign a name for each color that appeared above. With standard colors defined in `bemCol.adsklib`, the correspondence between color and electrode is clear, which make it easy for this step. You can also comment out some `ele_col.set_color_name()`, run all the codes again and observe the missing part in the printed figure, which corresponds to the electrode you comment out. The second method also serves as a double check. Besides, parameter in `stl.stl_to_mesh()` can be set as `quiet = False` to check whether the program reads planes correctly.

# In[4]:


# assign a name for each color
# taper htrap 2022
# assign a name for each color

# In od[key] = value, key is the color from the Fusion360 color library we made
# value assigns a specific name to that color in this program. 
# In future version, we may want to consider getting rid of this step for simplification
# then you just assign the color in fusion 360 and be done with it. The only concern there is you
from collections import OrderedDict
od = OrderedDict()
od['bem1']='DC1'
od['bem2']='DC2'
od['bem3']='DC3'
od['bem4']='DC4'
od['bem5']='DC5'
od['bem6']='DC6'
od['bem7']='DC7'
od['bem8']='DC8'
od['bem9']='DC9'
od['bem10']='DC10'
od['bem11']='DC11'
od['bem12']='DC12'
od['bem13']='DC13'
od['bem14']='DC14'
od['bem15']='DC15'
od['bem16']='DC16'
od['bem17']='DC17'
od['bem18']='DC18'
od['bem19']='DC19'
od['bem20']='DC20'
od['bem21'] = 'RF'
for key in list(od.keys()):
    ele_col.color_electrode(color=key,name=od[key])

# print colors still with no name. These meshes will be neglected in the code below. 
ele_col.drop_colors()

# read stl into mesh with electrode names
# unnamed meshes will not be imported at all

mesh = Mesh.from_mesh(stl.stl_to_mesh(*s_nta, scale=1,
    rename=ele_col.electrode_colors, quiet=True))


# ## (3) remesh
# 
# In this step, we generate triangle mesh with constraints. The meshes are 2-dimensional triangles on the surface of electrodes. The region enclosed by constraint shape can have finer mesh. Triangulation is done by `triangle` C library. Folowing variables are all in unit `mesh_unit` now.
# 
# Our remesh strategy consists of two steps of triangulation: 
# 1. global triangulation without constraint. This step eliminate some long and sharp triangles by combining and dividing, and obtains a coarse grain triangulated model.
# 2. local triangulation with constraint. This step refines each triangles in step 1, the triangle density is defined by `mesh.areas_from_constraints
# 
# parameters in the below code block should be tuned specificly for different trap geometries.`

# In[5]:


# here we define a spherical constriant zone:
# xl = 270*1e-3
# yl = -3.5*1e-3
# zl = 72.5*1e-3

#here is the location that will be the reference point for further meshing
#usually, this is the ion's location, specified in appropriate units (usually millimeters)
zl = 240*1e-3
yl = 50.0*1e-3
xl = 0.0*1e-3




mesh.triangulate(opts="a1Q",new = False)


rad = 1000*1e-3
inside=5e-3
outside=1e4
print('first triangulation:')
mesh.areas_from_constraints(Sphere(center=np.array([xl,yl,zl]),radius=rad, inside=inside, outside=outside))
mesh.triangulate(opts="q20Q",new = False)

# what happened to this feature (Box)? is this on bem savio?
# mesh.areas_from_constraints(Box(start=-0.5, end= 0.5,inside=inside, outside=outside))



# areas_from_constraints specifies sphere with finer mesh inside it.
#  "inside", "outside" set different mesh densities. 
print('second triangulation:')
rad =500e-3
inside=area
outside=1e4
mesh.areas_from_constraints(Sphere(center=np.array([xl,yl,zl]),radius=rad, inside=inside, outside=outside))
mesh.triangulate(opts="q20Q",new = False)

# save base mesh to a pickle file
with open(fout_name,'wb') as f:
    data = (mesh_unit,
            xl,
            yl,
            zl,
            mesh,
            list(od.values()))
    pickle.dump(data,f)


# ## view mesh
# Uses Pyvista package to generate interactive 3-d plot of the mesh

# In[6]:


# Plot triangle meshes of the final mesh
mesh.plot()


# Show number of points included in the mesh

# In[7]:


print(len(mesh.points))


# In[ ]:




