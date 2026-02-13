# ================================
# Values definition and imports
# ================================

import pickle
import numpy as np
import matplotlib as mpl
import sys
from utils.helper_functions import *
from bem.bemColors_lib.bemColors import bemColors
from bem import Electrodes, Sphere, Mesh, Grid, Configuration, Result, Box
from bem.formats import stl
from collections import OrderedDict

#here is the location that will be the reference point for further meshing
#usually, this is the ion's location, specified in appropriate units (usually millimeters)
zl = ( -450 + 560 ) *1e-3
yl = 75*1e-3
xl = 3.75*1e-3

mesh_unit = 1e-3
#radius is the radius of the sphere from the ions location to generate a finer
radius= 500e-3
#'area' is the area of the triangles inside the sphere
area = 1e-4
file_out = 'htrap'
stl_path = 'inter_results/htrap/htrap.stl'
fout_name = 'inter_results/htrap/'+file_out+'_'+str(radius)+'_'+str(area)+'.pkl'
module_path = os.path.abspath('')
s_nta = stl.read_stl(open(stl_path, "rb"))


# ================================
# Load STL file with colored electrodes
# ================================

# assign a name for each color
# taper htrap 2022
# assign a name for each color

# In od[key] = value, key is the color from the Fusion360 color library we made
# value assigns a specific name to that color in this program. 
# In future version, we may want to consider getting rid of this step for simplification
# then you just assign the color in fusion 360 and be done with it. The only concern there is you
ele_col = bemColors(np.array(list(set(s_nta[2]))),('fusion360','export_stl'))
ele_col.set_my_color(value = (178,178,178),cl_format = ('fusion360','export_stl','RGBA64'),name = 'self_defined')
ele_col.print_stl_colors()
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
od['bem21'] = 'DC21'
od['bem25'] = 'RF'
od['bem30'] = 'gnd'
for key in list(od.keys()):
    ele_col.color_electrode(color=key,name=od[key])

# print colors still with no name. These meshes will be neglected in the code below. 
ele_col.drop_colors()

# read stl into mesh with electrode names
# unnamed meshes will not be imported at all
mesh = Mesh.from_mesh(stl.stl_to_mesh(*s_nta, scale=1,
    rename=ele_col.electrode_colors, quiet=True))


# ================================
# Remesh the electrodes
# ================================

mesh.triangulate(opts="",new = False)

# for z in np.arange(-400e-3,700e-3,100e-3):
print('first triangulation:')
# for z in np.arange(-200e-3, 200e-3, 50e-3):
rad = radius*3
inside=area*30
outside=1
mesh.areas_from_constraints(Sphere(center=np.array([xl,yl,zl]),radius=rad, inside=inside, outside=outside))
mesh.triangulate(opts="a2Q",new = False)

# areas_from_constraints specifies sphere with finer mesh inside it.
#  "inside", "outside" set different mesh densities. 
print('second triangulation:')
rad =radius
inside=area*2
outside=1e4
mesh.areas_from_constraints(Sphere(center=np.array([xl,yl,zl]),radius=rad, inside=inside, outside=outside))
mesh.triangulate(opts="q5Q",new = False)

# save base mesh to a pickle file
with open(fout_name,'wb') as f:
    data = (mesh_unit,
            xl,
            yl,
            zl,
            mesh,
            list(od.values()))
    pickle.dump(data,f)


# ================================
# View the mesh and double check the mesh
# ================================

# Plot triangle mezshes of the final mesh
mesh.plot()

print(len(mesh.points))