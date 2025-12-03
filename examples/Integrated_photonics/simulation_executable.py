# ── Standard library ───────────────────────────────────────────────────────────
import os
import sys
import pickle
import multiprocessing
from time import time
from collections import OrderedDict

# ── Third-party ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import vtk
import pyvista as pv
import scipy.ndimage as im

# ── Project / local packages ───────────────────────────────────────────────────
from bem import Electrodes, Sphere, Mesh, Grid, Configuration, Result, Box
from bem import Result
from bem.formats import stl
from bem.bemColors_lib.bemColors import bemColors
from utils.helper_functions import *
from utils.multipoles import MultipoleControl
from utils.plottingfuncns import *
import argparse
import sys

def apply_dc_remap(name_to_array: dict, flip_sign: bool = False, invert_flag: bool = False) -> dict:
    """
    Remap DC electrodes according to DC_MAP.
    If flip_sign=True, all voltage arrays (potentials) are multiplied by -1.
    Non-DC keys (e.g., RF) are copied unchanged.
    """
    if invert_flag:
        dc_map = DC_MAP_INVERT
    else:
        dc_map = DC_MAP
    out = {}
    for key, arr in name_to_array.items():
        if key.startswith("DC"):
            try:
                src = int(key[2:])
            except ValueError:
                out[key] = -arr if flip_sign else arr
                continue
            tgt = dc_map.get(src, src)
            if tgt == 0:
                new_key = key  # Keep DC21 as-is
            else:
                new_key = f"DC{tgt}"
            out[new_key] = -arr if flip_sign else arr
        else:
            # e.g. 'RF' stays as is
            out[key] = -arr if flip_sign else arr
    return out
def find_voltages(zl, flip_sign=False):
    yl = 75*1e-3 #75 mm? probably due to scaling of the trap
    xl = 3.75*1e-3
    # meshing
    mesh_unit = 1e-3    

    radius = 500e-3
    area =1e-4
    file = 'htrap'
    stl_path = 'inter_results/htrap/htrap.stl'
    file_in_name = 'inter_results/htrap/'+file+'_'+str(radius)+'_'+str(area)+'.pkl'
    file_out_name = 'inter_results/htrap/'+file+'_'+str(radius)+'_'+str(area)+'_simulation'
    vtk_out = "inter_results/htrap/.vtks/"+file
    # file = 'cutout_with_ground'
    # stl_path = 'inter_results/htrap/cutout_with_ground.stl'
    # file_in_name = 'inter_results/htrap/'+file+'_'+str(radius)+'_'+str(area)+'.pkl'
    # file_out_name = 'inter_results/htrap/'+file+'_'+str(radius)+'_'+str(area)+'_simulation'
    # vtk_out = "inter_results/htrap/.vtks/"+file
    module_path = os.path.abspath('')
    s_nta = stl.read_stl(open(stl_path, "rb"))
    ele_col = bemColors(np.array(list(set(s_nta[2]))),('fusion360','export_stl'))
    ele_col.set_my_color(value = (178,178,178),cl_format = ('fusion360','export_stl','RGBA64'),name = 'self_defined')
    ele_col.print_stl_colors()
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
    od['bem21'] = 'DC21'
    od['bem25'] = 'RF'
    od['bem30'] = 'gnd'
    for key in list(od.keys()):
        ele_col.color_electrode(color=key,name=od[key])

    # print colors still with no name. These meshes will be neglected in the code below. 
    ele_col.drop_colors()


    mesh = Mesh.from_mesh(stl.stl_to_mesh(*s_nta, scale=1,
    rename=ele_col.electrode_colors, quiet=True))
    mesh.triangulate(opts="",new = False)
    rad = radius*3
    inside=area*30
    outside=1
    mesh.areas_from_constraints(Sphere(center=np.array([xl,yl,zl]),radius=rad, inside=inside, outside=outside))
    mesh.triangulate(opts="a2Q",new = False)
    print('second triangulation:')
    rad =radius
    inside=area*2
    outside=1e4
    mesh.areas_from_constraints(Sphere(center=np.array([xl,yl,zl]),radius=rad, inside=inside, outside=outside))
    mesh.triangulate(opts="q5Q",new = False)
    # save base mesh to a pickle file
    with open(file_in_name,'wb') as f:
        data = (mesh_unit,
                xl,
                yl,
                zl,
                mesh,
                list(od.values()))
        pickle.dump(data,f)
    with open(file_in_name,'rb') as f:
        mesh_unit,xl,yl,zl,mesh,electrode_names= pickle.load(f) # import results from mesh processing

    Lx, Ly, Lz = 11*1e-3,11*1e-3,11*1e-3# in the unit of scaled length mesh_unit
    s = 1e-3
    sx,sy,sz = s,s,s
    print("done")
    # ni is number of grid points, si is step size. To  on i direction you need to fix ni*si.
    nx, ny, nz = [int(Lx/sx),int(Ly/sy),int(Lz/sz)]
    print("Size/l:", Lx, Ly, Lz)
    print("Step/l:", sx, sy, sz)
    print("Shape (grid point numbers):", nx, ny, nz)
    grid = Grid(center=(xl,yl,zl), step=(sx, sy, sz), shape=(nx,ny,nz))
    # Grid center (nx, ny ,nz)/2 is shifted to origin
    print("lowval",grid.indices_to_coordinates([0,0,0]))
    print("Grid center index", grid.indices_to_coordinates((nx/2,ny/2,nz/2)))
    print("gridpts:",nx*ny*nz)
    center = (xl,yl,zl)
    step = (sx,sy,sz)
    shape = (nx,ny,nz)

    jobs = list(Configuration.select(mesh,'DC.*','RF'))    # select() picks one electrode each time.
    # run the different electrodes on the parallel pool
    pool = multiprocessing.Pool(3)
    pmap = pool.map # parallel map
    #pmap = map # serial map
    # print("before time")
    # t0 = time()
    # print("after time")
    # range(len(jobs))
    def run_map():
        out = pmap(run_job_bypassvtk, ((jobs[i], grid, vtk_out,i,len(jobs)) for i in np.arange(len(jobs))))
        # print( "Computing time: %f s"%(time()-t0))
        return out
        # run_job casts a word after finishing ea"ch electrode.

    worker_outputs = run_map()
    pool.close()
    pool.join()

    electrode_names = ['DC1','DC2','DC3','DC4','DC5','DC6','DC7','DC8','DC9','DC10',
                   'DC11','DC12','DC13','DC14','DC15','DC16','DC17','DC18','DC19','DC20','DC21',
                   'RF']

    by_name = {name: arr for (name, arr) in worker_outputs}
    by_name = apply_dc_remap(by_name, flip_sign=flip_sign, invert_flag = False)
    # Coordinates
    x, y, z = grid.to_xyz()

    # Build trap
    trap_t = {'X': x, 'Y': y, 'Z': z}
    for ele in ['DC1','DC2','DC3','DC4','DC5','DC6','DC7','DC8','DC9','DC10',
                'DC11','DC12','DC13','DC14','DC15','DC16','DC17','DC18','DC19','DC20','DC21','RF']:
        arr = by_name[ele]            # |E|^2 for RF, potential for DCs (as returned by run_job)
        trap_t[ele] = {
            'potential': arr,         # keep the key name 'potential' to match your downstream code
            'position': [0, 0],       # your original metadata; keep or update as needed
        }

    trap = {'X': trap_t['X'], 'Y': trap_t['Y'], 'Z': trap_t['Z'], 'electrodes': {}}
    for ele in trap_t.keys():
        if ele in ('X','Y','Z'): 
            continue
        trap['electrodes'][ele] = trap_t[ele]

    ###### loading in pickle file ###############################
    module_path = os.path.abspath('')

    radius= 700e-3
    # berkely change
    radius = 500e-3
    area = 1e-4

    path = module_path+'/inter_results/htrap/htrap_'+str(radius)+'_'+str(area)+'_simulation.pkl'
    print(path)

    print(str(area))
    # file_out = 'htrap_el4'
    # path = module_path+'/inter_results/htrap/'+file_out+'_'+str(radius)+'_'+str(area)+'_simulation.pkl'


    # f = open(path, 'rb')
    # trap = pickle.load(f)
    # trap['X'], trap['Y'], trap['Z'] are the position coordinates for the simulated volume
    print("x positions of the simulation")
    print(trap['Z'])
    #trap['electrodes'] is a dictionary with keys that correspond to each electrode
    print("electrodes simulated")
    print(trap['electrodes'].keys())
    print("shape of the simulated potential voltages for DC1")
    print("shape of the simulated |E|^2 for RF")
    print(np.shape(trap['electrodes']['RF']['potential']))

    #############################################################
    strs = list(trap['electrodes'].keys())
    xl = 3.75*1e-3
    yl = 75*1e-3

    position = [xl, yl, zl]

    nROI = 2
    roi = [nROI, nROI, nROI]
    order = 2

    #controlled electrodes- this will define which electrodes will be used to control your trap
    controlled_electrodes = []

    excl = {"RF":"gnd"}
    #     "DC1":"gnd",
    #     "DC2":"gnd",
    #     "DC3":"gnd",
    #     "DC4":"gnd",
    #     "DC5":"gnd",
    #     "DC7":"gnd",
    #     "DC8":"gnd",
    #     "DC10":"gnd",
    #     "DC11":"gnd",
    #     "DC12":"gnd",
    #     "DC13":"gnd",
    #     "DC14":"gnd",
    #     "DC15":"gnd",
    #     "DC17":"gnd",
    #     "DC18":"gnd",
    #     "DC20":"gnd"}
    electrode_names = ['DC1','DC2','DC3','DC4','DC5','DC6','DC7','DC8','DC9','DC10',
                    'DC11','DC12','DC13','DC14','DC15','DC16','DC17','DC18','DC19','DC20','DC21',
                    'RF']
    # this can get simplified for sure
    for electrode in electrode_names:
        if electrode not in excl:
            controlled_electrodes.append(electrode)

    used_order1multipoles = ['Ex', 'Ey', 'Ez']
    used_order2multipoles = ['U1', 'U2', 'U3','U4','U5']
    used_multipoles = used_order1multipoles + used_order2multipoles
    print(np.shape(trap['electrodes'][electrode]["potential"]))

    # create MultipoleControl object

    s = MultipoleControl(trap, position, roi, controlled_electrodes, used_multipoles, order)
    s.electrode_positions = OrderedDict([('DC1', [0, 1]), ('DC2', [0, 2]), ('DC3', [0, 3]), ('DC4', [0, 4]), 
                ('DC5', [0, 5]), ('DC6', [0, 6]), ('DC7', [0, 7]), ('DC8', [0, 8]), 
                ('DC9', [0, 9]), ('DC10', [0, 10]), ('DC11', [2, 1]), ('DC12', [2, 2]), 
                ('DC13', [2, 3]), ('DC14', [2, 4]), ('DC15', [2, 5]),('DC16', [2, 6]),
                ('DC17', [2, 7]),('DC18', [2, 8]),('DC19', [2, 9]),('DC20', [2, 10]),
                ('DC21', [1, 1]),('RF', [1, 2])])
    print(controlled_electrodes) 

    mult_voltages = {"DC1":0.1,
            "DC2":0.9,
            "DC3":1.3,
            "DC4":1.0,
            "DC5":-4.1,
            "DC6":1.3,
            "DC7":-0.1,
            "DC8":-1.00,
            "DC9":0.2,
            "DC10":0.0,
            "DC11":0.0,
            "DC12":0.4,
            "DC13":0.6,
            "DC14":-2.20,
            "DC15":-4.70,
            "DC16":-0.40,
            "DC17":-0.8,
            "DC18":-0.8,
            "DC19":-0.5,
            "DC20":0.1,
            "DC21":-0.4} 

    # ---- load pkl ----
    # pkl_path = "C:/Users/Ba133_IP/Code/startup/dac_load_settings.pkl"

    # with open(pkl_path, "rb") as f:
    #     raw = pickle.load(f)   # e.g. {0: v0, 1: v1, ..., 20: v20}
    # # ---- apply remap ----
    # mult_voltages_remapped = apply_dc_remap({f"DC{i}": float(raw.get(i, 0.0)) for i in range(21)}, flip_sign=False, invert_flag = True)
    # print(s.setVoltages(mult_voltages_remapped))
    # Generating pseudopotential
    ##############################################################################################################################
    #q = ion charge
    #l = units of used to specify simulation grid, should likely be 1 mm
    #Omega = trap drive angular frequency
    #voltage = RF voltage
    #v_to_mV = convert pseudopotential units from volts to millivolts
    q = 1.6*1e-19
    l = 1e-3
    m = 138*1.66054e-27
    Omega = 2*np.pi*22.78*1e6
    voltage = 145
    v_to_mV = 1000 

    #generating pseudopotential from simulated |E|^2 of RF field
    d = trap['electrodes']['RF']['potential'][:,:,0]*voltage**2*q/(4*m*Omega**2)/l**2*v_to_mV
    ##############################################################################################################################
    # freqs_rad_Hz, lambdas_rad, vecs_rad = compute_radial_secular_freq(
    #     trap=trap,
    #     d_mV=d,
    #     l=l,
    #     q=q,
    #     m=m,
    #     xl=xl,
    #     yl=yl,
    #     verbose=True
    # )
    # #generate figure to plot RF fields
    # plt.figure(figsize=(30/3,20/3))


    d = d
    x = trap['X']*1e3
    y = trap['Y']*1e3
        # ---------- NEW: compute radial secular frequencies from d ----------


    # 1. Get 1D coordinate arrays X_m, Y_m (in meters)
    X_full = np.array(trap['X'])
    Y_full = np.array(trap['Y'])

    if X_full.ndim == 1:
        X_m = X_full
    elif X_full.ndim == 2:
        X_m = X_full[:, 0]
    elif X_full.ndim == 3:
        X_m = X_full[:, 0, 0]
    else:
        raise ValueError(f"Unsupported X ndim={X_full.ndim}")

    if Y_full.ndim == 1:
        Y_m = Y_full
    elif Y_full.ndim == 2:
        Y_m = Y_full[0, :]
    elif Y_full.ndim == 3:
        Y_m = Y_full[0, :, 0]
    else:
        raise ValueError(f"Unsupported Y ndim={Y_full.ndim}")

    # 2. Convert pseudopotential to volts
    phi = d / 1000.0  # V
    Nx, Ny = phi.shape

    # 3. Choose the point where we evaluate curvature.
    #    Easiest robust choice: true RF null (minimum of pseudopotential).
    ix, iy = np.unravel_index(np.argmin(phi), phi.shape)

    # Make sure we're not on the very edge (for central differences)
    ix = max(1, min(ix, Nx - 2))
    iy = max(1, min(iy, Ny - 2))

    # 4. Grid spacing (assumed uniform)
    dx = X_m[1] - X_m[0]
    dy = Y_m[1] - Y_m[0]

    # 5. Second derivatives via central finite differences
    phi_xx = (phi[ix + 1, iy] - 2.0 * phi[ix, iy] + phi[ix - 1, iy]) / (dx ** 2)
    phi_yy = (phi[ix, iy + 1] - 2.0 * phi[ix, iy] + phi[ix, iy - 1]) / (dy ** 2)
    phi_xy = (phi[ix + 1, iy + 1] - phi[ix + 1, iy - 1]
              - phi[ix - 1, iy + 1] + phi[ix - 1, iy - 1]) / (4.0 * dx * dy)

    H = np.array([[phi_xx, phi_xy],
                  [phi_xy, phi_yy]])  # 2×2 Hessian in V/m^2

    # 6. Diagonalize Hessian: eigenvalues = curvatures along radial modes
    lambdas, vecs = np.linalg.eigh(H)   # lambdas in V/m^2

    # 7. Convert curvatures -> secular frequencies:
    #    m * omega^2 = q * lambda  =>  omega = sqrt(q * lambda / m)
    omegas = np.sqrt(np.clip(q * lambdas / m, 0.0, np.inf))
    freqs_Hz = omegas / (2.0 * np.pi)

    print("RF null used at grid index (ix, iy):", ix, iy)
    print("RF null position [m]:                ", X_m[ix], Y_m[iy])
    print("Radial Hessian eigenvalues (V/m^2): ", lambdas)
    print("Radial secular frequencies (Hz):    ", freqs_Hz*1e3)
    print("Radial secular frequencies (MHz):   ", freqs_Hz * 1e-3)

    # #setting contour values
    # min_val = np.min(d)
    # max_val =  np.max(d)*0.5
    # steps = 5
    # c_range = np.arange(min_val,max_val,(max_val-min_val)/(steps+1))

    # #plotting contours
    # cmap = LinearSegmentedColormap.from_list('black', ['black', 'black'])
    # CS = plt.contour(x,y,np.transpose(d),levels=c_range,cmap=cmap,zorder=2)

    # #labeling contours
    # def fmt(x):
    #     s = f"{x:.1f}"
    #     if s.endswith("0"):
    #         s = f"{x:.0f}"
    #     return rf"{s} mV" if plt.rcParams["text.usetex"] else f"{s} mV"
    # plt.clabel(CS, CS.levels, inline=True,fmt=fmt,fontsize=12,use_clabeltext=True)


    # Find location of the RF null
    ##############################################################################################################################
    # min_value = np.min(d[:,:])
    # coordinates = np.where(d == min_value)
    # plt.plot(x[coordinates[0][0]],y[coordinates[1][0]],'kx')
    # min_x = np.round(x[coordinates[0]],0)[0]
    # min_y = np.round(y[coordinates[1]],0)[0]
    # plt.text(min_x-0.5,min_y-0.5,
    #         r"x ($\mathrm{\mu m}$) , y ($\mathrm{\mu m}$) , V (mV): "
    #         +str(min_x)+' , '+str(min_y)+' , '+str(np.round(d[coordinates[0][0],coordinates[1][0]],4)),color='k',
    #         horizontalalignment='center')
    # ##############################################################################################################################
    # #Final Plot Params
    # ##############################################################################################################################
    # plt.xlabel(r"Horizontal Location ($\mathrm{\mu}$m)",fontsize=20)
    # plt.grid()
    # plt.xticks(np.arange(-40,40,10))
    # plt.yticks(np.arange(0,180,20))a
    # plt.ylabel(r"Vertical Location ($\mathrm{\mu}$m)",fontsize=20)
    # plt.xticks(np.arange(-5,10,1))
    # plt.yticks(np.arange(65,85,1))
    # plt.xlim(3.75-5,3.75+5)
    # plt.ylim(75-5,75+5)
    # plt.show()
    ##############################################################################################################################
    height_list = trap['Y'][np.arange(nROI,len(trap['Z'])-nROI)]
    print(np.arange(nROI,len(trap['Z'])-nROI))
    numMUltipoles = len(s.multipole_print_names)
    ne = len(s.electrode_names)
    multipoles_vs_height = np.zeros((len(height_list), numMUltipoles, ne))
    print('height_list:',height_list)
    for i, height in enumerate(height_list):
        position1 = [xl, height, zl]
        s.update_origin_roi(position1, roi)
        multipoles_vs_height[i] = np.asarray(s.multipole_expansion.loc[s.multipole_names])
    height_list = trap['Z'][2:len(trap['Z'])-2]
    numMUltipoles = len(s.multipole_print_names)


    ne = len(s.electrode_names)
    multipoles_vs_height = np.zeros((len(height_list), numMUltipoles, ne))

    for i, height in enumerate(height_list):
        position1 = [xl, yl,height]
        s.update_origin_roi(position1, roi)
        multipoles_vs_height[i] = np.asarray(s.multipole_expansion.loc[s.multipole_names])

    # size = 20
    # electrode_list = ['DC3','DC13','DC4','DC14']
    # fig = plt.figure()
    # fig.canvas.draw()
    # fig.tight_layout(pad=1)
    # for 400 um solution
    #save_muls(s,xl,zl,roi,height= yl*1e3, ez=0.5, ex=-0.75, ey=-0.9,u2=6.0, u5=29.0, u1=-0.0, u3=-11.0,u4=0.0)
    save_muls(s,xl,zl,roi,height= yl*1e3, ez=0.0, ex=0.0, ey=1.0,u2=6.0, u5=0.0, u1=0.0, u3=2.0,u4=0.0)
    
DC_MAP = {
    1: 21,  2: 20,  3: 19,  4: 18,  5: 17,
    6: 16,  7: 15,  8: 14,  9: 13, 10: 12,
   11: 10, 12:  9, 13:  8, 14:  7, 15:  6,
   16:  5, 17:  4, 18:  3, 19:  2, 20:  1,
   21: 11,
}

DC_MAP_INVERT = {
   21:  1, 20:  2, 19:  3, 18:  4, 17:  5,
   16:  6, 15:  7, 14:  8, 13:  9, 12: 10,
   10: 11,  9: 12,  8: 13,  7: 14,  6: 15,
    5: 16,  4: 17,  3: 18,  2: 19,  1: 20,
   11: 21,
}

#new DC_MAP
# DC_MAP = {
#     1: 10,  2: 9,  3: 8,  4: 7,  5: 6,
#     6: 5,  7: 4,  8: 3,  9: 2, 10: 1,
#    11: 11, 12:  12, 13:  13, 14:  14, 15:  15,
#    16:  16, 17:  17, 18:  18, 19:  19, 20:  20,
#    21:  0,
# }


def main(start: int, stop: int, step_abs: int = 5, convert_to_meters: bool = False, flip_sign: bool = False):
    """
    Calls find_voltages for start, start±5, ..., stop (inclusive).
    If convert_to_meters=True, passes values as meters (value * 1e-3).
    """
    step = (-abs(step_abs) if start > stop else abs(step_abs))
    val = start

    def done(a, b):
        return a < b if step < 0 else a > b

    while not done(val, stop):
        zl = (val * 1e-3) if convert_to_meters else val  # <-- use the flag
        find_voltages(zl, flip_sign=flip_sign)
        val += step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep zl and run find_voltages.")
    parser.add_argument("start", type=int, help="Start value (e.g., -100)")
    parser.add_argument("stop", type=int, help="Stop value (e.g., -300)")
    parser.add_argument("--meters", action="store_true",
                        help="If set, convert values from mm to meters before calling find_voltages.")
    parser.add_argument("--step", type=int, default=5,
                        help="Absolute step size (default: 5)")
    parser.add_argument("--flip", action="store_true",
                    help="If set, flip the sign of all DC voltages.")
    args = parser.parse_args()
    main(args.start, args.stop, step_abs=args.step, convert_to_meters=args.meters, flip_sign=args.flip)
