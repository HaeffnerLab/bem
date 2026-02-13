"""
multipoles.py

Container class for post-processing
resuls from BEM simulations.

"""
import numpy as np
import pandas as pd
from collections import OrderedDict
from utils.expansion import spher_harm_expansion, spher_harm_cmp, nullspace, NamesUptoOrder2, PrintNamesUptoOrder2, NormsUptoOrder2
from utils.optimsaddle import exact_saddle, find_saddle
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import cvxpy as cvx

class MultipoleControl:

    # List are all stored values after running the class.
    X = None
    Y = None
    Z = None

    electrode_names = None
    electrode_positions = OrderedDict()
    electrode_potential = OrderedDict()

    # Below are all related to origin, roi

    origin = None
    origin_ind = None

    X_roi = None
    Y_roi = None
    Z_roi = None
    
    electrode_potential_roi = OrderedDict()

    # Below are all related to expansion order

    multipole_names = NamesUptoOrder2
    multipole_print_names = pd.Series(PrintNamesUptoOrder2, index = multipole_names)
    normalization_factors = NormsUptoOrder2

    order = None

    multipole_expansion = pd.DataFrame()
    electrode_potential_regenerated = OrderedDict()

    # Below are related to controlled_electrodes, used_multipoles

    controlled_elecs = None
    used_multipoles = None

    expansion_matrix = pd.DataFrame()
    pinv_matrix = pd.DataFrame()

    def __init__(self, trap, origin, roi, controlled_electrodes, used_multipoles, order = 2):
        '''
        trap['X'] = 1d array stores X values (in mm)
        trap['Y'] = 1d array stores Y values (in mm)
        trap['Z'] = 1d array stores Z values (in mm)
        trap['electrodes']['DC1']['position'] = i.e. [1, 1] # position on the chip (just used for pretty plotting)
        trap['electrodes']['DC1']['potential'] = 3d array stores grid of potential values (should be a nx by ny by nz matrix)
        
        '''

        # First read all basic values
        self.controlled_elecs = controlled_electrodes
        self.used_multipoles = used_multipoles
        self.order = order

        self.X, self.Y, self.Z = trap['X'], trap['Y'], trap['Z']

        self.electrode_names = list(trap['electrodes'].keys())

        for key in trap['electrodes'].keys():
            self.electrode_positions[key] = trap['electrodes'][key]['position']
            Vs = trap['electrodes'][key]['potential']
            self.electrode_potential[key] = Vs

        # Below setting up origin and region of interest and so on
        self.update_origin_roi(origin, roi)

        return

    def update_origin_roi(self, origin, roi,update_soln = True):
        '''
        This function updates origin and roi, and also update everything related to them.
        Can be called to update orgin and roi from object.
        '''
        self.origin = np.array(origin)

        x0_ind = (np.abs(self.X - self.origin[0])).argmin()
        y0_ind = (np.abs(self.Y - self.origin[1])).argmin()
        z0_ind = (np.abs(self.Z - self.origin[2])).argmin()
        self.origin_ind = np.array([x0_ind, y0_ind, z0_ind])

        self.X_roi = self.X[x0_ind - roi[0] : x0_ind + roi[0] + 1]
        self.Y_roi = self.Y[y0_ind - roi[1] : y0_ind + roi[1] + 1]
        self.Z_roi = self.Z[z0_ind - roi[2] : z0_ind + roi[2] + 1]


        for key in self.electrode_names:
            self.electrode_potential_roi[key] = self.electrode_potential[key][x0_ind - roi[0] : x0_ind + roi[0] + 1, 
                                                  y0_ind - roi[1] : y0_ind + roi[1] + 1, 
                                                  z0_ind - roi[2] : z0_ind + roi[2] + 1]
        if update_soln:
            self.update_expansion_order(self.order)
        return


    def update_expansion_order(self, order):
        '''
        This function updates expansion order, and also update everything related to them.
        Can be called from object to update all related staff.
        '''
        self.order = order
        self.multipole_expansion, self.electrode_potential_regenerated = self.expand_potentials_spherHarm(self.electrode_potential_roi, self.origin, self.X_roi, self.Y_roi, self.Z_roi, order, self.multipole_names)
        self.update_control(self.controlled_elecs, self.used_multipoles)
        return

    def update_control(self, controlled_electrodes, used_multipoles):
        '''
        This function updates controlled electrodes and used multipoles, 
        and also the control matrix retrieved from min norm problem.
        '''
        self.controlled_elecs = controlled_electrodes
        self.used_multipoles = used_multipoles
        trim_elecs = self.multipole_expansion[self.controlled_elecs]
        self.expansion_matrix = trim_elecs.loc[self.used_multipoles]

        numpoles = len(used_multipoles)
        soln_matrix = np.zeros((numpoles, len(controlled_electrodes)))
        for i in np.arange(numpoles):
            y = np.zeros(numpoles)
            y[i] = 1
            X = self.expansion_matrix
            soln = self.min_linf(y, X)
            soln_matrix[i] = soln

        self.pinv_matrix = pd.DataFrame(np.linalg.pinv(self.expansion_matrix), self.expansion_matrix.columns, self.expansion_matrix.index)
        # self.pinv_matrix = pd.DataFrame(np.transpose(soln_matrix), self.expansion_matrix.columns,
                                        # self.expansion_matrix.index)
        return self.expansion_matrix, self.pinv_matrix

    @staticmethod
    def expand_potentials_spherHarm(potential_roi, r0, X_roi, Y_roi, Z_roi, order, multipole_names):
        '''
        This function expands potentials, and drop shperical harmonics normalization factors.
        It renames multipoles
        up to 2nd order: multipole_names = ['C','Ey','Ez', 'Ex', 'U3=xy', 'U4=yz', r'U2=z^2-(x^2+y^2)/2',
                                            'U5=zx', r'U1=x^2-y^2']
                         normalization_factors = [np.sqrt(1/4/np.pi), np.sqrt(3/4/np.pi), np.sqrt(3/4/np.pi), 
                                                  np.sqrt(3/4/np.pi), np.sqrt(15/4/np.pi), np.sqrt(15/4/np.pi), 
                                                  np.sqrt(20/16/np.pi), np.sqrt(15/4/np.pi), np.sqrt(15/16/np.pi)]
        '''

        N = (order + 1)**2
        assert N >= len(multipole_names)
        multipoles = pd.DataFrame()
        multipoles_index_names = list(np.arange(0, N, 1))
        multipoles_index_names[:len(multipole_names)] = multipole_names
        potential_regenerated = {}
        for ele in potential_roi:
            Mj,Yj,scale = spher_harm_expansion(potential_roi[ele], r0, X_roi, Y_roi, Z_roi, order)
            multipoles[ele] = pd.Series(Mj[0:N].T[0], index = multipoles_index_names)

            Vregen = spher_harm_cmp(Mj,Yj,scale,order)
            potential_regenerated[ele] = Vregen.reshape([len(X_roi), len(Y_roi), len(Z_roi)])

        return multipoles, potential_regenerated

    def setVoltages(self, voltages):
        '''
        This function takes volteges you apply and returns multipole coefficients you get.
        input i.e. vs = {'DC1':1, 'DC2':2}
        '''
        M = (self.order + 1)**2
        coeffs = pd.Series(np.zeros(M), index = self.multipole_expansion.index)
        for key in voltages.keys():
            coeffs += self.multipole_expansion[key] * voltages[key]
        return coeffs

    def setMultipoles(self, coeffs):
        '''
        This function takes a set of desired multipole coefficients and returns the voltages needed to achieve that.
        Method: min norm
        input i.e. coeffs = {'Ex: 1', 'U2': 20}
        '''
        N = len(self.controlled_elecs)
        voltages = pd.Series(np.zeros(N), index = self.controlled_elecs)
        for key in coeffs.keys():
            voltages += self.pinv_matrix[key] * coeffs[key]
        return voltages

    def potentialControl_all(self, vs):
        '''
        This function takes voltages and returns the potential you get over the full space.
        input i.e. vs = {'DC1':1, 'DC2':2}
        '''
        output = np.zeros((len(self.X), len(self.Y), len(self.Z)))
        for key in vs.keys():
            output += self.electrode_potential[key] * vs[key]

        return output

    def potentialControl_roi(self, vs):
        '''
        This function takes voltages and returns the potential you get over the roi.
        i.e. vs = {'DC1':1, 'DC2':2}
        '''
        output_roi = np.zeros((len(self.X_roi), len(self.Y_roi), len(self.Z_roi)))
        for key in vs.keys():
            output_roi += self.electrode_potential_roi[key] * vs[key]

        return output_roi

    def potentialControl_regen(self, vs):
        '''
        This function takes voltages and returns the potential regenerated from multipole coefficients over the roi.
        i.e. vs = {'DC1':1, 'DC2':2}
        '''
        output_roi = np.zeros((len(self.X_roi), len(self.Y_roi), len(self.Z_roi)))
        for key in vs.keys():
            output_roi += self.electrode_potential_regenerated[key] * vs[key]

        return output_roi


    def write_txt(self,filename,strs,excl):
        outarray = []
        allmpl = ['Ex', 'Ey', 'Ez','U1', 'U2', 'U3', 'U4','U5']
        for multipole in allmpl:
            if multipole in self.pinv_matrix:
                for key in strs:
                    if key not in excl:
                        outarray = np.append(outarray, self.pinv_matrix[multipole][key])
                    elif excl[key] != "gnd":
                        outarray = np.append(outarray, self.pinv_matrix[multipole][excl[key]])
                    else:
                        outarray = np.append(outarray, 0)
            else:
                outarray = np.append(outarray,np.zeros(21))
        print(np.shape(outarray))
        pd.DataFrame(outarray).to_csv(filename+'.txt', header=None, index=None, float_format='%.15f')
    @staticmethod
    def min_linf(y, X):
        '''
        This function computes a constraint probelm: min(max(w)) s.t. X @ w = y.
        It returns w^{hat} that satisfy the above problem.
        '''
        X_mat = np.asarray(X)
        y_mat = np.asarray(y)
        w = cvx.Variable(X_mat.shape[1]) #b is dim x  
        objective = cvx.Minimize(cvx.norm(w,'inf')) #L_1 norm objective function
        constraints = [X_mat @ w == y_mat] #y is dim a and M is dim a by b
        prob = cvx.Problem(objective,constraints)
        result = prob.solve(verbose=False)
        w_hat = pd.Series(np.array(w.value), index = X.columns)
        return w_hat






