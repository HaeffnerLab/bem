# -*- coding: utf8 -*-
#
#   bem: triangulation and fmm/bem electrostatics tools 
#
#   Copyright (C) 2011-2012 Robert Jordens <jordens@gmail.com>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pickle
import logging, re, operator, os

import numpy as np
import pyvista as pv


from .fastlap import (fastlap, centroid,
        TRIANGLE, NO_SOURCE, CONSTANT_SOURCE, INDIRECT, FIELD)

from .triangulation import Mesh
from .grid import Grid


class Result(object):
    configuration = None
    grid = None
    potential = None
    field = None
    field_square = None


    def to_vtk(self, prefix):
        """
        export the result to vtk
        the mesh and its triangle data go to prefix_name_mesh.vtk,
        the potential/field/field_square go to prefix_name.vtk
        all arrays are named
        """
        if self.configuration is not None:
            self.configuration.to_vtk(prefix)
        sg = pv.ImageData(dimensions = self.grid.shape, 
                            spacing = self.grid.step, 
                            origin = self.grid.get_origin())
        for data_name in "potential field field_square".split():
            data = getattr(self, data_name)
            if data is None:
                continue
            if len(data.shape) > 3:
                data = data.T.reshape(-1, data.shape[0])
            else:
                data = data.T.flatten()
            sg.point_data[data_name] = data
        file_name = "%s_%s.vtk" % (prefix, self.configuration.name)
        sg.save(file_name)
        logging.info("wrote %s", file_name)

    @classmethod
    def from_vtk(cls, prefix, name):
        """
        read in a result object from available vtk files
        """
        obj = cls()
        obj.configuration = Configuration.from_vtk(prefix, name)
        file_name="%s_%s.vtk" % (prefix, name)
        ug = pv.ImageData(file_name)
        for name in ug.array_names:
            data = ug.point_data[name]
            step = ug.spacing
            origin = ug.origin
            shape = ug.dimensions
            dim = shape[::-1]
            if data.ndim > 1:
                dim += (data.shape[-1],)
            data = np.asarray(data.reshape(dim).T)
            setattr(obj, name, data)
        # FIXME: only uses last array's data for grid
        center = origin+(np.array(shape)-1)/2.*step
        obj.grid = Grid(center=center, step=step, shape=shape)
        return obj

    @classmethod
    def from_pkl(cls, prefix, name):
        file_name="%s_%s.pkl" % (prefix, name)
        with open(file_name,'rb') as f:
            res = pickle.load(f)
        return res

    def to_pkl(self, prefix):
        file_name = "%s_%s.pkl" % (prefix, self.configuration.name)
        with open(file_name,'wb') as f:
            pickle.dump(self, f)
            print('dump name',file_name)


    @classmethod
    def load(cls, prefix, name, format):
        obj = cls()
        if format == 'vtk':
            return obj.from_vtk(prefix, name)
        if format == 'pkl':
            return obj.from_pkl(prefix, name)

    def save(self, prefix,format):
        if format == 'vtk':
            return self.to_vtk(prefix)
        if format == 'pkl':
            return self.to_pkl(prefix)


    @staticmethod
    def view(prefix, name):
        """
        construct a generic visualization of base mesh, refined mesh,
        and potential/field/pseudopotential data using pyvista

        """

        plotter = pv.Plotter(notebook = False)

        base_mesh_name = "%s_mesh.vtk" % prefix
        mesh_name = "%s_%s_mesh.vtk" % (prefix, name)

        def callback_func(electrode_name):
            idc_name = np.where(mesh.cell_data['electrode_name'] == electrode_name)
            selected_faces = mesh.faces.reshape(-1, 4)[idc_name]
            selected_mesh = pv.PolyData(mesh.points, faces = selected_faces)
            plotter.add_mesh(selected_mesh, name = 'select', style = 'wireframe', line_width=5, color='black')

        if os.access(mesh_name, os.R_OK):
            mesh = pv.PolyData(mesh_name)
            charge_min = mesh.cell_data['charge'].min()
            charge_max = mesh.cell_data['charge'].max()
            charge_absmax = abs(mesh.cell_data['charge']).max()

            Ncolors = len(pv.LookupTable('bwr').values)
            Ncolors_half = int(Ncolors/2)
            mid_to_left = int(np.rint(charge_min/charge_absmax * Ncolors_half))
            mid_to_right = int(np.rint(charge_max/charge_absmax * Ncolors_half))
            cmap_values = pv.LookupTable('bwr').values[Ncolors_half + mid_to_left : Ncolors_half + mid_to_right]
            cmap = pv.LookupTable(values = cmap_values, scalar_range = (charge_min, charge_max))
            # cmap = pv.LookupTable(cmap = 'bwr', scalar_range = (-4, 4))
            plotter.add_mesh(mesh, scalars = 'charge', cmap = cmap, show_edges = True)
            electrode_list = list(np.unique(mesh.cell_data['electrode_name']))
            plotter.add_text_slider_widget(callback_func, electrode_list,  pointa=(0.05, 0.9), pointb=(0.3, 0.9), interaction_event = 'always')
        elif os.access(base_mesh_name, os.R_OK):
            mesh = pv.PolyData(base_mesh_name)
            # colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', 
            #           '#2ca02c', '#98df8a', '#d62728', '#ff9896', 
            #           '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', 
            #           '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', 
            #           '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
            #           # above is colormap tab20 from matplotlib
            #           '#0000FF', '#7FFF00', '#4B0082', '#FF00FF', '#800000', '#FFFF00', '#B8860B', '#F0E68C', '#FFFFFF',
            #           # above is blue, chartreuse, indigo, magenta, maroon, yellow, darkgoldenrod, khaki, white
            #           ]
            plotter.add_mesh(mesh, scalars = 'electrode_name', 
                            scalar_bar_args = {'interactive': True, 'label_font_size': 15}, 
                            # cmap = colors, 
                            show_edges = True)
            electrode_list = list(np.unique(mesh.cell_data['electrode_name']))
            plotter.add_text_slider_widget(callback_func, electrode_list,  pointa=(0.05, 0.9), pointb=(0.3, 0.9), interaction_event = 'always')

        data_name = "%s_%s.vtk" % (prefix, name)
        if os.access(data_name, os.R_OK):
            data = pv.ImageData(data_name)
            if 'field_square' in data.point_data.keys():
                scalar_name = 'field_square'
            else:
                scalar_name = 'potential'
            iso_surfaces = data.contour(isosurfaces = 10, scalars = scalar_name, 
                                        # rng = [0, 0.015]
                                    )
            plotter.add_mesh(iso_surfaces, cmap = 'Greys', opacity = 1)

            def callback_func_iso(value):
                iso_slider = data.contour(isosurfaces = 1, scalars = scalar_name, rng = [value, value])
                plotter.add_mesh(iso_slider, name = 'iso', cmap = 'Greys', scalars = scalar_name, opacity = 1, show_scalar_bar = False)
            
            rng_iso = data.get_data_range(scalar_name)
            plotter.add_slider_widget(
                    callback=callback_func_iso,
                    rng=rng_iso,
                    title='potential/field_square',
                    # color='dimgray',
                    pointa=(0.4, 0.9),
                    pointb=(0.9, 0.9), 
                    interaction_event = 'always',
                )
            # plotter.add_mesh_isovalue(data, cmap = 'Greys', scalars = scalar_name,  pointa=(0.4, 0.9), pointb=(0.9, 0.9))

        plotter.show_bounds()
        plotter.add_camera_orientation_widget()
        plotter.show()


class Configuration(object):
    """
    a simulation configuration: simulate given mesh for certain
    potentials on certain electrodes
    """
    mesh = None
    potentials = None
    name = None

    charge = None

    opts = None
    data = None

    def __init__(self, mesh, potentials, name=None):
        self.mesh = mesh
        self.potentials = potentials
        self.name = name

    @classmethod
    def select(cls, mesh, *electrodes):
        """
        yields unit-potential simulation configurations given regexps for
        electrode names.
        """
        names = mesh.keys() # needs to be an OrderedDict
        for i, name in enumerate(names):
            match = any(re.match(p, name) for p in electrodes)
            if match or not electrodes:
                potentials = np.zeros(len(names))
                potentials[i] = 1.
                obj = cls(mesh, potentials, name)
                yield obj

    def set_data(self):
        """
        prepare arrays to be passed to fastlap
        """
        x = np.ascontiguousarray(self.mesh.fastlap_points())
        m = x.shape[0]
        # # get mesh.groups index referenced to mesh.keys()
        # keys_array = np.asarray(list(self.mesh.keys()))
        # indices = np.where(self.mesh.groups[:, None] == keys_array[None, :])[1]
        panel_potential = self.potentials[self.mesh.groups]
        shape = TRIANGLE*np.ones((m,), dtype=np.intc)
        panel_centroid = centroid(x, shape)
        constant_source = CONSTANT_SOURCE*np.ones((m,), dtype=np.intc)
        no_source = NO_SOURCE*np.ones((m,), dtype=np.intc)
        index = np.arange(m, dtype=np.intc)
        typep = np.zeros((m,), dtype=np.intc)
        self.opts = dict(x=x, shape=shape, lhs_index=index, rhs_index=index)
        self.data = dict(constant_source=constant_source,
                no_source=no_source, typep=typep,
                potential=panel_potential,
                centroid=panel_centroid)

    def solve_singularities(self, num_mom=4, num_lev=3, max_iter=200,
            tol=1e-5, **fastlap_opts):
        """
        solve singularties' strengths (charge) on the panels. For other
        options, see fastlap().
        """
        logging.info("solving singularities %s", self.name or
                self.potentials)
        self.set_data()
        fastlap_opts.update(self.opts)
        fastlap_opts.update(dict(num_mom=num_mom, num_lev=num_lev,
            max_iter=max_iter, tol=tol))
        n_itr, tol, self.charge, self.data["area"] = fastlap(
                lhs_type=self.data["constant_source"],
                rhs_type=self.data["no_source"],
                rhs_vect=self.data["potential"],
                xf=self.data["centroid"],
                type=self.data["typep"],
                job=INDIRECT, ret_areas=True, **fastlap_opts)
        logging.info("n_tri=%i, iter=%i, tol=%g",
                len(self.charge), n_itr, tol)
        return n_itr, tol

    def collect_charges(self):
        """
        returns total accumulated charge per electrode
        Q_i=C_ij*U_j
        if U_j = 1V*delta_ij, this is the column Ci of the capacitance matrix
        in CGS units, multiply by 4*pi*epsilon_0*length_scale to get SI Farad
        (or length_scale*1e7/c**2)
        """
        return np.bincount(self.mesh.groups,
                self.charge * self.data["area"])

    def adapt_mesh(self, triangles=1e3, opts="qQ", min_area=1e-4,
            max_area=1e4, observation=np.array([0., 0., 1.])):
        """
        refine the mesh such that triangles are sized according
        to the field strength they create at `observation`.
        About `triangles` triangles are created (typically more).
        Refinement is globally limited to between min_area and
        max_area per triangle.
        """
        self.solve_singularities(num_mom=2, tol=1e-2)
        distance2 = np.square(self.data["centroid"]-observation).sum(axis=-1)
        weight = np.absolute(self.charge)/distance2
        max_areas = (self.data["area"]*weight).sum()/(triangles*weight)
        np.clip(max_areas, min_area, max_area, out=max_areas)
        logging.info("estimate at least %i triangles",
                np.ceil(self.data["area"]/max_areas).sum())
        self.mesh.set_max_areas(max_areas)
        self.mesh = self.mesh.triangulate(opts=opts, new=True)

    def evaluate(self, xm, segsize=None, derivative=False,
            num_mom=4, num_lev=3, **fastlap_opts):
        """
        evaluate potential (derivative=False) or field (derivative=True)
        at the given points `xm`. Fragment into `segsize` points. For other
        options, see fastlap().
        """
        fastlap_opts.update(self.opts)
        fastlap_opts.update(dict(num_mom=num_mom, num_lev=num_lev))
        xm = np.ascontiguousarray(xm)
        n = xm.shape[0] # m*n/(20e3*25e3) takes 8GB, 3min

        if segsize is None:
            segsize = int(25e7/len(self.charge)) # about 4GB
        segsize = min(n, segsize)

        if derivative:
            field = np.empty((3, n), dtype=np.double)
            type = np.ones((segsize,), dtype=np.intc)
            xnrm_ = np.zeros((segsize, 3), dtype=np.double)
            xnrm_[:, 0] = 1
        else:
            field = np.empty((1, n), dtype=np.double)
            type = np.zeros((segsize,), dtype=np.intc)
            xnrm = None

        for i in range(0, n, segsize):
            segsize = min(segsize, n-i)
            for j in range(field.shape[0]):
                if derivative:
                    xnrm = np.roll(xnrm_, j, axis=1)[:segsize]
                fastlap(lhs_type=self.data["no_source"],
                        rhs_type=self.data["constant_source"],
                        rhs_vect=self.charge,
                        xf=xm[i:i+segsize],
                        lhs_vect=field[j, i:i+segsize],
                        xnrm=xnrm,
                        type=type[:segsize],
                        job=FIELD, **fastlap_opts)
                logging.info("derivative=%s, j=%i: %i-%i/%i",
                        derivative, j, i, i+segsize, n)
        if not derivative:
            field = field[0]
        return field

    def simulate(self, grid, potential=True, field=False,
            pseudopotential=True, **fastlap_opts):
        """performs the simulations for observation
        points in `grid`. other options passed
        down to `fastlap()`.
        returns a `Result()` object containing the data.
        use num_lev=1 for fmm_on_eval=False
        """

        res = Result()
        res.configuration = self
        res.grid = grid

        xm = grid.to_points()

        if potential:
            pot = self.evaluate(xm, derivative=False, **fastlap_opts)
            res.potential = pot.reshape(grid.shape)
        if field:
            field = self.evaluate(xm, derivative=True, **fastlap_opts)
            res.field = field.reshape((3,) + grid.shape)
            if pseudopotential:
                pp = np.square(field).sum(axis=0).reshape(grid.shape)
                res.field_square = pp

        logging.info("done with job %s, %s, %s, %s", self.name, potential, field,
                pseudopotential)
        return res

    def to_vtk(self, prefix):
        self.mesh.to_vtk("%s_%s" % (prefix, self.name),
            area=self.data["area"],
            potential=self.data["potential"],
            charge=self.charge)

    @classmethod
    def from_vtk(cls, prefix, name):
        mesh, datasets = Mesh.from_vtk("%s_%s" % (prefix, name))
        pot = datasets.get("potential")
        if pot is not None:
            eles, reles = np.unique(mesh.groups, return_index=True)
            potentials = pot[reles]
        else:
            potentials = None
        obj = cls(mesh, potentials, name)
        try:
            obj.data = dict(area=datasets.get("area"), 
                            potential = datasets.get("potential"), )
        except:
            print("\"Configuration.data\" loading failed")
        try:
            obj.charge = datasets.get("charge")
        except:
            print("\"Configuration.charge\" loading failed")

        return obj
