import numpy as np
import pandas as pd
from itertools import compress
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid.inset_locator import inset_axes, InsetPosition, mark_inset, zoomed_inset_axes
from ipywidgets import interactive, interact
import ipywidgets as widgets
import os
from matplotlib import cm
from datetime import datetime


colormap = cm.get_cmap('tab20')
# %% md
# define some useful functions
# %%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.3f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',  # Horizontally center label
            va=va)  # Vertically align label differently for
        # positive and negative values
def update_colors(ax):
    lines = ax.lines
    colors = colormap(np.linspace(0, 1, len(lines)))
    for line, c in zip(lines, colors):
        line.set_color(c)

def plot_multipole_vs_expansion_height(height,position,roi):
    position1 = position
    s.update_origin_roi(position1, roi)
    #     print np.dot(s.multipole_expansions,vs[0])

    Nmulti = s.multipole_expansion.shape[0]

    fig, ax = plt.subplots(len(vs), 1, figsize=(10, 24))
    for i, v in enumerate(vs):
        coeffs = s.setVoltages(v)
        ax[i].bar(range(Nmulti), coeffs)
        max_coeff = np.max(coeffs)
        min_coeff = np.min(coeffs)
        margin = (max_coeff - min_coeff) * 0.5
        ymax = max_coeff + margin
        ymin = min_coeff - margin
        ax[i].set_ylim(ymin, ymax)
        ax[i].set_title(s.electrode_names[i])
        fig.canvas.draw()
        add_value_labels(ax[i])
    plt.xticks(range(Nmulti), s.multipole_print_names, rotation=-90)
    fig.tight_layout(pad=1)
    plt.show()
#----------------------------------------------------------------------------------------------------------
def save_voltages_csv(path, elecs, voltages, label=None):
    """
    Append voltages as a new column to a CSV whose first column is 'electrode'
    and contains DC1..DC21 (one per row). Creates the file if it doesn't exist.

    Parameters
    ----------
    path : str
        CSV file path.
    elecs : Sequence[str]
        Electrode names, e.g. ['DC1', ..., 'DC21'].
    voltages : Sequence[float]
        Voltages aligned with `elecs`.
    label : str or None
        Column name for this run. If None, uses a timestamp.
    """
    if label is None:
        label = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    # Build a tidy Series indexed by electrode for easy alignment
    s = pd.Series(voltages, index=pd.Index(elecs, name="electrode"), name=label)
    print(label)
    if not os.path.exists(path):
        # Create new file: first column 'electrode', then one column of voltages
        df = s.reset_index()
        df.to_csv(path, index=False)
        return

    # Append as a new column, aligning by electrode name
    existing = pd.read_csv(path)

    # Ensure the first column is named 'electrode'
    if existing.columns[0].lower() != "electrode":
        raise ValueError(f"Expected first column to be 'electrode', got {existing.columns[0]!r}")

    existing = existing.set_index("electrode")

    # Add/overwrite column for this label, automatically aligning rows by name
    existing[label] = s

    # (optional) ensure the canonical order DC1..DC21 if you have it
    # canonical = [f"DC{i}" for i in range(1, 22)]
    # existing = existing.reindex(canonical)

    # Write back
    existing.reset_index().to_csv(path, index=False)
# %% md
# plot voltage solution of a group of multipole coefficients
# %%
def plot_muls(s,xl,zl,roi,height, **multipoles): # ey, ez, ex, u3, u2, u5, u1, u4):
    position1 = [xl, height * 1e-3, zl]
    s.update_origin_roi(position1, roi)
    multipole_coeffs = {key:multipoles[key.lower()] for key in s.used_multipoles}
    voltages = s.setMultipoles(multipole_coeffs)
    elecs = getattr(s, "controlled_elecs", getattr(s, "controlled_electrodes", s.electrode_names))

    # Optional: use height in the column label so you can tell runs apart
    label = f"zl_{zl*1e3}um"

    save_voltages_csv("shuttle.csv", elecs, voltages, label=label)
    # #multipole_coeffs = {'Ey': ey, 'Ez': ez, 'Ex': ex, 'U3': u3, 'U2': u2, 'U5': u5, 'U1': u1,'U4':u4}
    # voltages = s.setMultipoles(multipole_coeffs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 12))
    fig.suptitle(f'Trap Height: {height:0.0f} $\mu$m')
    ax1.bar(s.controlled_elecs, voltages)
    ax1.set_xticklabels(s.controlled_elecs, rotation=45, fontsize=12)
    #     ax1.set_ylim(-25, 40)
    ax1.set_ylabel('V')
    print(s.controlled_elecs)
    xpos = [s.electrode_positions[ele][0] for ele in s.controlled_elecs]
    ypos = [s.electrode_positions[ele][1] for ele in s.controlled_elecs]
    plot = ax2.scatter(xpos, ypos, 1000, list(voltages), cmap='bwr')
    fig.colorbar(plot)
    ax2.set_ylabel("axial electrode location")
    ax2.set_xlabel("radial electrode location")
    ax2.set_xlim(min(xpos) - 1, max(xpos) + 1)
    ax2.set_ylim(min(ypos) - 1, max(ypos) + 1)
    plt.subplots_adjust(bottom=0.25)
    print(voltages)
    plt.show()
    
def save_muls(s, xl, zl, roi, height, **multipoles):
    position1 = [xl, height * 1e-3, zl]
    s.update_origin_roi(position1, roi)
    multipole_coeffs = {key:multipoles[key.lower()] for key in s.used_multipoles}
    voltages = s.setMultipoles(multipole_coeffs)
    elecs = getattr(s, "controlled_elecs", getattr(s, "controlled_electrodes", s.electrode_names))

    # Optional: use height in the column label so you can tell runs apart
    label = f"zl_{zl*1e3}um"

    save_voltages_csv("shuttle.csv", elecs, voltages, label=label)
    print(voltages)
    
def U2_to_mhz(u2):
    m = 40.078 * 1.66e-27
    e = 1.6e-19
    return np.sqrt(2 * e * u2 * 1e6 / m) / 2 / np.pi / 1e6


def mhz_to_U2(mhz):
    m = 40.078 * 1.66e-27
    e = 1.6e-19
    return 4 * np.pi ** 2 * 1e12 * 1e-6 * m * mhz ** 2 / 2 / e

# %% md
# plot potential projection along x, y, z axis when apply voltage solution of a group of multipole coefficients
# %%
def plot_1d(s,xl,zl,roi,height, ey, ez, ex, u3, u2, u5, u1):
    position1 = [xl, height * 1e-3, zl]
    s.update_origin_roi(position1, roi)
    multipole_coeffs = {'Ey': ey, 'Ez': ez, 'Ex': ex, 'U3': u3, 'U2': u2, 'U5': u5, 'U1': u1}
    voltages = s.setMultipoles(multipole_coeffs)
    # this is the 'original' plot- this corresponds to plugging your solutions for your electrodes
    # (i.e. your c-file where for this case we set pure u2) into the bem simulation
    # to directly create the field
    potential_roi = s.potentialControl_roi(voltages)
    # this takes the electrode solutions
    potential_regen = s.potentialControl_regen(voltages)

    nearestZ = find_nearest(s.Z, height * 1e-3)
    indNearestZ_roi = np.abs(s.Z_roi - nearestZ).argmin()
    x0 = find_nearest(s.X, 0)
    indx0_roi = np.abs(s.X_roi - x0).argmin()
    y0 = find_nearest(s.Y, 0)
    indy0_roi = np.abs(s.Y_roi - y0).argmin()
    #     print(s.X_trunc)

    potential_z_roi = potential_roi[indx0_roi, indy0_roi, :]
    potential_z_regen = potential_regen[indx0_roi, indy0_roi, :]
    potential_x_roi = potential_roi[:, indy0_roi, indNearestZ_roi]
    potential_x_regen = potential_regen[:, indy0_roi, indNearestZ_roi]
    potential_y_roi = potential_roi[indx0_roi, :, indNearestZ_roi]
    potential_y_regen = potential_regen[indx0_roi, :, indNearestZ_roi]

    offset = potential_roi[indx0_roi, indy0_roi, indNearestZ_roi] - potential_regen[
        indx0_roi, indy0_roi, indNearestZ_roi]
    print('offset',offset)

    fsize = 20

    fig1, ax = plt.subplots(3, 1, figsize=(10, 20))
    ax[0].set_title('Potential along X', fontsize=fsize)
    ax[0].grid()
    ax[0].plot(s.X_roi * 1e3, potential_x_roi, label='original')
    ax[0].plot(s.X_roi * 1e3, potential_x_regen + offset, label='regenerated + offset')
    ax[0].set_xlabel('X ($\mu$m)', fontsize=fsize)
    ax[0].set_ylabel('V (v)', fontsize=fsize)
    ax[0].legend(fontsize=fsize)

    ax[1].set_title('Potential along Y', fontsize=fsize)
    ax[1].grid()
    ax[1].plot(s.Y_roi * 1e3, potential_y_roi, label='original')
    ax[1].plot(s.Y_roi * 1e3, potential_y_regen + offset, label='regenerated + offset')
    ax[1].set_xlabel('Y ($\mu$m)', fontsize=fsize)
    ax[1].set_ylabel('V (v)', fontsize=fsize)
    ax[1].legend(fontsize=fsize)

    ax[2].set_title('Potential along Z', fontsize=fsize)
    ax[2].grid()
    ax[2].plot(s.Z_roi * 1e3, potential_z_roi, label='original')
    ax[2].plot(s.Z_roi * 1e3, potential_z_regen + offset, label='regenerated + offset')
    ax[2].set_xlabel('Z ($\mu$m)', fontsize=fsize)
    ax[2].set_ylabel('V (v)', fontsize=fsize)
    ax[2].legend(fontsize=fsize)

    fig1.tight_layout(pad=1)

    plt.show()
# %% md
# plot potential projection in xy plane when apply voltage solution of a group of multipole coefficients
# %%
def plot_U2(s,xl,zl,roi,height, ey, ez, ex, u3, u2, u5, u1,u4,view='zy'):
    position1 = [xl, height * 1e-3, zl]
    s.update_origin_roi(position1, roi)
    multipole_coeffs = {'Ey': ey, 'Ez': ez, 'Ex': ex, 'U3': u3, 'U2': u2, 'U5': u5, 'U1': u1,'U4':u4}
    voltages = s.setMultipoles(multipole_coeffs)
    print(np.shape(voltages))
    potential_roi = s.potentialControl_roi(voltages)
    potential_regen = s.potentialControl_regen(voltages)
    # nearestZ = find_nearest(s.Z, height * 1e-3)
    # indNearestZ_roi = np.abs(s.Z_roi - nearestZ).argmin()
    if view == 'zy':
        nearestX = find_nearest(s.X, height * 1e-3)
        indNearestX_roi = np.abs(s.X_roi - nearestX).argmin()
        potential_xy_roi = potential_roi[indNearestX_roi]
        potential_xy_regen = potential_regen[indNearestX_roi]
    else:
        nearestZ = find_nearest(s.Z, height * 1e-3)
        indNearestZ_roi = np.abs(s.Z_roi - nearestZ).argmin()
        potential_xy_roi = potential_roi[:,:,indNearestZ_roi]
        potential_xy_regen = potential_regen[:,:,indNearestZ_roi]
    fsize = 10

    fig1 = plt.figure(figsize=(20, 16))
    grid = plt.GridSpec(2, 2)
    ax1 = fig1.add_subplot(grid[0, 0])
    ax2 = fig1.add_subplot(grid[0, 1])
    ax3 = fig1.add_subplot(grid[1, :])
    ax1.set_title('Simulated potential', fontsize=fsize)
    levels1 = np.linspace(np.amin(potential_xy_roi), np.amax(potential_xy_roi), 100)
    plot1 = ax1.contourf(s.X_roi * 1e3, s.Y_roi * 1e3, potential_xy_roi, levels1, cmap=plt.cm.viridis)
    plot1_line = ax1.contour(s.Z_roi * 1e3, s.Y_roi * 1e3, potential_xy_roi, colors='w')
    #     ax1.clabel(plot1_line, inline = 1, fontsize = fsize)
    ax1.clabel(plot1_line, colors='w', fmt='%2.3f', fontsize=fsize)
    ax1.set_xlabel('X ($\mu$m)', fontsize=fsize)
    ax1.set_ylabel('Y ($\mu$m)', fontsize=fsize)
    plt.colorbar(plot1, ax=ax1)

    ax2.set_title('Regenerated potential', fontsize=fsize)
    levels2 = np.linspace(np.amin(potential_xy_regen), np.amax(potential_xy_regen), 100)
    plot2 = ax2.contourf(s.X_roi * 1e3, s.Y_roi * 1e3, potential_xy_regen, levels2, cmap=plt.cm.viridis)
    plot2_line = ax2.contour(s.X_roi * 1e3, s.Y_roi * 1e3, potential_xy_regen, colors='w')
    #     ax2.clabel(plot2, inline = 1, fontsize = fsize)
    ax2.clabel(plot2_line, colors='w', fmt='%2.3f', fontsize=fsize)
    ax2.set_xlabel('X ($\mu$m)', fontsize=fsize)
    plt.colorbar(plot2, ax=ax2)

    coeffs = s.setVoltages(voltages)
    #     print(coeffs.index)
    ax3.bar(np.arange(8), np.asarray(coeffs)[0:8])
    max_coeff = np.max(coeffs)
    min_coeff = np.min(coeffs)
    margin = (max_coeff - min_coeff) * 0.1
    ymax = max_coeff + margin
    ymin = min_coeff - margin
    ax3.set_ylim(ymin, ymax)
    add_value_labels(ax3)
    ax3.set_xticks(range(9))
    ax3.set_xlim(-1,9)
    ax3.set_ylim(0,11)
    ax3.set_xticklabels(s.multipole_names, rotation=-90, fontsize=fsize)
    plt.show()
# %%
