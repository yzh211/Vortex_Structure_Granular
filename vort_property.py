import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from vortex2paraview import VorticesOutput
import more_itertools
import seaborn as sns
from array2vtk import write_vtk_points, interp_zMesh_nearest
from sklearn.neighbors import KernelDensity


plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["figure.titlesize"] = 8
plt.rcParams["figure.figsize"] = (3.5, 2)


# class Vortices_property(VorticesOutput):
class Vortices_property:

    def __init__(self, vortices, mesh, xmin=5, xmax=55, ymin=20, ymax=100):

        # super(Vortices_property, self).__init__(
        #     mesh, vortices, xmin, xmax, ymin, ymax
        # )

        """NOTE: mesh needs to be regular"""
        self.vortices = vortices
        self.name = vortices.filename.split("/")[-1].split(".h5")[0]
        self.mesh = mesh
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __repr__(self):
        return "{}".format(self.__class__.__name__)

    def calc_center(self):
        centers = []
        for vortex in self.vortices["vortices"]:
            cx = self.vortices["vortices"][vortex]["center"][
                0
            ]  # x-coordinate of the center
            cy = self.vortices["vortices"][vortex]["center"][
                1
            ]  # y-coordinate of the center

            # find the coordinates
            center_x = self.mesh["xMesh"][int(cy), int(cx)]
            center_y = self.mesh["yMesh"][int(cy), int(cx)]

            centers.append([center_x, center_y])

        return centers

    def calc_rortex(self):
        """
        Filter vortices to include only those with y-coordinates within the specified range.
        """
        # dx = self.vortices["params"]["grid_dx"][0]
        # dy = self.vortices["params"]["grid_dx"][1]
        # dx = 1
        # dy = 1
        # dr = np.sqrt(dx**2 + dy**2)
        rortex = []
        for vortex in self.vortices["vortices"]:
            cx = self.vortices["vortices"][vortex]["center"][
                0
            ]  # y-coordinate of the center
            cy = self.vortices["vortices"][vortex]["center"][
                1
            ]  # y-coordinate of the center
            if self.xmin <= cx <= self.xmax:
                if self.ymin <= cy <= self.ymax:
                    rortex_matrix = self.vortices["vortices"][vortex]["rortex"][:]
                    rortex_avg = np.mean(rortex_matrix)
                    rortex.append(rortex_avg)

        return rortex

    def calc_radii(self):
        """
        Filter vortices to include only those with y-coordinates within the specified range.
        """
        # dx = self.vortices["params"]["grid_dx"][0]
        # dy = self.vortices["params"]["grid_dx"][1]
        # dx = 1
        # dy = 1
        # dr = np.sqrt(dx**2 + dy**2)
        radii = []
        for vortex in self.vortices["vortices"]:
            cx = self.vortices["vortices"][vortex]["center"][
                0
            ]  # y-coordinate of the center
            cy = self.vortices["vortices"][vortex]["center"][
                1
            ]  # y-coordinate of the center
            if self.xmin <= cx <= self.xmax:
                if self.ymin <= cy <= self.ymax:
                    radii.append(self.vortices["vortices"][vortex]["radius"][0])

        return radii


def sum_vortices_centers(
    vortices_folder,
    mesh_folder,
    intervals,
    key=".h5",
):
    """
    Collects vortex centers information based on specified strain intervals.

    Parameters:
    - vortices_folder (str): Path to the folder containing vortex .h5 files.
    - mesh_folder (str): Path to the folder containing corresponding mesh files.
    - intervals (list): List of strain intervals defining the collection stages.
    - key (str): File extension to look for (default is '.h5').

    Returns:
    - dict: A dictionary containing vortex centers grouped by strain intervals.
    """

    # Initialize a dictionary to store centers for each interval
    centers_by_interval = {
        f"{intervals[i]}-{intervals[i+1]}": [[], [], []]
        for i in range(len(intervals) - 1)
    }

    # Iterate through files in the vortices folder
    for subdir, dirs, files in os.walk(vortices_folder):
        for file in files:
            if file.endswith(key):
                file_path = os.path.join(subdir, file)

                # Extract strain values from the file name using regex
                match = re.match(r"strain_(\d+\.\d+)strain_(\d+\.\d+)", file)
                if match:
                    strain_start = float(match.group(1))
                    strain_end = float(match.group(2))

                # Extract strain values from the file name
                for i in range(len(intervals) - 1):
                    lower_strain = float(intervals[i].split("_")[1])
                    upper_strain = float(intervals[i + 1].split("_")[1])

                    # Include files where the strain start is within the current interval
                    if lower_strain <= strain_start < upper_strain:
                        try:
                            # Load corresponding mesh file in mesh folder
                            mesh_file_name = file.replace(".h5", "_regularMesh.txt")
                            mesh_path = os.path.join(mesh_folder, mesh_file_name)

                            # Ensure the mesh file exists
                            if not os.path.exists(mesh_path):
                                print(f"Mesh file not found: {mesh_path}")
                                continue

                            with open(mesh_path, "rb") as mesh_file:
                                mesh = pickle.load(mesh_file)
                                xMesh = mesh["xMesh"]
                                yMesh = mesh["yMesh"]
                                zMesh = mesh["zMesh"]

                            print(f"Processing: {file} with mesh {mesh_file_name}")

                            # Load vortex data from the .h5 file
                            with h5py.File(os.path.join(subdir, file), "r") as vortices:
                                # Assuming Vortices_property is a defined class elsewhere
                                vort_obj = Vortices_property(
                                    vortices=vortices, mesh=mesh
                                )
                                centers_xy = vort_obj.calc_center()

                                x = [p[0] for p in centers_xy]
                                y = [p[1] for p in centers_xy]
                                z = list(
                                    interp_zMesh_nearest(xMesh, yMesh, zMesh, x, y)
                                )

                                centers = [x, y, z]

                                # Store the center in the appropriate interval
                                interval_key = f"{intervals[i]}-{intervals[i+1]}"
                                centers_by_interval[interval_key][0].extend(x)
                                centers_by_interval[interval_key][1].extend(y)
                                centers_by_interval[interval_key][2].extend(z)

                        except Exception as e:
                            print(f"Error processing file {file}: {e}")

    return centers_by_interval


def process_center_xy(centers):
    x_sum = []
    y_sum = []

    # skip the first iteration
    first = True
    for key, value in centers.items():
        x = value[0]
        y = value[1]

        if first:
            first = False
            continue

        x_sum.extend(x)
        y_sum.extend(y)

    return x_sum, y_sum


def calc_rortex_radii(vortices_folder, mesh_folder, num_tests=12, key=".h5"):
    """
    Recursively reads all .h5 files in the specified directory and its subdirectories,
    """
    # Walk through all directories and files in the specified directory
    count = 0  # Initialize a counter
    rortex_list = []
    radii_list = []
    for subdir, dirs, files in os.walk(vortices_folder):
        rortex_list_temp = []
        radii_list_temp = []
        for file in files:
            if file.endswith(key):
                file_path = os.path.join(subdir, file)

                # Load corresponding mesh file in mesh folder
                mesh_file_name = file.replace(".h5", "_regularMesh.txt")
                mesh_path = os.path.join(mesh_folder, mesh_file_name)
                mesh = pickle.load(open(mesh_path, "rb"))
                print(mesh_file_name)

                # # Load non-regular mesh file in mesh folder
                # nonRegular_mesh_file_name = file.replace(".h5", "_Mesh.txt")
                # nonRegular_mesh_path = os.path.join(
                #     mesh_folder, nonRegular_mesh_file_name
                # )
                # nonRegular_mesh = pickle.load(open(nonRegular_mesh_path, "rb"))

                with h5py.File(file_path, "r") as vortices:
                    # vortices group
                    vort_obj = Vortices_property(vortices=vortices, mesh=mesh)
                    rortex = vort_obj.calc_rortex()
                    radii = vort_obj.calc_radii()

                rortex_list_temp.append(rortex)
                radii_list_temp.append(radii)

                count += 1
                if count == num_tests:
                    rortex_flat = list(more_itertools.flatten(rortex_list_temp))
                    radii_flat = list(more_itertools.flatten(radii_list_temp))
                    rortex_list.append(rortex_flat)
                    radii_list.append(radii_flat)
                    rortex_list_temp = []
                    radii_list_temp = []
                    count = 0

    return rortex_list, radii_list


def plt_rortex_radii_heatmap(rortex, radii, save_path):
    x = np.arange(len(rortex))
    y = [np.mean(r) for r in radii]  # Using mean radii for each sublist as y-values
    X, Y = np.meshgrid(x, y)

    # Since rortex sublists can be different lengths, create a uniform grid for Z
    Z = np.full((len(y), len(x)), np.nan)  # Initialize with NaNs

    # Fill the Z grid with rortex values
    for xi, sublist in enumerate(rortex):
        for yi, val in enumerate(sublist):
            if yi < len(Z):
                Z[yi, xi] = val

    # Plotting
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, Z, shading="auto", cmap="PiYG")  # Adjust cmap as needed
    ax.set_title("Heatmap of Rortex Values")
    ax.set_xlabel("Sublist Index")
    ax.set_ylabel("Average Radii")

    # Plotting scatter plots on the same figure
    for xi, (sub_rortex, sub_radii) in enumerate(zip(rortex, radii)):
        ax.scatter(
            [xi] * len(sub_rortex),
            sub_radii,
            c=sub_rortex,
            cmap="viridis",
            edgecolor="w",
        )
    # Add a colorbar
    fig.colorbar(c, ax=ax)
    fig.savefig(save_path, dpi=1000)


def plot_sublist_statistics(rortex, radii, save_path=None):
    """
    Plots and optionally saves statistics about sublists: the count of values, average radii with standard deviation,
    and average rortex with standard deviation.

    Parameters:
    - rortex (list of lists): Each sublist contains rortex values.
    - radii (list of lists): Each sublist contains radii values, corresponding to the rortex lists.
    - save_path (str, optional): Path to save the plot image file, saves if provided.
    """
    # Calculating statistics
    counts = [len(sublist) for sublist in rortex]  # Number of values in each sublist
    mean_radii = [np.mean(sublist) for sublist in radii]
    std_radii = [np.std(sublist) for sublist in radii]
    mean_rortex = [np.mean(sublist) for sublist in rortex]
    std_rortex = [np.std(sublist) for sublist in rortex]
    x = np.arange(len(rortex))  # x-axis labels

    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))  # Three subplots

    # Number of values per sublist
    ax[0].plot(x, counts, marker="o", linestyle="-", color="b")
    ax[0].set_title("Number of Values per Sublist")
    ax[0].set_xlabel("Sublist Index")
    ax[0].set_ylabel("Count")
    ax[0].grid(True)

    # Average and standard deviation of radii
    ax[1].errorbar(x, mean_radii, yerr=std_radii, marker="o", linestyle="-", color="r")
    ax[1].set_title("Average Radii with Standard Deviation")
    ax[1].set_xlabel("Sublist Index")
    ax[1].set_ylabel("Average Radii")
    ax[1].grid(True)

    # Average and standard deviation of rortex
    ax[2].errorbar(
        x, mean_rortex, yerr=std_rortex, marker="o", linestyle="-", color="g"
    )
    ax[2].set_title("Average Rortex with Standard Deviation")
    ax[2].set_xlabel("Sublist Index")
    ax[2].set_ylabel("Average Rortex")
    ax[2].grid(True)

    # Improve layout to prevent overlap
    plt.tight_layout()

    # Save the figure if a save path is provided
    fig.savefig(save_path, dpi=1000)


def plot_vorticity(vorticity, save_path):
    """Plots a boxplot with significance annotations using statannotations."""
    # Colors from grey to black
    colors = [plt.cm.Greys(i / 9) for i in range(8)]

    # Prepare data
    data = [np.array(sublist) for sublist in vorticity]
    all_data = [item for sublist in data for item in sublist]
    group_labels = [f"{i}-{i+1}" for i in range(1, 9)]
    # group_labels = [f"Interval {i + 1}" for i in range(len(data))]
    repeated_labels = [
        label for i, label in enumerate(group_labels) for _ in range(len(data[i]))
    ]

    # Create the DataFrame for Seaborn
    import pandas as pd

    df = pd.DataFrame(
        {"Strain interval (%)": repeated_labels, "Vortices vorticity": all_data}
    )

    # Define comparisons
    comparisons = [
        ("1-2", "2-3"),
        ("3-4", "4-5"),
        ("3-4", "5-6"),
        ("6-7", "7-8"),
        ("6-7", "8-9"),
    ]

    # Define flier properties for smaller outliers
    flierprops = {"marker": "o", "color": "black", "alpha": 0.5, "markersize": 1}

    # Plot
    plt.figure(figsize=(3.6, 2))
    ax = sns.boxplot(
        x="Strain interval (%)",
        y="Vortices vorticity",
        data=df,
        palette=colors,
        width=0.5,
        showmeans=False,
        meanline=False,
        flierprops=flierprops,
        whis=2,
        medianprops={"color": "black", "lw": 0.4},  # Reduced median line width
        boxprops={"linewidth": 0.4},  # Reduced box line width
        whiskerprops={"linewidth": 0.4},  # Reduced whisker line width
        capprops={"linewidth": 0.4},  # Reduced cap line width
        # meanprops={"color": "red", "ls": "--", "lw": 1},
    )

    # # Add annotations
    # annotator = Annotator(ax, comparisons, data=df, x="Strain interval (%)", y="Hurst exponent")
    # annotator.configure(
    #     test="Mann-Whitney",
    #     text_format="star",
    #     loc="inside",
    #     verbose=2,
    #     fontsize=7,  # Reduced font size for annotator
    #     line_width=0.5  # Reduced line width for annotator
    # )
    # annotator.apply_and_annotate()

    # Customizations
    plt.xlabel("Strain intervals (%)", fontsize=8)
    plt.ylabel("Vortices vorticity", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)


def plot_radii(radii, save_path):
    """Plots a boxplot with significance annotations using statannotations."""
    # Colors from grey to black
    colors = [plt.cm.Greys(i / 9) for i in range(8)]

    # Prepare data
    data = [np.array(sublist) for sublist in radii]
    all_data = [item for sublist in data for item in sublist]
    group_labels = [f"{i}-{i+1}" for i in range(1, 9)]
    # group_labels = [f"Interval {i + 1}" for i in range(len(data))]
    repeated_labels = [
        label for i, label in enumerate(group_labels) for _ in range(len(data[i]))
    ]

    # Create the DataFrame for Seaborn
    import pandas as pd

    df = pd.DataFrame(
        {"Strain interval (%)": repeated_labels, "vortices radii (mm)": all_data}
    )

    # Define comparisons
    comparisons = [
        ("1-2", "2-3"),
        ("3-4", "4-5"),
        ("3-4", "5-6"),
        ("6-7", "7-8"),
        ("6-7", "8-9"),
    ]

    # Define flier properties for smaller outliers
    flierprops = {"marker": "o", "color": "black", "alpha": 0.5, "markersize": 1}

    # Plot
    plt.figure()
    ax = sns.boxplot(
        x="Strain interval (%)",
        y="vortices radii (mm)",
        data=df,
        palette=colors,
        width=0.5,
        showmeans=False,
        meanline=False,
        flierprops=flierprops,
        whis=2,
        medianprops={"color": "black", "lw": 0.4},  # Reduced median line width
        boxprops={"linewidth": 0.4},  # Reduced box line width
        whiskerprops={"linewidth": 0.4},  # Reduced whisker line width
        capprops={"linewidth": 0.4},  # Reduced cap line width
        # meanprops={"color": "red", "ls": "--", "lw": 1},
    )

    # # Add annotations
    # annotator = Annotator(ax, comparisons, data=df, x="Strain interval (%)", y="Hurst exponent")
    # annotator.configure(
    #     test="Mann-Whitney",
    #     text_format="star",
    #     loc="inside",
    #     verbose=2,
    #     fontsize=7,  # Reduced font size for annotator
    #     line_width=0.5  # Reduced line width for annotator
    # )
    # annotator.apply_and_annotate()

    # Customizations
    plt.xlabel("Strain intervals (%)", fontsize=8)
    plt.ylabel("vortices radii (mm)", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)


def plot_nested_pie(rortex, groups, save_path):
    """
    Plots a nested pie chart of rortex data grouped by specified indices using a grey-to-black colormap.

    Parameters:
    - rortex (list of lists): The main data where each sublist has multiple items.
    - groups (list of tuples): Each tuple contains indices of rortex that form a group.
    - save_path (str): Path to save the resulting plot.
    """
    # Set the aesthetic style of the plots
    sns.set(style="white")  # Apply seaborn styles for improved aesthetics

    # Calculate counts for each sublist and each group
    sizes = [len(rortex[i]) for group in groups for i in group]
    labels = [f"Sublist {i+1}" for group in groups for i in group]
    group_sizes = [sum(len(rortex[i]) for i in group) for group in groups]
    group_labels = [f"Group {i+1}" for i, group in enumerate(groups)]

    # Generate colors from grey to black for groups and sublists
    group_colors = [plt.cm.Greys(0.3 + i * 0.2) for i in range(len(groups))]
    sublist_colors = [plt.cm.Greys(0.2 + i * 0.05) for i in range(len(sizes))]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Outer ring - Groups
    wedges, _ = ax.pie(
        group_sizes,
        labels=group_labels,
        radius=1.4,
        colors=group_colors,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=1),
    )

    # Inner ring - Sublists
    ax.pie(
        sizes,
        labels=labels,
        radius=0.9,
        colors=sublist_colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=1),
    )

    # Ensure the pie chart is a circle
    ax.set(aspect="equal")

    # Save the figure
    plt.tight_layout()
    fig.savefig(save_path, dpi=1000)


def plot_bars(rortex, save_path):
    """
    Plots a simple horizontal bar plot of rortex data using a grey-to-black colormap.

    Parameters:
    - rortex (list of lists): The main data where each sublist has multiple items.
    - save_path (str): Path to save the resulting plot.
    """
    # # Set the aesthetic style of the plots
    # sns.set(style="whitegrid")  # Apply seaborn styles for improved aesthetics

    # Calculate counts for each sublist
    sizes = [len(sublist) for sublist in rortex]
    labels = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9"]

    # Generate colors from grey to black
    colors = [plt.cm.Greys(0.3 + i / 13) for i in range(8)]

    # Plot the horizontal bars
    fig, ax = plt.subplots(figsize=(2, 3.7))
    y_positions = np.arange(len(rortex))

    ax.barh(y_positions, sizes, color=colors, edgecolor="white", height=0.7)

    # # Add labels to the bars
    # for i, size in enumerate(sizes):
    #     ax.text(size + 0.1, y_positions[i], f"{size}", va='center', fontsize=10)

    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Count of Vortices", fontsize=8)
    ax.set_ylabel("Strain intervals (%)", fontsize=8)

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)

    # """
    # Plots a nested pie chart of rortex data grouped by specified indices.

    # Parameters:
    # - rortex (list of lists): The main data where each sublist has multiple items.
    # - groups (list of tuples): Each tuple contains indices of rortex that form a group.
    # - title (str): Title for the chart.
    # """
    # # Set the aesthetic style of the plots
    # sns.set(style="white")  # Apply seaborn styles for improved aesthetics

    # # Calculate counts for each sublist and each group
    # sizes = [len(rortex[i]) for group in groups for i in group]
    # labels = [f"Sublist {i+1}" for group in groups for i in group]
    # group_sizes = [sum(len(rortex[i]) for i in group) for group in groups]
    # group_labels = [f"Group {i+1}" for i, group in enumerate(groups)]

    # # Color    base_palette = sns.color_palette("pastel", len(groups))
    # # base_palette = sns.color_palette("pastel", len(groups))
    # # group_colors = [base_palette[i] for i in range(len(groups))]
    # # sublist_colors = [color for i, color in enumerate(base_palette) for _ in groups[i]]
    # base_palette = sns.color_palette("tab20c")
    # group_colors = [base_palette[0], base_palette[4], base_palette[12]]
    # sublist_colors = [
    #     base_palette[1],
    #     base_palette[2],
    #     base_palette[5],
    #     base_palette[6],
    #     base_palette[7],
    #     base_palette[13],
    #     base_palette[14],
    #     base_palette[15],
    # ]

    # fig, ax = plt.subplots()
    # # Outer ring - Groups
    # ax.pie(
    #     group_sizes,
    #     labels=group_labels,
    #     radius=1.4,
    #     colors=group_colors,
    #     wedgeprops=dict(width=0.5, edgecolor="w"),
    # )

    # # Inner ring - Sublists
    # ax.pie(
    #     sizes,
    #     labels=labels,
    #     radius=0.9,
    #     colors=sublist_colors,
    #     wedgeprops=dict(width=0.4, edgecolor="w"),
    # )

    # ax.set(aspect="equal")
    # fig.savefig(save_path, dpi=1000)


def plot_vortices_2D(x, y, save_path):

    def fmt(x, pos):
        return "{0:.3f}".format(x)

    def kde2D(x, y, bandwidth, xbins=50j, ybins=50j, **kwargs):
        """Build 2D kernel density estimate (KDE)."""
        # x = np.array(x)
        # y = np.array(y)
        # create grid of sample locations (default: 100x100)
        xx, yy = np.mgrid[-30:30:xbins, 0:153.81:ybins]

        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
        xy_train = np.vstack([y, x]).T

        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(xy_train)

        # score_samples() returns the log-likelihood of the samples
        z = np.exp(kde_skl.score_samples(xy_sample))
        return xx, yy, np.reshape(z, xx.shape)

    fig, ax = plt.subplots(figsize=(7, 4))
    # ax.set_facecolor('#440154FF')
    xx, yy, zz = kde2D(x, y, 5)
    h = ax.pcolormesh(xx, yy, zz, cmap="Greys")
    ax.scatter(x, y, s=4, lw=0, facecolor="black")
    # ax.plot(data.iloc[:,i], data.iloc[:,0], lw=0,
    #         marker='o', color='white', markersize=0.1)
    ax.set_xlabel("x (mm)")
    # plt.tight_layout()
    ax.set_ylabel("y (mm)")
    # ax.xaxis.set_ticks(np.linspace(np.min(xx), round(np.max(xx)), 5, endpoint=True))
    ax.set_xlim(left=np.min(xx), right=np.max(xx))
    ax.set_ylim(bottom=np.min(yy), top=np.max(yy))
    ax.set_aspect("equal")
    plt.colorbar(h, ax=ax, format=ticker.FuncFormatter(fmt))
    plt.tight_layout()

    plt.savefig(save_path, dpi=1000)


def main():
    # read vortices file in order
    vortices_folder = "./data/fine_vortices_h5/"
    mesh_folder = "./data/reMesh_mesh/"
    heatmap_path = "./pics/strain_vortices_heatMap.jpg"
    statsPlot_path = "./pics/strain_vortices_stats.jpg"
    count_vortices_path = "./pics/non_affine_strain/count_vortices.pdf"
    vortices_radii_path = "./pics/non_affine_strain/vortices_radii.pdf"
    vortices_vorticity_path = "./pics/non_affine_strain/vortices_vorticity.pdf"
    vortices_2d_path = "./pics/non_affine_strain/vortices_2D.pdf"

    vorticity, radii = calc_rortex_radii(
        vortices_folder=vortices_folder, mesh_folder=mesh_folder
    )

    # plot_bars(
    #     rortex=rortex,
    #     save_path=count_vortices_path,
    # )

    # plot_radii(radii, save_path=vortices_radii_path)
    plot_vorticity(vorticity, save_path=vortices_vorticity_path)

    # intervals=["strain_1.0", "strain_3.0", "strain_6.0", "strain_9.0"]
    # centers_intervals = sum_vortices_centers(
    #     vortices_folder=vortices_folder,
    #     mesh_folder=mesh_folder,
    #     intervals=intervals
    # )

    # x, y = process_center_xy(centers_intervals)
    # plot_vortices_2D(x, y, save_path=vortices_2d_path)

    # for key, value in centers_intervals.items():
    #     fileName = "./data/vortices_intervals/" + key + "_vortices.vtk"
    #     x = np.array(value[0])
    #     y = np.array(value[1])
    #     z = np.array(value[2])

    # write_vtk_points(x, y, z, filename=fileName)

    # plt_rortex_radii_heatmap(rortex, radii, heatmap_path)
    # plot_sublist_statistics(rortex, radii, statsPlot_path)
    # plot_nested_pie(
    #     rortex=rortex,
    #     groups=[(0, 1), (2, 3, 4), (5, 6, 7)],
    #     save_path=count_vortices_path,
    # )


if __name__ == "__main__":
    main()
