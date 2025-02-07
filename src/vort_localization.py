import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import seaborn as sns
from residuals import *
from array2vtk import *


plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["figure.titlesize"] = 8
plt.rcParams["figure.figsize"] = (3.5, 2)

radius = 35.44  # mm


def run_strain_vtk(
    mesh_path, start_frames, end_frames, save_path, key, y_min=None, y_max=None
):
    for i, item in enumerate(start_frames):
        start_disp = "./data/disp_data/" + start_frames[i] + "_extrap_hiRes.txt"
        end_disp = "./data/disp_data/" + end_frames[i] + "_extrap_hiRes.txt"

        residual_obj = residuals.from_pickle(
            mesh_path=mesh_path, start_path=start_disp, end_path=end_disp, key=key
        )

        div = residual_obj.divergence()
        vorticity = residual_obj.vorticity()
        grad_r = residual_obj.calc_gradient()[0]
        grad_y = residual_obj.calc_gradient()[8]
        xMesh, yMesh, zMesh = residual_obj.calc_coordinates()
        xMesh = xMesh * 2 * radius
        yMesh = yMesh * 2 * radius
        zMesh = zMesh * 2 * radius

        # save paths
        save_div = (
            save_path
            + start_frames[i]
            + "_"
            + end_frames[i]
            + "_"
            + key
            + "_"
            + "div.vtk"
        )
        save_r = (
            save_path
            + start_frames[i]
            + "_"
            + end_frames[i]
            + "_"
            + key
            + "_"
            + "grad_r.vtk"
        )
        save_y = (
            save_path
            + start_frames[i]
            + "_"
            + end_frames[i]
            + "_"
            + key
            + "_"
            + "grad_y.vtk"
        )
        save_vorticity = (
            save_path
            + start_frames[i]
            + "_"
            + end_frames[i]
            + "_"
            + key
            + "_"
            + "vorticity.vtk"
        )

        # Save localization into vtk files
        if y_min is not None and y_max is not None:
            y_grid = yMesh[:, 0]
            indices = np.where((y_grid >= y_min) & (y_grid <= y_max))[0]
            xMesh = xMesh[indices, ::]
            yMesh = yMesh[indices, ::]
            zMesh = zMesh[indices, ::]
            div = div[indices, ::]
            grad_r = grad_r[indices, ::]
            grad_y = grad_y[indices, ::]
            vorticity = vorticity[indices, :]

            # save paths
            save_div = (
                save_path
                + start_frames[i]
                + "_"
                + end_frames[i]
                + "_"
                + key
                + "_"
                + "div_clip.vtk"
            )
            save_r = (
                save_path
                + start_frames[i]
                + "_"
                + end_frames[i]
                + "_"
                + key
                + "_"
                + "grad_r_clip.vtk"
            )
            save_y = (
                save_path
                + start_frames[i]
                + "_"
                + end_frames[i]
                + "_"
                + key
                + "_"
                + "grad_y_clip.vtk"
            )
            save_vorticity = (
                save_path
                + start_frames[i]
                + "_"
                + end_frames[i]
                + "_"
                + key
                + "_"
                + "vorticity_clip.vtk"
            )
        surface_to_vtk(
            xmesh=xMesh, ymesh=yMesh, zmesh=zMesh, values=div, filename=save_div
        )
        surface_to_vtk(
            xmesh=xMesh, ymesh=yMesh, zmesh=zMesh, values=grad_r, filename=save_r
        )
        surface_to_vtk(
            xmesh=xMesh, ymesh=yMesh, zmesh=zMesh, values=grad_y, filename=save_y
        )
        surface_to_vtk(
            xmesh=xMesh, ymesh=yMesh, zmesh=zMesh, values=vorticity, filename=save_vorticity
        )

def main():
    # read vortices file in order
    mesh_path = "./data/mesh_hiRes.txt"
    pic_path = "./data/strain_field/"

    start_frames = ["strain_1.0", "strain_3.0", "strain_5.0", "strain_7.0"]
    end_frames = ["strain_3.0", "strain_5.0", "strain_7.0", "strain_9.0"]
    key = "Z40C_092903b-192"

    run_strain_vtk(
        mesh_path=mesh_path,
        start_frames=start_frames,
        end_frames=end_frames,
        save_path=pic_path,
        key=key,
        y_min=30,
        y_max=130,
    )


if __name__ == "__main__":
    main()
