import numpy as np
import scipy
from scipy import ndimage
import sys
import scipy.optimize as opt
from scipy.interpolate import Rbf
import pickle


def reMesh_non_regular(xMesh, yMesh, zMesh, uDisp, vDisp, wDisp, time_step):
    # reduce the mesh data
    x_mesh = xMesh[30:-30:time_step, 5:-5:time_step]
    y_mesh = yMesh[30:-30:time_step, 5:-5:time_step]
    z_mesh = zMesh[30:-30:time_step, 5:-5:time_step]

    # reduce displacement data
    u_disp = uDisp[30:-30:time_step, 5:-5:time_step]
    v_disp = vDisp[30:-30:time_step, 5:-5:time_step]
    w_disp = wDisp[30:-30:time_step, 5:-5:time_step]

    new_mesh = {"xMesh": x_mesh, "yMesh": y_mesh, "zMesh": z_mesh}

    new_disp = {
        "u_disp": u_disp,
        "v_disp": v_disp,
        "w_disp": w_disp,
    }

    return new_mesh, new_disp


def reMesh(xMesh, yMesh, zMesh, uDisp, vDisp, wDisp, time_step):
    # reduce the mesh data
    x_mesh = xMesh[30:-30, 5:-5]
    y_mesh = yMesh[30:-30, 5:-5]
    z_mesh = zMesh[30:-30, 5:-5]

    # reduce displacement data
    u_disp = uDisp[30:-30, 5:-5]
    v_disp = vDisp[30:-30, 5:-5]
    w_disp = wDisp[30:-30, 5:-5]

    # Define the uniform x grid based on the original x range
    x_uniform = np.linspace(x_mesh.min().min(), x_mesh.max().max(), x_mesh.shape[1])

    # Create new meshgrid for the uniform x direction and existing y
    x_new, y_new = np.meshgrid(x_uniform, y_mesh[:, 0])

    # Radius of the cylinder
    radius = np.sqrt(x_mesh**2 + z_mesh**2).max().max()

    # Calculate the new z mesh based on the uniform x and radius
    z_new = np.sqrt(radius**2 - x_new**2)

    # Interpolate displacements onto the new mesh
    interpolator_u = Rbf(
        x_mesh[::time_step, ::time_step],
        y_mesh[::time_step, ::time_step],
        z_mesh[::time_step, ::time_step],
        u_disp[::time_step, ::time_step],
        function="linear",
    )

    interpolator_v = Rbf(
        x_mesh[::time_step, ::time_step],
        y_mesh[::time_step, ::time_step],
        z_mesh[::time_step, ::time_step],
        v_disp[::time_step, ::time_step],
        function="linear",
    )

    interpolator_w = Rbf(
        x_mesh[::time_step, ::time_step],
        y_mesh[::time_step, ::time_step],
        z_mesh[::time_step, ::time_step],
        w_disp[::time_step, ::time_step],
        function="linear",
    )

    u_new = interpolator_u(
        x_new[::time_step, ::time_step],
        y_new[::time_step, ::time_step],
        z_new[::time_step, ::time_step],
    )

    v_new = interpolator_v(
        x_new[::time_step, ::time_step],
        y_new[::time_step, ::time_step],
        z_new[::time_step, ::time_step],
    )

    w_new = interpolator_w(
        x_new[::time_step, ::time_step],
        y_new[::time_step, ::time_step],
        z_new[::time_step, ::time_step],
    )

    xMesh = x_new[::time_step, ::time_step]
    yMesh = y_new[::time_step, ::time_step]
    zMesh = z_new[::time_step, ::time_step]
    u_disp = u_new
    v_disp = v_new
    w_disp = w_new

    new_mesh = {"xMesh": xMesh, "yMesh": yMesh, "zMesh": zMesh}

    new_disp = {
        "u_disp": u_disp,
        "v_disp": v_disp,
        "w_disp": w_disp,
    }

    return new_mesh, new_disp


def main():

    mesh_path = "./data/mesh_hiRes.txt"
    disp_path = "./data/strain_8.0_extrap_hiRes.txt"
    load_step = "Z40C_092903b-192"

    # re-mesh path
    mesh_save_path = "./data/mesh_strain_8.0_092903b-192.txt"
    disp_save_path = "./data/disp_strain_8.0_092903b-192_dispAll.txt"

    reMesh(
        mesh_path=mesh_path,
        disp_path=disp_path,
        load_step=load_step,
        time_step=2,
        mesh_save_path=mesh_save_path,
        disp_save_path=disp_save_path,
    )


if __name__ == "__main__":
    main()
