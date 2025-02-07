import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pickle
import seaborn as sns
import h5py
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

def run_radial_vertical_div(
    mesh_path, start_frames, end_frames, save_path, key, y_min=None, y_max=None
):
    for i, item in enumerate(start_frames):
        start_disp = "./data/disp_data/" + start_frames[i] + "_extrap_hiRes.txt"
        end_disp = "./data/disp_data/" + end_frames[i] + "_extrap_hiRes.txt"

        # Load residual object
        residual_obj = residuals.from_pickle(
            mesh_path=mesh_path, start_path=start_disp, end_path=end_disp, key=key
        )

        # Calculate divergence and gradients
        u_rsdl, v_rsdl, w_rsdl = residual_obj.calc_residuals()
        vorticity = residual_obj.vorticity()
        div = residual_obj.divergence()
        grad_r = residual_obj.calc_gradient()[0]
        grad_y = residual_obj.calc_gradient()[8]
        xMesh, yMesh, zMesh = residual_obj.calc_coordinates()
        xMesh = xMesh * 2 * radius
        yMesh = yMesh * 2 * radius
        zMesh = zMesh * 2 * radius

        # Apply y-range filtering if specified
        if y_min is not None and y_max is not None:
            y_grid = yMesh[:, 0]
            indices = np.where((y_grid >= y_min) & (y_grid <= y_max))[0]
            xMesh = xMesh[indices, :]
            yMesh = yMesh[indices, :]
            zMesh = zMesh[indices, :]
            u_rsdl = u_rsdl[indices, :]
            v_rsdl = v_rsdl[indices, :]
            w_rsdl = w_rsdl[indices, :]
            div = div[indices, :]
            grad_r = grad_r[indices, :]
            grad_y = grad_y[indices, :]
            vorticity = vorticity[indices, :]

        # Create save paths for HDF5 files
        save_div = os.path.join(
            save_path, f"{start_frames[i]}_{end_frames[i]}_{key}_div.h5"
        )
        save_vorticity = os.path.join(
            save_path, f"{start_frames[i]}_{end_frames[i]}_{key}_vorticity.h5"
        )
        save_grad_r = os.path.join(
            save_path, f"{start_frames[i]}_{end_frames[i]}_{key}_grad_r.h5"
        )
        save_grad_y = os.path.join(
            save_path, f"{start_frames[i]}_{end_frames[i]}_{key}_grad_y.h5"
        )
        save_rsdl = os.path.join(
            save_path, f"{start_frames[i]}_{end_frames[i]}_{key}_rsdl.h5"
        )

        # Save divergence data to HDF5
        with h5py.File(save_div, "w") as f:
            f.create_dataset("xMesh", data=xMesh)
            f.create_dataset("yMesh", data=yMesh)
            f.create_dataset("zMesh", data=zMesh)
            f.create_dataset("div", data=div)
        print(f"Saved divergence data to {save_div}")

        # Save vorticity data to HDF5
        with h5py.File(save_vorticity, "w") as f:
            f.create_dataset("xMesh", data=xMesh)
            f.create_dataset("yMesh", data=yMesh)
            f.create_dataset("zMesh", data=zMesh)
            f.create_dataset("vorticity", data=vorticity)
        print(f"Saved divergence data to {save_vorticity}")

        # Save grad_r data to HDF5
        with h5py.File(save_grad_r, "w") as f:
            f.create_dataset("xMesh", data=xMesh)
            f.create_dataset("yMesh", data=yMesh)
            f.create_dataset("zMesh", data=zMesh)
            f.create_dataset("grad_r", data=grad_r)
        print(f"Saved grad_r data to {save_grad_r}")

        # Save grad_y data to HDF5
        with h5py.File(save_grad_y, "w") as f:
            f.create_dataset("xMesh", data=xMesh)
            f.create_dataset("yMesh", data=yMesh)
            f.create_dataset("zMesh", data=zMesh)
            f.create_dataset("grad_y", data=grad_y)
        print(f"Saved grad_y data to {save_grad_y}")

        # Save residuals data to HDF5
        with h5py.File(save_rsdl, "w") as f:
            f.create_dataset("xMesh", data=xMesh)
            f.create_dataset("yMesh", data=yMesh)
            f.create_dataset("zMesh", data=zMesh)
            f.create_dataset("u_rsdl", data=u_rsdl)
            f.create_dataset("v_rsdl", data=v_rsdl)
            f.create_dataset("w_rsdl", data=w_rsdl)
        print(f"Saved residuals data to {save_rsdl}")

def plot_strain_rotate(div, grad_y, vorticity, disp, degree, save_path):
    # read
    div = h5py.File(div, "r")
    grad_y = h5py.File(grad_y, "r")
    disp = h5py.File(disp, "r")
    vorticity = h5py.File(vorticity, "r")
    # grad_r = h5py.File(grad_r, "r")

    xMesh = div["xMesh"][()]
    yMesh = div["yMesh"][()]
    u_rsdl = disp["u_rsdl"][()]
    v_rsdl = disp["v_rsdl"][()]

    strain_div = div["div"][()]
    strain_grad_y = grad_y["grad_y"][()]
    strain_vorticity  = vorticity["vorticity"][()]
    # strain_grad_r = grad_r["grad_r"][()]

    theta = degree*np.pi/180
    xMeshRot = xMesh*np.cos(theta) + yMesh*np.sin(theta)
    yMeshRot = -xMesh*np.sin(theta)+ yMesh*np.cos(theta)
    uRot = u_rsdl*np.cos(theta) + v_rsdl*np.sin(theta)
    vRot = -u_rsdl*np.sin(theta)+ v_rsdl*np.cos(theta)

    # div Zoom
    masked_strain_div = np.ma.masked_where((strain_div > -0.00) & (strain_div < 0.03), strain_div)
    fig = plt.figure(figsize=(3, 3.5))
    ax_1 = fig.add_subplot(131)
    surf_1 = ax_1.contourf(xMeshRot, yMeshRot, masked_strain_div, levels=10, 
                           cmap=cm.RdBu_r, vmin=-0.10, vmax=0.14)
#    axzoom.set_xlabel("$x_{norm}$")
#    axzoom.set_ylabel("$y_{norm}$")
    ax_1.axis('scaled')
    ax_1.set_xlim(35, 55)
    # ax_1.set_ylim(35, 65)
    # axZoom_1.set_yticks([0.2, 0.3, 0.4])
#    axZoom_1.set_xticklabels([])
    
    # m1 = plt.cm.ScalarMappable(cmap=cm.jet_r)
    # m1.set_array(strain_div)
    # m1.set_clim(-0.10, 0.10)
    # fig.colorbar(m1,boundaries=np.round(np.linspace(-0.10, 0.10, 10), 2),
    #                  fraction=0.01, ticks=[-0.10, -0.05, 0.00, 0.05, 0.10])
    # fig.subplots_adjust(left=0.080)
    # plt.savefig(figDir + i + '_curlMean_r_zoom.tiff', dpi=600)
    # plt.close(figZoom_1)

    # grad_y Zoom
    # figZoom_2 = plt.figure(figsize=(6,1.5))
    masked_strain_y = np.ma.masked_where((strain_grad_y > -0.04) & (strain_grad_y < 0.1), strain_grad_y)
    ax_2 = fig.add_subplot(132)
    surf_2 = ax_2.contourf(xMeshRot, yMeshRot, masked_strain_y, 10, 
                           cmap=cm.RdYlBu, vmin=-0.12, vmax=0.1)
#    axzoom.set_xlabel("$x_{norm}$")
#    axzoom.set_ylabel("$y_{norm}$")
    ax_2.axis('scaled')
    # ax_2.set_ylim(35, 65)
    ax_2.set_xlim(35, 55)
    # axZoom_2.set_ylim(0.2, 0.4)
    # axZoom_2.set_yticks([0.2, 0.3, 0.4])
    ax_2.set_yticklabels([])
    
    # m2 = plt.cm.ScalarMappable(cmap=cm.jet_r)
    # m2.set_array(grad_y)
    # m2.set_clim(-0.15, 0.0)
    # fig.colorbar(m2,boundaries=np.round(np.linspace(-0.22,-0.1,7),2),
    #                  fraction=0.01, ticks=[-0.22, -0.18, -0.14, -0.10])
    # figZoom_2.subplots_adjust(left=0.080)
    # plt.savefig(figDir + i + '_gradMean_y_zoom.tiff', dpi=600)
    # plt.close(figZoom_2)

    # disp Zoom
    # figZoom_2 = plt.figure(figsize=(6,1.5))
    masked_strain_vorticity = np.ma.masked_where((strain_vorticity > -0.14) & (strain_vorticity < 0.02), strain_vorticity)
    ax_3 = fig.add_subplot(133)
    surf_32 = ax_3.contourf(xMeshRot, yMeshRot, masked_strain_vorticity, 5, alpha=1,
                           cmap=cm.Blues, vmin=-0.00, vmax=0.15)
    surf_31 = ax_3.quiver(xMeshRot[::4], yMeshRot[::4], uRot[::4], vRot[::4], angles='xy', units='xy', scale=0.10)
#    axzoom.set_xlabel("$x_{}$")
#    axzoom.set_ylabel("$y_{norm}$")
    ax_3.axis('scaled')
    # ax_3.set_ylim(35, 65)
    ax_3.set_xlim(35, 55)
    # axZoom_2.set_ylim(0.2, 0.4)
    # axZoom_2.set_yticks([0.2, 0.3, 0.4])
    ax_3.set_yticklabels([])
    
    # m2 = plt.cm.ScalarMappable(cmap=cm.jet_r)
    # m2.set_array(grad_y)
    # m2.set_clim(-0.15, 0.0)
    # fig.colorbar(m2,boundaries=np.round(np.linspace(-0.22,-0.1,7),2),
    #                  fraction=0.01, ticks=[-0.22, -0.18, -0.14, -0.10])
    # figZoom_2.subplots_adjust(left=0.080)
    # plt.savefig(figDir + i + '_gradMean_y_zoom.tiff', dpi=600)
    # plt.close(figZoom_2)

    plt.savefig(save_path, dpi=1000)

def main_gen_strain():
    # read vortices file in order
    mesh_path = "./data/mesh_hiRes.txt"
    save_path = "./data/strain_field/"

    start_frames = ["strain_1.0", "strain_3.0", "strain_5.0", "strain_7.0"]
    end_frames = ["strain_3.0", "strain_5.0", "strain_7.0", "strain_9.0"]
    key = "Z40C_092903b-192"

    run_radial_vertical_div(
        mesh_path=mesh_path,
        start_frames=start_frames,
        end_frames=end_frames,
        save_path=save_path,
        key=key,
        y_min=30,
        y_max=130,
    )

def main_plot_strain():
    div_h5 = "./data/strain_field/strain_5.0_strain_7.0_Z40C_092903b-192_div.h5"
    grad_y_h5 = "./data/strain_field/strain_5.0_strain_7.0_Z40C_092903b-192_grad_y.h5"
    grad_r_h5 = "./data/strain_field/strain_5.0_strain_7.0_Z40C_092903b-192_grad_r.h5"
    disp_h5 = "./data/strain_field/strain_5.0_strain_7.0_Z40C_092903b-192_rsdl.h5"
    vorticity_h5 = "./data/strain_field/strain_5.0_strain_7.0_Z40C_092903b-192_vorticity.h5"

    # with open(disp_pickle, "rb") as file:
    #     disp = pickle.load(file)

    plot_strain_rotate(
        div=div_h5,
        grad_y=grad_y_h5,
        vorticity=vorticity_h5,
        disp=disp_h5,
        degree=30,
        save_path="./pics/strain_local/zoom_rotate.pdf"
    )

if __name__ == "__main__":
    # main_gen_strain()
    main_plot_strain()