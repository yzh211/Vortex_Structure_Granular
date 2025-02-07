from residuals import *
from strain_localization import *
from vort_localization import *

plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["figure.titlesize"] = 8
plt.rcParams["figure.figsize"] = (3.5, 2)

radius = 35.44  # mm

def main_gen_strain():
    # read vortices file in order
    mesh_path = "./data/mesh_hiRes.txt"
    save_path = "./data/strain_field/"

    start_frames = ["strain_1.0", "strain_3.0", "strain_5.0", "strain_7.0"]
    end_frames = ["strain_3.0", "strain_5.0", "strain_7.0", "strain_9.0"]
    key = "Z40C_100103a-192"

    run_radial_vertical_div(
        mesh_path=mesh_path,
        start_frames=start_frames,
        end_frames=end_frames,
        save_path=save_path,
        key=key,
        y_min=30,
        y_max=130,
    )

    run_strain_vtk(
        mesh_path=mesh_path,
        start_frames=start_frames,
        end_frames=end_frames,
        save_path=save_path,
        key=key,
        y_min=30,
        y_max=130,
    )


if __name__ == "__main__":
    main_gen_strain()