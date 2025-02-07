import numpy as np
import pickle
import sys
from pykrige.ok3d import OrdinaryKriging3D
from scipy.interpolate import Rbf


class VectorField:
    """
    Data file

    initialize the variables.

    :param file_path: file path
    :type  file_path: str
    :param time_step: current time step
    :type  time_step: int
    :param mean_file_path: in case of a mean field subtraction
    :type  mean_file_path: str
    :param file_type: 'piv_netcdf', 'dns, 'dns2', 'piv_tecplot', 'openfoam'
    :type  file_type: str
    :param x_coordinate_matrix: spatial mesh
    :type  x_coordinate_matrix: ndarray
    :param y_coordinate_matrix: spatial mesh
    :type  y_coordinate_matrix: ndarray
    :param z_coordinate_matrix: spatial mesh, optional
    :type  z_coordinate_matrix: ndarray
    :param u_velocity_matrix: 1st component velocity field
    :type  u_velocity_matrix: ndarray
    :param v_velocity_matrix: 2nd component velocity field
    :type  v_velocity_matrix: ndarray
    :param w_velocity_matrix: 3rd component velocity field, optional
    :type  w_velocity_matrix: ndarray
    :param normalization_flag: for normalization of the swirling field
    :type  normalization_flag: boolean
    :param normalization_direction: 'None', 'x' or 'y'
    :type  normalization_direction: str
    :param x_coordinate_step: for homogeneous mesh, provides a unique step
    :type  x_coordinate_step: float
    :param y_coordinate_step: for homogeneous mesh, provides a unique step
    :type  y_coordinate_step: float
    :param z_coordinate_step: for homogeneous mesh, provides a unique step
    :type  z_coordinate_step: float
    :param derivative: contains 'dudx', 'dudy', 'dvdx', 'dvdy'.
                       Can be extended to the 3rd dimension
    :type  derivative: dict
    :returns: vfield, an instance of the VelocityField class
    :rtype: class VelocityField
    """

    def __init__(
        self,
        mesh_path,
        disp_path,
        load_step,
        time_step=None,
        normalization_flag=False,
        normalization_direction=None,
        average_flag=False,
        average_direction=None,
    ):

        self.mesh_path = mesh_path
        self.time_step = time_step
        self.disp_path = disp_path
        self.load_step = load_step
        self.normalization_flag = normalization_flag
        self.normalization_direction = normalization_direction
        self.average_flag = average_flag
        self.average_direction = average_direction

        try:
            # Load mesh data
            mesh_whole = pickle.load(open(self.mesh_path, "rb"))
            xMesh = mesh_whole["xMesh"]
            yMesh = mesh_whole["yMesh"]
            zMesh = mesh_whole["zMesh"]

            # Load displacement data
            disp = pickle.load(open(self.disp_path, "rb"))
            u_disp = disp["u_disp"]
            v_disp = disp["v_disp"]
            w_disp = disp["w_disp"]
        except IOError:
            sys.exit("\nReading error. Maybe a wrong file type?\n")

        self.xMesh = xMesh[:: self.time_step, :: self.time_step]
        self.yMesh = yMesh[:: self.time_step, :: self.time_step]
        self.zMesh = zMesh[:: self.time_step, :: self.time_step]

        self.u_disp = u_disp[:: self.time_step, :: self.time_step]
        self.v_disp = v_disp[:: self.time_step, :: self.time_step]
        self.w_disp = w_disp[:: self.time_step, :: self.time_step]

        if self.average_flag:
            self.average(self.average_direction)

        if self.normalization_flag:
            self.normalize(self.normalization_direction)

        # coordinate size if mesh is homogeneous, could be useful
        # self.x_coordinate_size = self.u_velocity_matrix.shape[1]
        # self.y_coordinate_size = self.u_velocity_matrix.shape[0]
        # self.z_coordinate_size = 1

    # def reMesh(self):
    #     # Load mesh data
    #     mesh_whole = pickle.load(open(self.mesh_path, "rb"))
    #     x_mesh = mesh_whole["xMesh"]
    #     y_mesh = mesh_whole["yMesh"]
    #     z_mesh = mesh_whole["zMesh"]

    #     # Load displacement data
    #     disp = pickle.load(open(self.disp_path, "rb"))
    #     u_disp = disp[self.load_step][0]
    #     v_disp = disp[self.load_step][1]
    #     w_disp = disp[self.load_step][2]

    #     # Define the uniform x grid based on the original x range
    #     x_uniform = np.linspace(x_mesh.min().min(), x_mesh.max().max(), x_mesh.shape[1])

    #     # Create new meshgrid for the uniform x direction and existing y
    #     x_new, y_new = np.meshgrid(x_uniform, y_mesh[:, 0])

    #     # Radius of the cylinder
    #     radius = np.sqrt(x_mesh**2 + z_mesh**2).max().max()

    #     # Calculate the new z mesh based on the uniform x and radius
    #     z_new = np.sqrt(radius**2 - x_new**2)

    #     # NOTE: too slow
    #     # # Interpolate displacements onto the new mesh
    #     # interpolator_u = OrdinaryKriging3D(
    #     #     x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel(), u_disp.ravel(), variogram_model="linear"
    #     # )

    #     # interpolator_v = OrdinaryKriging3D(
    #     #     x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel(), v_disp.ravel(), variogram_model="linear"
    #     # )

    #     # interpolator_w = OrdinaryKriging3D(
    #     #     x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel(), w_disp.ravel(), variogram_model="linear"
    #     # )

    #     # Interpolate displacements onto the new mesh
    #     interpolator_u = Rbf(
    #         x_mesh[:: self.time_step, :: self.time_step],
    #         y_mesh[:: self.time_step, :: self.time_step],
    #         z_mesh[:: self.time_step, :: self.time_step],
    #         u_disp[:: self.time_step, :: self.time_step],
    #         function="linear"
    #     )

    #     interpolator_v = Rbf(
    #         x_mesh[:: self.time_step, :: self.time_step],
    #         y_mesh[:: self.time_step, :: self.time_step],
    #         z_mesh[:: self.time_step, :: self.time_step],
    #         v_disp[:: self.time_step, :: self.time_step],
    #         function="linear"
    #     )

    #     interpolator_w = Rbf(
    #         x_mesh[:: self.time_step, :: self.time_step],
    #         y_mesh[:: self.time_step, :: self.time_step],
    #         z_mesh[:: self.time_step, :: self.time_step],
    #         w_disp[:: self.time_step, :: self.time_step],
    #         function="linear"
    #     )

    #     u_new = interpolator_u(
    #         x_new[:: self.time_step, :: self.time_step],
    #         y_new[:: self.time_step, :: self.time_step],
    #         z_new[:: self.time_step, :: self.time_step],
    #     )

    #     v_new = interpolator_v(
    #         x_new[:: self.time_step, :: self.time_step],
    #         y_new[:: self.time_step, :: self.time_step],
    #         z_new[:: self.time_step, :: self.time_step],
    #     )

    #     w_new = interpolator_w(
    #         x_new[:: self.time_step, :: self.time_step],
    #         y_new[:: self.time_step, :: self.time_step],
    #         z_new[:: self.time_step, :: self.time_step],
    #     )
    #     # self.xMesh = x_new[:: self.time_step, ::self.time_step]
    #     # self.yMesh = y_new[:: self.time_step, ::self.time_step]
    #     # self.zMesh = z_new[:: self.time_step, ::self.time_step]
    #     # self.u_disp = u_new[:: self.time_step, ::self.time_step]
    #     # self.v_disp = v_new[:: self.time_step, ::self.time_step]
    #     # self.w_disp = w_new[:: self.time_step, ::self.time_step]

    #     self.xMesh = x_new[:: self.time_step, :: self.time_step]
    #     self.yMesh = y_new[:: self.time_step, :: self.time_step]
    #     self.zMesh = z_new[:: self.time_step, :: self.time_step]
    #     self.u_disp = u_new
    #     self.v_disp = v_new
    #     self.w_disp = w_new

    def normalize(self, axis):
        """
        normalize the displacement fields

        :param velocity_matrix: velocity field
        :type velocity_matrix: ndarray
        :param homogeneous_axis: False, 'x', or 'y'. The axis which the mean is subtracted
        :type homogeneous_axis: str

        :returns: normalized array
        :rtype: ndarray
        """
        if axis is None:
            self.u_disp = self.u_disp / np.sqrt(
                np.mean(self.u_disp**2, axis=1, keepdims=True)
            )
            self.v_disp = self.v_disp / np.sqrt(
                np.mean(self.v_disp**2, axis=0, keepdims=True)
            )
        elif axis == "x":
            self.u_disp = self.u_disp / np.sqrt(
                np.mean(self.u_disp**2, axis=1, keepdims=True)
            )
        elif axis == "y":
            self.v_disp = self.v_disp / np.sqrt(
                np.mean(self.v_disp**2, axis=0, keepdims=True)
            )
        else:
            sys.exit("Invalid homogeneity axis.")

    def average(self, axis):
        """
        Average the displacement fields

        :param velocity_matrix: velocity field
        :type velocity_matrix: ndarray
        :param homogeneous_axis: False, 'x', or 'y'. The axis which the mean is subtracted
        :type homogeneous_axis: str

        :returns: normalized array
        :rtype: ndarray
        """
        if axis is None:
            self.u_disp = self.u_disp - np.mean(self.u_disp, axis=1, keepdims=True)
            self.v_disp = self.v_disp - np.mean(self.v_disp, axis=0, keepdims=True)
        elif axis == "x":
            self.u_disp = self.u_disp - np.mean(self.u_disp, axis=1, keepdims=True)
        elif axis == "y":
            self.v_disp = self.v_disp - np.mean(self.v_disp, axis=0, keepdims=True)
        else:
            sys.exit("Invalid homogeneity axis.")


def main():
    mesh_path = "./data/mesh_hiRes.txt"
    disp_path = "./data/strain_8.0_residualDisp_all.txt"
    load_step = "Z40C_092903b-192"

    vfield = VectorField(
        mesh_path=mesh_path,
        disp_path=disp_path,
        load_step=load_step,
        time_step=2,
        reMesh=True,
    )


if __name__ == "__main__":
    main()
