import os
import h5py
import numpy as np
import pickle
from scipy.interpolate import Rbf, griddata, NearestNDInterpolator


class VorticesOutput:
    def __init__(self, vortices, mesh, xmin=5, xmax=55, ymin=20, ymax=100):
        """NOTE: mesh needs to be regular"""
        self.vortices = vortices
        self.mesh = mesh
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.filtered_centers = []
        self.filtered_radii = []
        self.filtered_orientations = []
        self.filtered_centers_indices = []
        self.filtered_radius_lenIndex = []

        self.filter_vortices()

    def filter_vortices(self):
        """
        Filter vortices to include only those with y-coordinates within the specified range.
        """
        # dx = self.vortices["params"]["grid_dx"][0]
        # dy = self.vortices["params"]["grid_dx"][1]
        dx = 1
        dy = 1
        dr = np.sqrt(dx**2 + dy**2)

        for vortex in self.vortices["vortices"]:
            cx = self.vortices["vortices"][vortex]["center"][
                0
            ]  # y-coordinate of the center
            cy = self.vortices["vortices"][vortex]["center"][
                1
            ]  # y-coordinate of the center
            if self.xmin <= cx <= self.xmax:
                if self.ymin <= cy <= self.ymax:
                    self.filtered_centers.append(
                        self.vortices["vortices"][vortex]["center"]
                    )
                    self.filtered_radii.append(
                        self.vortices["vortices"][vortex]["radius"]
                    )
                    self.filtered_orientations.append(
                        self.vortices["vortices"][vortex]["orientation"][0]
                    )

                    # determine centers indices
                    x_index = round(cx / dx)
                    y_index = round(cy / dy)
                    # # x axis needs to be flipped due to the default setting of swirl
                    # x_len = self.mesh["xMesh"].shape[1]
                    # x_index = x_len - x_index
                    self.filtered_centers_indices.append(list((x_index, y_index)))

                    # Determine the length index for radius
                    dr_index = self.vortices["vortices"][vortex]["radius"]
                    self.filtered_radius_lenIndex.append(dr_index)

    def vortices_nonRegular_mesh(self, nonRegular_mesh):
        "Find the center of vortices in non-regular mesh"
        vortices_nonRegularMesh = []
        xMesh = np.flip(nonRegular_mesh["xMesh"], axis=1)
        yMesh = np.flip(nonRegular_mesh["yMesh"], axis=1)
        for vortex in self.filtered_centers_indices:
            x = xMesh[vortex[1], vortex[0]]
            y = yMesh[vortex[1], vortex[0]]
            vortices_nonRegularMesh.append(list((x, y)))

        return vortices_nonRegularMesh

    def vortices_nonRegular_radius(self, nonRegular_mesh):
        "Find the radius on non-regular mesh"
        xMesh = np.flip(nonRegular_mesh["xMesh"], axis=1)
        yMesh = np.flip(nonRegular_mesh["yMesh"], axis=1)
        vortices_nonRegularRadius = []
        for i, vortex in enumerate(self.filtered_centers_indices):
            # Using averaged distance in the adjacent points
            dx = (
                (xMesh[vortex[1], vortex[0] - 1] - xMesh[vortex[1], vortex[0]])
                + (xMesh[vortex[1], vortex[0]] - xMesh[vortex[1], vortex[0] + 1])
            ) / (2)
            dy = (
                (yMesh[vortex[1] - 1, vortex[0]] - yMesh[vortex[1], vortex[0]])
                + (yMesh[vortex[1], vortex[0]] - yMesh[vortex[1] + 1, vortex[0]])
            ) / (2)
            dr = np.sqrt(dx**2 + dy**2)
            r = dr * self.filtered_radius_lenIndex[i]
            vortices_nonRegularRadius.append(r)

        return vortices_nonRegularRadius

    def interp_zMesh_nearest(self, x, y, nonRegular_mesh):
        # Points where the data is defined
        points = np.column_stack(
            (nonRegular_mesh["xMesh"].flatten(), nonRegular_mesh["yMesh"].flatten())
        )

        # Values at the points
        values = nonRegular_mesh["zMesh"].flatten()

        # Interpolator
        interpolator = NearestNDInterpolator(points, values)

        # Perform extrapolation
        z = interpolator(x, y)
        return z

    def interp_zMesh(self, x, y, nonRegular_mesh):
        """
        Perform linear interpolation of the zMesh at points (x, y).

        Parameters:
        - x, y: Coordinates at which to interpolate. These can be scalar values or arrays.

        Returns:
        - z: Interpolated z-values at the specified x and y coordinates.
        """
        # Points where the data is defined
        points = np.column_stack(
            (nonRegular_mesh["xMesh"].flatten(), nonRegular_mesh["yMesh"].flatten())
        )

        # Values at the points
        values = nonRegular_mesh["zMesh"].flatten()

        # Points where we want to interpolate
        xi = np.column_stack((x, y))

        # Perform linear interpolation
        z = griddata(points, values, xi, method="linear")

        return z

    def interp_zMesh(self, x, y, nonRegular_mesh):
        interpolator = Rbf(
            nonRegular_mesh["xMesh"].flatten(),
            nonRegular_mesh["yMesh"].flatten(),
            nonRegular_mesh["zMesh"].flatten(),
            function="linear",
        )

        z = interpolator(x, y)

        return z

    def write_to_vtk(self, filename, nonRegular_mesh, num_points=30):
        """
        Write the filtered circle data to a VTK file, including orientation as cell data.

        Parameters:
        - filename: Name of the file to save.
        - num_points: Number of points to use for each circle.
        """
        with open(filename, "w") as file:
            file.write("# vtk DataFile Version 3.0\n")
            file.write("Vortex Circles with Orientation\n")
            file.write("ASCII\n")
            file.write("DATASET POLYDATA\n")

            total_points = num_points * len(self.filtered_centers)
            file.write(f"POINTS {total_points} float\n")

            vortices = self.vortices_nonRegular_mesh(nonRegular_mesh=nonRegular_mesh)
            radii = self.vortices_nonRegular_radius(nonRegular_mesh=nonRegular_mesh)
            # Generate and write each circle's points
            for center, radius in zip(vortices, radii):
                angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
                x = center[0] + radius * np.cos(angles)
                y = center[1] + radius * np.sin(angles)
                # z = self.interp_zMesh(x, y, nonRegular_mesh)
                z = self.interp_zMesh_nearest(x, y, nonRegular_mesh)
                for px, py, pz in zip(x, y, z):
                    file.write(f"{px} {py} {pz}\n")

            # Write polygon data
            file.write(
                f"POLYGONS {len(self.filtered_centers)} {len(self.filtered_centers) * (num_points + 1)}\n"
            )
            for i in range(len(self.filtered_centers)):
                indices = range(i * num_points, (i + 1) * num_points)
                file.write(f"{num_points} " + " ".join(map(str, indices)) + "\n")

            # Write orientation as cell data
            file.write("CELL_DATA {}\n".format(len(self.filtered_centers)))
            file.write("SCALARS orientation int 1\n")
            file.write("LOOKUP_TABLE default\n")
            for orientation in self.filtered_orientations:
                file.write(f"{int(orientation)}\n")


def work_h5_files(vortices_folder, mesh_folder, vortices_vtk_folder, key=".h5"):
    """
    Recursively reads all .h5 files in the specified directory and its subdirectories,
    """
    # Walk through all directories and files in the specified directory
    for subdir, dirs, files in os.walk(vortices_folder):
        for file in files:
            if file.endswith(key):
                file_path = os.path.join(subdir, file)

                # Load corresponding mesh file in mesh folder
                mesh_file_name = file.replace(".h5", "_regularMesh.txt")
                mesh_path = os.path.join(mesh_folder, mesh_file_name)
                mesh = pickle.load(open(mesh_path, "rb"))
                print(mesh_file_name)

                # Load non-regular mesh file in mesh folder
                nonRegular_mesh_file_name = file.replace(".h5", "_Mesh.txt")
                nonRegular_mesh_path = os.path.join(
                    mesh_folder, nonRegular_mesh_file_name
                )
                nonRegular_mesh = pickle.load(open(nonRegular_mesh_path, "rb"))

                # define vtk path
                file_name, extension = os.path.splitext(file)
                vtk_path = os.path.join(vortices_vtk_folder, file_name + ".vtk")

                with h5py.File(file_path, "r") as vortices:
                    # vortices group
                    vort_obj = VorticesOutput(vortices=vortices, mesh=mesh)
                    vort_obj.write_to_vtk(
                        filename=vtk_path, nonRegular_mesh=nonRegular_mesh
                    )

def write_circle_to_vtk(centers, radii, orientations, nonRegular_mesh, filename, num_points=30):
    """
    Write a circle defined by its center and radius to a VTK file.

    Parameters:
    - center (tuple): The (x, y) coordinates of the circle's center.
    - radius (float): The radius of the circle.
    - nonRegular_mesh (dict): The non-regular mesh data.
    - filename (str): Name of the VTK file to save.
    - num_points (int): Number of points to use for the circle.
    """
    def interp_zMesh_nearest(x, y, nonRegular_mesh):
        """
        Interpolate z-values for given x and y coordinates using the nearest neighbor method.

        Parameters:
        - x (array): X-coordinates.
        - y (array): Y-coordinates.
        - nonRegular_mesh (dict): The non-regular mesh data.

        Returns:
        - z (array): Interpolated Z-coordinates.
        """
        # Points where the data is defined
        nonRegular_mesh_file = pickle.load(open(nonRegular_mesh, "rb"))
        points = np.column_stack(
            (nonRegular_mesh_file["xMesh"].flatten(), nonRegular_mesh_file["yMesh"].flatten())
        )

        # Values at the points
        values = nonRegular_mesh_file["zMesh"].flatten()

        # Interpolator
        interpolator = NearestNDInterpolator(points, values)

        # Perform extrapolation
        z = interpolator(x, y)
        return z

    with open(filename, "w") as file:
        file.write("# vtk DataFile Version 3.0\n")
        file.write("Circles Data\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")

        # Calculate total points and polygons
        total_points = num_points * len(centers)
        total_polygons = len(centers)

        # Write total points
        file.write(f"POINTS {total_points} float\n")

        # Generate and write each circle's points
        point_offset = 0
        for center, radius in zip(centers, radii):
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            z = interp_zMesh_nearest(x, y, nonRegular_mesh)

            for px, py, pz in zip(x, y, z):
                file.write(f"{px} {py} {pz}\n")

            point_offset += num_points

        # Write polygons for each circle
        file.write(f"POLYGONS {total_polygons} {total_polygons * (num_points + 1)}\n")
        for i in range(len(centers)):
            indices = range(i * num_points, (i + 1) * num_points)
            file.write(f"{num_points} " + " ".join(map(str, indices)) + "\n")

        # Write orientation as cell data
        file.write(f"CELL_DATA {total_polygons}\n")
        file.write("SCALARS orientation int 1\n")
        file.write("LOOKUP_TABLE default\n")
        for orientation in orientations:
            file.write(f"{orientation}\n")

# def main():
#     # read vortices file in sequence
#     vortices_folder = "./data/vortices_h5/"
#     mesh_folder = "./data/reMesh_mesh/"
#     vortices_vtk_folder = "./data/vortices_vtk/"

#     work_h5_files(
#         vortices_folder=vortices_folder,
#         mesh_folder=mesh_folder,
#         vortices_vtk_folder=vortices_vtk_folder,
#     )

def main_adjust_vortices():
    # read vortices file in sequence
    filename = "./data/vortices_vtk_nature/strain_5.0_6.0_100103a_adjust.vtk"
    mesh = "./data/reMesh_mesh/strain_5.0strain_6.0_Z40C_100103a-192_regularMesh.txt"
    centers = [
        (-10, 101),
        (-20, 83),
        (-10, 71),
        (15, 65),
        (18, 102)
    ]
    radii = [
        3,
        2,
        3,
        4,
        3
    ]
    orientations=[
      1,
      1,
      -1,
      1,
      -1
    ]

    write_circle_to_vtk(
        centers=centers,
        radii=radii,
        nonRegular_mesh=mesh,
        filename=filename,
        orientations=orientations
    )

if __name__ == "__main__":
    # main()
    main_adjust_vortices()
