import pickle
import numpy as np
from vtk import vtkStructuredGrid, vtkPoints, vtkDoubleArray, vtkStructuredGridWriter, vtkIntArray
import vtk
from residuals import residuals, cal_drdt, cart2pol
from remesh import reMesh, reMesh_non_regular
from vtk.util import numpy_support
import pyvista as pv

# from scipy.interpolate import NearestNDInterpolator


def write_vtk_points(x, y, z, filename):
    """
    Write x, y, z coordinates to a VTK file for visualization in ParaView.

    Parameters:
    - x: Array of x coordinates.
    - y: Array of y coordinates.
    - z: Array of z coordinates.
    - filename: Name of the output VTK file.
    """
    n_points = len(x)
    assert (
        len(y) == n_points and len(z) == n_points
    ), "x, y, and z arrays must be the same length."

    with open(filename, "w") as file:
        # VTK header
        file.write("# vtk DataFile Version 3.0\n")
        file.write("3D points\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")
        file.write(f"POINTS {n_points} float\n")

        # Write the point data
        for i in range(n_points):
            file.write(f"{x[i]} {y[i]} {z[i]}\n")

        # Specify that there are no lines or polygons (only points)
        file.write(f"VERTICES 1 {n_points + 1}\n")
        file.write(f"{n_points}")
        for i in range(n_points):
            file.write(f" {i}")
        file.write("\n")

def create_vtk_grid(mesh_x, mesh_y, mesh_z, disp_x, disp_y, disp_z, flag=False):
    """
    Create a VTK structured grid from mesh and displacement data.
    
    Parameters:
    - mesh_x (numpy.ndarray): X-coordinates of the mesh.
    - mesh_y (numpy.ndarray): Y-coordinates of the mesh.
    - mesh_z (numpy.ndarray): Z-coordinates of the mesh.
    - disp_x (numpy.ndarray): X-components of the displacement.
    - disp_y (numpy.ndarray): Y-components of the displacement.
    - disp_z (numpy.ndarray): Z-components of the displacement.
    - flag (bool): If True, adds a new attribute based on z-displacement.
    
    Returns:
    - vtkStructuredGrid: The structured grid with displacement vectors and optional z-displacement flag.
    """

    # Assuming mesh_x, mesh_y, mesh_z, disp_x, disp_y, disp_z are 2D arrays of the same shape
    rows, cols = mesh_x.shape
    grid = vtkStructuredGrid()
    grid.SetDimensions(rows, cols, 1)

    # Initialize points and displacement vectors
    points = vtkPoints()
    vectors = vtkDoubleArray()
    vectors.SetNumberOfComponents(3)  # For 3D displacement vectors
    vectors.SetName("Displacement")

    # Initialize z-displacement flag array if flag is True
    if flag:
        radial_disp_flag = vtkIntArray()
        radial_disp_flag.SetName("Z_Displacement_Flag")

    # Insert points and displacement vectors
    for i in range(rows):
        for j in range(cols):
            x, y, z = mesh_x[i, j], mesh_y[i, j], mesh_z[i, j]
            u, v, w = disp_x[i, j], disp_y[i, j], disp_z[i, j]
            
            points.InsertNextPoint(x, y, z)
            vectors.InsertNextTuple([u, v, w])
            
            # Add z-displacement flag if the flag is True
            if flag:
                radius = np.sqrt(x**2 + z**2)
                theta = np.arctan2(z, x)
                dr, dt = cal_drdt(u, w, theta)
                # Flag +1 for increasing radius, -1 for decreasing radius
                radial_disp_flag.InsertNextValue(1 if dr > 0 else -1)

    # Set points and vectors to the grid
    grid.SetPoints(points)
    grid.GetPointData().SetVectors(vectors)

    # Add z-displacement flag to the grid if the flag is True
    if flag:
        grid.GetPointData().AddArray(radial_disp_flag)

    return grid

# def create_vtk_grid(mesh_x, mesh_y, mesh_z, disp_x, disp_y, disp_z):
#     """Create a VTK structured grid from mesh and displacement data."""
#     # Assuming mesh_x, mesh_y, mesh_z, disp_x, disp_y, disp_z are 2D arrays of the same shape
#     rows, cols = mesh_x.shape
#     grid = vtkStructuredGrid()
#     grid.SetDimensions(rows, cols, 1)

#     points = vtkPoints()
#     vectors = vtkDoubleArray()
#     vectors.SetNumberOfComponents(3)  # For 3D displacement vectors
#     vectors.SetName("Displacement")

#     for i in range(rows):
#         for j in range(cols):
#             points.InsertNextPoint(mesh_x[i, j], mesh_y[i, j], mesh_z[i, j])
#             vectors.InsertNextTuple([disp_x[i, j], disp_y[i, j], disp_z[i, j]])

#     grid.SetPoints(points)
#     grid.GetPointData().SetVectors(vectors)

#     return grid


def save_vtk_file(vtk_data, file_name):
    """Save VTK structured grid data to a file."""
    writer = vtkStructuredGridWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(vtk_data)
    writer.Write()


def surface_to_vtk(xmesh, ymesh, zmesh, values, filename):
    """
    Exports xmesh, ymesh, zmesh, and values to a VTK file for visualization in ParaView.

    Args:
        xmesh (np.ndarray): 2D array of x coordinates.
        ymesh (np.ndarray): 2D array of y coordinates.
        zmesh (np.ndarray): 2D array of z coordinates.
        values (np.ndarray): 2D array of values at each point in the mesh.
        filename (str): The output filename (default: "output.vtk").
    """
    # Check if the input arrays have the same shape
    if not (xmesh.shape == ymesh.shape == zmesh.shape == values.shape):
        raise ValueError("All input arrays must have the same shape.")

    # Get the dimensions of the structured grid
    nx, ny = xmesh.shape

    # Create a VTK Structured Grid object
    structured_grid = vtk.vtkStructuredGrid()
    structured_grid.SetDimensions(nx, ny, 1)

    # Flatten the coordinate arrays and create VTK points
    points = vtk.vtkPoints()
    for i in range(nx):
        for j in range(ny):
            points.InsertNextPoint(xmesh[i, j], ymesh[i, j], zmesh[i, j])
    structured_grid.SetPoints(points)

    # Convert the values array to a VTK array
    vtk_values = numpy_support.numpy_to_vtk(
        num_array=values.ravel(), deep=True, array_type=vtk.VTK_FLOAT
    )
    vtk_values.SetName("Values")

    # Add the values array to the structured grid
    structured_grid.GetPointData().AddArray(vtk_values)

    # Write the structured grid to a VTK file
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(structured_grid)
    writer.Write()


def disp2vtk(
    mesh_path,
    start_path,
    end_path,
    save_path,
    key,
    radius=residuals.radius,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    radial_differential=False
):
    residuals_disp = residuals.from_pickle(
        mesh_path=mesh_path,
        start_path=start_path,
        end_path=end_path,
        key=key,
    )

    xMesh_N, yMesh_N, zMesh_N = residuals_disp.calc_coordinates()
    xMesh = xMesh_N * radius * 2
    yMesh = yMesh_N * radius * 2
    zMesh = zMesh_N * radius * 2

    # calculate displacement data
    u_rsdl, v_rsdl, w_rsdl = residuals_disp.calc_residuals()

    # remesh the displacement data
    new_mesh, new_disp = reMesh_non_regular(
        xMesh=xMesh,
        yMesh=yMesh,
        zMesh=zMesh,
        uDisp=u_rsdl,
        vDisp=v_rsdl,
        wDisp=w_rsdl,
        time_step=2,
    )

    new_xMesh = new_mesh["xMesh"]
    new_yMesh = new_mesh["yMesh"]
    new_zMesh = new_mesh["zMesh"]
    new_u = new_disp["u_disp"]
    new_v = new_disp["v_disp"]
    new_w = new_disp["w_disp"]

    # Apply x-axis constraints
    if x_min is not None and x_max is not None:
        x_grid = new_mesh["xMesh"][0, :]
        x_indices = np.where((x_grid >= x_min) & (x_grid <= x_max))[0]
        new_xMesh = new_xMesh[:, x_indices]
        new_yMesh = new_yMesh[:, x_indices]
        new_zMesh = new_zMesh[:, x_indices]
        new_u = new_u[:, x_indices]
        new_v = new_v[:, x_indices]
        new_w = new_w[:, x_indices]

    if y_min is not None and y_max is not None:
        y_grid = new_mesh["yMesh"][:, 0]
        indices = np.where((y_grid >= y_min) & (y_grid <= y_max))[0]
        new_xMesh = new_xMesh[indices, ::]
        new_yMesh = new_yMesh[indices, ::]
        new_zMesh = new_zMesh[indices, ::]
        new_u = new_u[indices, ::]
        new_v = new_v[indices, ::]
        new_w = new_w[indices, ::]

    if radial_differential:
        vtk_grid = create_vtk_grid(new_xMesh, new_yMesh, new_zMesh, new_u, new_v, new_w, flag=True)
        save_vtk_file(vtk_grid, save_path)
    else:
        vtk_grid = create_vtk_grid(new_xMesh, new_yMesh, new_zMesh, new_u, new_v, new_w)
        save_vtk_file(vtk_grid, save_path)

def create_vtk_grid_diff(xMesh, yMesh, zMesh, u_disp, v_disp, w_disp):
    """
    Creates a PyVista StructuredGrid object with displacement data.

    Parameters:
    - xMesh (numpy.ndarray): X-coordinates of the mesh.
    - yMesh (numpy.ndarray): Y-coordinates of the mesh.
    - zMesh (numpy.ndarray): Z-coordinates of the mesh.
    - u_disp (numpy.ndarray): X-components of the displacement.
    - v_disp (numpy.ndarray): Y-components of the displacement.
    - w_disp (numpy.ndarray): Z-components of the displacement.

    Returns:
    - grid (pyvista.StructuredGrid): A PyVista StructuredGrid with displacement data.
    """

    # Ensure the input arrays are flattened for VTK compatibility
    points = np.vstack((xMesh.flatten(), yMesh.flatten(), zMesh.flatten())).T

    # Create a PyVista StructuredGrid object
    dims = xMesh.shape  # The grid dimensions
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (dims[1], dims[0], 1)  # PyVista expects (nx, ny, nz)

    # Add displacement data as point data
    grid.point_data["u_disp"] = u_disp.flatten()
    grid.point_data["v_disp"] = v_disp.flatten()
    grid.point_data["w_disp"] = w_disp.flatten()

    # Add the magnitude of the displacement vector
    displacement_magnitude = np.sqrt(u_disp**2 + v_disp**2 + w_disp**2).flatten()
    grid.point_data["displacement_magnitude"] = displacement_magnitude

    return grid

def save_vtk_file_diff(vtk_grid, file_path):
    """
    Saves a PyVista StructuredGrid to a VTK file.

    Parameters:
    - vtk_grid (pyvista.StructuredGrid): The grid to save.
    - file_path (str): Path to save the VTK file.
    """
    try:
        vtk_grid.save(file_path)
        print(f"Saved VTK file to: {file_path}")
    except Exception as e:
        print(f"Error saving VTK file {file_path}: {e}")

def radial2vtk(
    mesh_path, start_path, end_path, save_path, key, radius=residuals.radius
):
    """calculate radial displacement and transform to vtk surface"""
    residuals_d = residuals.from_pickle(
        mesh_path=mesh_path,
        start_path=start_path,
        end_path=end_path,
        key=key,
    )

    xMesh_N, yMesh_N, zMesh_N = residuals_d.calc_coordinates()
    xMesh = xMesh_N * radius * 2
    yMesh = yMesh_N * radius * 2
    zMesh = zMesh_N * radius * 2

    # calculate displacement data
    u_rsdl, v_rsdl, w_rsdl = residuals_d.calc_residuals()

    # remesh the displacement data
    new_mesh, new_disp = reMesh_non_regular(
        xMesh=xMesh,
        yMesh=yMesh,
        zMesh=zMesh,
        uDisp=u_rsdl,
        vDisp=v_rsdl,
        wDisp=w_rsdl,
        time_step=2,
    )

    new_xMesh = new_mesh["xMesh"]
    new_yMesh = new_mesh["yMesh"]
    new_zMesh = new_mesh["zMesh"]
    new_u = new_disp["u_disp"]
    new_v = new_disp["v_disp"]
    new_w = new_disp["w_disp"]

    # calculate the radial displacement
    pol = cart2pol(new_xMesh, new_zMesh)
    drdt = cal_drdt(new_u, new_w, pol[1])
    dr = drdt[0]

    # Create and save VTK grid
    surface_to_vtk(new_xMesh, new_yMesh, new_zMesh, dr, save_path)


def interp_zMesh(xMesh, yMesh, zMesh, x, y):
    interpolator = Rbf(
        xMesh.flatten(),
        yMesh.flatten(),
        zMesh.flatten(),
        function="linear",
    )

    z = interpolator(x, y)

    return z


def interp_zMesh_nearest(xMesh, yMesh, zMesh, x, y):
    # Points where the data is defined
    points = np.column_stack((xMesh.flatten(), yMesh.flatten()))

    # Values at the points
    values = zMesh.flatten()

    # Interpolator
    interpolator = NearestNDInterpolator(points, values)

    # Perform extrapolation
    z = interpolator(x, y)
    return z


def main():
    radius = residuals.radius

    mesh_path = "./data/mesh_hiRes.txt"
    start_path = "./data/fine_disp_data/strain_5.0_extrap_hiRes.txt"
    end_path = "./data/fine_disp_data/strain_6.0_extrap_hiRes.txt"

    key = "Z40C_100103a-192"
    # save_path = "./data/stain_5_7_092903b_radial.vtk"
    save_path = "./data/disp_vtk/stain_5.0_6.0_100103a_non_regular_clip.vtk"

    # radial2vtk(
    #     mesh_path=mesh_path,
    #     start_path=start_path,
    #     end_path=end_path,
    #     save_path=save_path,
    #     key=key,
    # )

    disp2vtk(
        mesh_path=mesh_path,
        start_path=start_path,
        end_path=end_path,
        save_path=save_path,
        key=key,
        x_min=-26,
        x_max=20,
        y_min=50,
        y_max=125,
        radial_differential=True
    )


if __name__ == "__main__":
    main()
