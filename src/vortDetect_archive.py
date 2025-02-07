import numpy as np
import scipy
from scipy import ndimage
import sys
import scipy.optimize as opt
from scipy.interpolate import Rbf
import pickle
import vtk
from vtk.util import numpy_support


def calc_swirling(kinematics):
    """
    2D Swirling strength

    :param kinematics: contains meshes, vectors, and kinematics
    :type kinematics: class Kinematics()
    :returns: swirling strength criterion
    :rtype: ndarray
    """
    print("Detection method: swirling strength")

    matrix_a = np.array(
        [
            [kinematics.du_x.ravel(), kinematics.du_y.ravel()],
            [kinematics.dv_x.ravel(), kinematics.dv_y.ravel()],
        ]
    )

    matrix_a = matrix_a.transpose((2, 1, 0))
    eigenvalues = np.linalg.eigvals(matrix_a)
    swirling = np.max(eigenvalues.imag, axis=1).reshape(
        kinematics.xMesh[:, 0].size, kinematics.yMesh[0, :].size
    )
    print("Max value of swirling: ", np.round(np.max(swirling), 2))
    return swirling


def compute_swirling(kinematics):
    """
    Compute the swirling strength from the velocity gradient components.

    Parameters:
    - kinematics.du_x: Partial derivative of u with respect to x.
    - kinematics.du_y: Partial derivative of u with respect to y.
    - kinematics.dv_x: Partial derivative of v with respect to x.
    - kinematics.dv_y: Partial derivative of v with respect to y.

    Returns:
    - swirling_strength: A 2D array of swirling strength at each point in the field.
    """
    num_rows, num_cols = (
        kinematics.du_x.shape
    )  # Assuming all inputs have the same shape

    swirling_strength = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            # Construct the velocity gradient tensor for each point
            grad_u = np.array(
                [
                    [kinematics.grad[0][i, j], kinematics.grad[1][i, j]],
                    [kinematics.grad[2][i, j], kinematics.grad[3][i, j]],
                ]
            )

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(grad_u)

            # The swirling strength is the magnitude of the imaginary part of the eigenvalues
            swirling_strength[i, j] = np.abs(np.imag(eigenvalues)).max()

    return swirling_strength


def compute_rortex(kinematics):
    """
    Compute the Rortex for identifying vortex structure centers.

    Parameters:
    - kinematics: An object containing the partial derivatives of the velocity components.

    Returns:
    - rortex: A 2D array of Rortex at each point in the field.
    """
    num_rows, num_cols = (
        kinematics.du_x.shape
    )  # Assuming all inputs have the same shape

    rortex = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            # Construct the antisymmetric part of the velocity gradient tensor
            omega = (
                np.array(
                    [
                        [0, kinematics.grad[1][i, j] - kinematics.grad[2][i, j]],
                        [kinematics.grad[2][i, j] - kinematics.grad[1][i, j], 0],
                    ]
                )
                / 2
            )

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(omega)

            # The Rortex is the magnitude of the imaginary part of the eigenvalues
            rortex[i, j] = np.abs(np.imag(eigenvalues)).max()

    return rortex


def find_peaks(data, threshold, box_size):
    """
    Find local peaks in an image that are above a specified
    threshold value.

    Peaks are the maxima above the "threshold" within a local region.
    The regions are defined by the "box_size" parameters.
    "box_size" defines the local region around each pixel
    as a square box.

    :param data: The 2D array of the image/data.
    :param threshold: The data value or pixel-wise data values to be used for the
        detection threshold.  A 2D "threshold" must have the same
        shape as "data".
    :param box_size: The size of the local region to search for peaks at every point
    :type data: ndarray
    :type threshold: float
    :type box_size: int

    :returns: An array containing the x and y pixel location of the peaks and their values.
    :rtype: list
    """

    if np.all(data == data.flat[0]):
        return []

    data_max = ndimage.maximum_filter(data, size=box_size, mode="constant", cval=0.0)

    # get absolute values of data
    data = abs(data)

    peak_goodmask = data == data_max  # good pixels are True

    peak_goodmask = np.logical_and(peak_goodmask, (data > threshold))
    y_peaks, x_peaks = peak_goodmask.nonzero()
    peak_values = data[y_peaks, x_peaks]
    peaks = (y_peaks, x_peaks, peak_values)
    return peaks


def direction_rotation(vorticity, peaks):
    """
    Identify the direction of the vortices rotation using the vorticity.

    :param vorticity: 2D array with the computed vorticity
    :param peaks: list of the detected peaks
    :type vorticity: ndarray
    :type peaks: list

    :returns: vortices_clockwise, vortices_counterclockwise, arrays containing the direction of rotation for each vortex
    :rtype: list
    """

    vortices_clockwise_x, vortices_clockwise_y, vortices_clockwise_cpt = [], [], []
    (
        vortices_counterclockwise_x,
        vortices_counterclockwise_y,
        vortices_counterclockwise_cpt,
    ) = ([], [], [])
    for i in range(len(peaks[0])):
        if vorticity[peaks[0][i], peaks[1][i]] > 0.0:
            vortices_clockwise_x.append(peaks[0][i])
            vortices_clockwise_y.append(peaks[1][i])
            vortices_clockwise_cpt.append(peaks[2][i])
        else:
            vortices_counterclockwise_x.append(peaks[0][i])
            vortices_counterclockwise_y.append(peaks[1][i])
            vortices_counterclockwise_cpt.append(peaks[2][i])
    vortices_clockwise = (
        vortices_clockwise_x,
        vortices_clockwise_y,
        vortices_clockwise_cpt,
    )
    vortices_counterclockwise = (
        vortices_counterclockwise_x,
        vortices_counterclockwise_y,
        vortices_counterclockwise_cpt,
    )
    vortices_clockwise = np.asarray(vortices_clockwise)
    vortices_counterclockwise = np.asarray(vortices_counterclockwise)
    return vortices_clockwise, vortices_counterclockwise


def velocity_model(core_radius, gamma, x_real, y_real, u_advection, v_advection, x, y):
    """Generates the Lamb-Oseen vortex velocity array

    :param core_radius: core radius of the vortex
    :param gamma: circulation contained in the vortex
    :param x_real: relative x position of the vortex center
    :param y_real: relative y position of the vortex center
    :param u_advection: u advective velocity at the center
    :param v_advection: v advective velocity at the center
    :param x:
    :param y:
    :type core_radius: float
    :type gamma: float
    :type x_real: float
    :type y_real: float
    :type u_advection: float
    :type v_advection: float
    :type x: float
    :type y: float
    :returns: velx, vely
    :rtype: float
    """
    r = np.hypot(x - x_real, y - y_real)
    vel = (gamma / (2 * np.pi * r)) * (1 - np.exp(-(r**2) / core_radius**2))
    vel = np.nan_to_num(vel)
    velx = u_advection - vel * (y - y_real) / r
    vely = v_advection + vel * (x - x_real) / r
    velx = np.nan_to_num(velx)
    vely = np.nan_to_num(vely)
    # print(core_radius, gamma, x_real, y_real, u_advection, v_advection, x, y)
    return velx, vely


def window(kinematics, x_center_index, y_center_index, dist):
    """
    Defines a window around (x; y) coordinates

    :param kinematics: full size velocity field
    :type kinematics: ndarray
    :param x_center_index: box center index (x)
    :type x_center_index: int
    :param y_center_index: box center index (y)
    :type y_center_index: int
    :param dist: size of the vortex (mesh units)
    :param dist: int

    :returns: cropped arrays for x, y, u and v
    :rtype: 2D arrays of floats

    """
    if x_center_index - dist > 0:
        x1 = x_center_index - dist
    else:
        x1 = 0
    if y_center_index - dist > 0:
        y1 = y_center_index - dist
    else:
        y1 = 0
    if x_center_index + dist <= kinematics.u_disp.shape[1]:
        x2 = x_center_index + dist
    else:
        x2 = kinematics.u_disp.shape[1]
    if y_center_index + dist <= kinematics.v_disp.shape[0]:
        y2 = y_center_index + dist
    else:
        y2 = kinematics.v_disp.shape[0]

    # x_index, y_index = np.meshgrid(
    #     kinematics.xMesh[int(x1) : int(x2)],
    #     kinematics.yMesh[int(y1) : int(y2)],
    #     indexing="xy",
    # )
    x_index = kinematics.xMesh[int(y1) : int(y2), int(x1) : int(x2)]
    y_index = kinematics.yMesh[int(y1) : int(y2), int(x1) : int(x2)]
    u_data = kinematics.u_disp[int(y1) : int(y2), int(x1) : int(x2)]
    v_data = kinematics.v_disp[int(y1) : int(y2), int(x1) : int(x2)]
    return x_index, y_index, u_data, v_data


def get_vortices(kinematics, peaks, vorticity, rmax=0.5, correlation_threshold=0.25):
    """
    General routine to check if the detected vortex is a real vortex

    :param vfield: data from the input file
    :param peaks: list of vortices
    :param vorticity: calculated field
    :param rmax: maximum radius (adapt it to your data domain)
    :param correlation_threshold: threshold to detect a vortex (default is 0.75)
    :type vfield: class VelocityField
    :type peaks: list
    :type vorticity: ndarray
    :type rmax: float
    :type correlation_threshold: float
    :returns: list of detected vortices
    :rtype: list
    """

    vortices = list()
    cpt_accepted = 0

    dx = kinematics.x_avg
    dy = kinematics.y_avg

    for i in range(len(peaks[0])):
        x_center_index = peaks[1][i]
        y_center_index = peaks[0][i]
        print(
            i, "Processing detected swirling at (x, y)", x_center_index, y_center_index
        )
        if rmax == 0.0:
            core_radius = 2 * np.hypot(dx, dy)
        else:
            core_radius = rmax  # guess on the starting vortex radius
        gamma = vorticity[y_center_index, x_center_index] * np.pi * core_radius**2

        vortices_parameters = full_fit(
            core_radius, gamma, kinematics, x_center_index, y_center_index
        )
        if vortices_parameters[6] < 2:
            correlation_value = 0
        else:
            x_index, y_index, u_data, v_data = window(
                kinematics,
                x_center_index,
                y_center_index,
                vortices_parameters[6],
            )
            u_model, v_model = velocity_model(
                vortices_parameters[0],
                vortices_parameters[1],
                vortices_parameters[2],
                vortices_parameters[3],
                vortices_parameters[4],
                vortices_parameters[5],
                x_index,
                y_index,
            )
            correlation_value = correlation_coef(
                u_data - vortices_parameters[4],
                v_data - vortices_parameters[5],
                u_model - vortices_parameters[4],
                v_model - vortices_parameters[5],
            )
        if correlation_value > correlation_threshold:
            print(
                "Accepted! Correlation = {:1.2f} (vortex #{:2d})".format(
                    correlation_value, cpt_accepted
                )
            )
            u_theta = (
                vortices_parameters[1] / (2 * np.pi * vortices_parameters[0])
            ) * (
                1 - np.exp(-1)
            )  # compute the tangential velocity at critical radius
            vortices.append(
                [
                    vortices_parameters[0],
                    vortices_parameters[1],
                    vortices_parameters[2],
                    vortices_parameters[3],
                    vortices_parameters[4],
                    vortices_parameters[5],
                    vortices_parameters[6],
                    correlation_value,
                    u_theta,
                ]
            )
            cpt_accepted += 1
    return vortices


def correlation_coef(u_data, v_data, u_model, v_model):
    """Calculates the correlation coefficient between two 2D arrays

    :param u_data: velocity u from the data at the proposed window
    :param v_data: velocity v from the data at the proposed window
    :param u_model: velocity u from the calculated model
    :param v_model: velocity v from the calculated model
    :type u_data: ndarray
    :type v_data: ndarray
    :type u_model: ndarray
    :type v_model: ndarray
    :returns: correlation
    :rtype: float
    """
    u_data = u_data.ravel()
    v_data = v_data.ravel()
    u = u_model.ravel()
    v = v_model.ravel()

    prod_piv_mod = np.mean(u_data * u + v_data * v)
    prod_piv = np.mean(u * u + v * v)
    prod_mod = np.mean(u_data * u_data + v_data * v_data)
    correlation = prod_piv_mod / (max(prod_piv, prod_mod))

    return correlation


def full_fit(core_radius, gamma, kinematics, x_center_index, y_center_index):
    """Full fitting procedure

    :param core_radius: core radius of the vortex
    :param gamma: circulation contained in the vortex
    :param kinematics: data from the input file
    :param x_center_index: x index of the vortex center
    :param y_center_index: y index of the vortex center
    :type core_radius: float
    :type gamma: float
    :type vfield: class
    :type x_center_index: int
    :type y_center_index: int
    :returns: fitted[i], dist
    :rtype: list
    """

    fitted = [[], [], [], [], [], []]
    fitted[0] = core_radius
    fitted[1] = gamma
    fitted[2] = kinematics.xMesh[0, x_center_index]
    fitted[3] = kinematics.yMesh[y_center_index, 0]
    dx = kinematics.x_avg
    dy = kinematics.y_avg
    dist = 0
    # correlation_value = 0.0
    for i in range(10):
        # WARN: do not know what's these meant for?
        # x_center_index = int(round((fitted[2] - kinematics.xMesh[0, 0]) / dx))
        # y_center_index = int(round((fitted[3] - kinematics.yMesh[0, 0]) / dy))
        if x_center_index >= kinematics.u_disp.shape[1]:
            x_center_index = kinematics.u_disp.shape[1] - 1
        if x_center_index <= 2:
            x_center_index = 3
        if y_center_index >= kinematics.v_disp.shape[0]:
            y_center_index = kinematics.v_disp.shape[0] - 1
        # Note: why do not fix y_center lower end?
        r1 = fitted[0]
        x1 = fitted[2]
        y1 = fitted[3]
        dist = int(round(fitted[0] / np.hypot(dx, dy), 0)) + 1
        if fitted[0] < 2 * np.hypot(dx, dy):
            break
        fitted[4] = kinematics.u_disp[y_center_index, x_center_index]  # u_advection
        fitted[5] = kinematics.v_disp[y_center_index, x_center_index]  # v_advection
        x_index, y_index, u_data, v_data = window(
            kinematics, x_center_index, y_center_index, dist
        )

        fitted = fit(
            fitted[0],
            fitted[1],
            x_index,
            y_index,
            fitted[2],
            fitted[3],
            u_data,
            v_data,
            fitted[4],
            fitted[5],
            i,
        )
        if i > 0:
            # break if radius variation is less than 10% and accepts
            if abs(fitted[0] / r1 - 1) < 0.1:
                if (abs((fitted[2] / x1 - 1)) < 0.1) or (
                    abs((fitted[3] / y1 - 1)) < 0.1
                ):
                    break
            # break if x or y position is out of the window and discards
            if (abs((fitted[2] - x1)) > dist * dx) or (
                abs((fitted[3] - y1)) > dist * dy
            ):
                dist = 0
                break
    return fitted[0], fitted[1], fitted[2], fitted[3], fitted[4], fitted[5], dist


def fit(
    core_radius,
    gamma,
    x,
    y,
    x_real,
    y_real,
    u_data,
    v_data,
    u_advection,
    v_advection,
    i,
):
    """
    Fitting  of the Lamb-Oseen Vortex

    :param core_radius: core radius of the vortex
    :param gamma: circulation contained in the vortex
    :param x: x position
    :param y: y position
    :param x_real: x position of the vortex center
    :param y_real: y position of the vortex center
    :param u_data: velocity u from the data at the proposed window
    :param v_data: velocity v from the data at the proposed window
    :param u_advection: uniform advection velocity u
    :param v_advection: uniform advection velocity u
    :param i: current iteration for fitting
    :type core_radius: float
    :type gamma: float
    :type x: ndarray
    :type y: ndarray
    :type x_real: float
    :type y_real: float
    :type u_data: ndarray
    :type v_data: ndarray
    :type u_advection: float
    :type v_advection: float
    :type i: iterator
    :returns: fitted parameters (core_radius, gamma, xcenter, ycenter, u_advection, v_advection...)
    :rtype: list
    """
    # Method for opt.least_squares fitting. Can be
    # 'trf': Trust Region Reflective algorithm
    # 'dogbox': dogleg algorithm
    # 'lm': Levenberg-Marquardt algorithm
    method = "trf"

    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    x = x.ravel()
    y = y.ravel()
    u_data = u_data.ravel()
    v_data = v_data.ravel()
    # dx = x[0] - x[1]
    # dy = y[1] - y[0]

    def lamb_oseen_model(fitted):
        """
        Lamb-Oseen velocity model used for the nonlinear fitting

        :param fitted: parameters of a vortex (core_radius, gamma,xcenter,ycenter, u_advection, v_advection)
        :type fitted: list
        :returns: velocity field, following a Lamb-Oseen model
        :rtype: ndarray
        """

        core_radius_model = fitted[0]
        gamma_model = fitted[1]
        xcenter_model = fitted[2]
        ycenter_model = fitted[3]
        u_advection_model = fitted[4]
        v_advection_model = fitted[5]
        r = np.hypot(x - xcenter_model, y - ycenter_model)
        u_theta_model = (
            gamma_model / (2 * np.pi * r) * (1 - np.exp(-(r**2) / core_radius_model**2))
        )
        u_theta_model = np.nan_to_num(u_theta_model)
        u_model = u_advection_model - u_theta_model * (y - ycenter_model) / r - u_data
        v_model = v_advection_model + u_theta_model * (x - xcenter_model) / r - v_data
        u_model = np.nan_to_num(u_model)
        v_model = np.nan_to_num(v_model)
        lamb_oseen_model = np.append(u_model, v_model)
        return lamb_oseen_model

    if i > 0:
        m = 4
    # NOTE: the first step is large.
    else:
        m = 4

    if method == "trf":
        epsilon = 0.001
        bnds = (
            [
                0,
                gamma - abs(gamma) * m / 2 - epsilon,
                x_real - m * dx - epsilon,
                y_real - m * dy - epsilon,
                u_advection - abs(u_advection) - epsilon,
                v_advection - abs(v_advection) - epsilon,
            ],
            [
                core_radius + core_radius * m,
                gamma + abs(gamma) * m / 2 + epsilon,
                x_real + m * dx + epsilon,
                y_real + m * dy + epsilon,
                u_advection + abs(u_advection) + epsilon,
                v_advection + abs(v_advection) + epsilon,
            ],
        )

        sol = opt.least_squares(
            lamb_oseen_model,
            [core_radius, gamma, x_real, y_real, u_advection, v_advection],
            method="trf",
            bounds=bnds,
        )
    elif method == "dogbox":
        epsilon = 0.001
        bnds = (
            [
                0,
                gamma - abs(gamma) * m / 2 - epsilon,
                x_real - m * dx - epsilon,
                y_real - m * dy - epsilon,
                u_advection - abs(u_advection) - epsilon,
                v_advection - abs(v_advection) - epsilon,
            ],
            [
                core_radius + core_radius * m,
                gamma + abs(gamma) * m / 2 + epsilon,
                x_real + m * dx + epsilon,
                y_real + m * dy + epsilon,
                u_advection + abs(u_advection) + epsilon,
                v_advection + abs(v_advection) + epsilon,
            ],
        )

        sol = opt.least_squares(
            lamb_oseen_model,
            [core_radius, gamma, x_real, y_real, u_advection, v_advection],
            method="dogbox",
            bounds=bnds,
        )
    elif method == "lm":
        sol = opt.least_squares(
            lamb_oseen_model,
            [core_radius, gamma, x_real, y_real, u_advection, v_advection],
            method="lm",
            xtol=10 * np.hypot(dx, dy),
        )

    return sol.x


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


def interp_zMesh(kinematics, x, y):
    interpolator = Rbf(
        kinematics.xMesh.flatten(),
        kinematics.yMesh.flatten(),
        kinematics.zMesh.flatten(),
        function="linear",
    )

    z = interpolator(x, y)

    return z


def reMesh(mesh_path, disp_path, load_step, time_step, mesh_save_path, disp_save_path):
    # Load mesh data
    mesh_whole = pickle.load(open(mesh_path, "rb"))
    x_mesh = mesh_whole["xMesh"][30:-30, 5:-5]
    y_mesh = mesh_whole["yMesh"][30:-30, 5:-5]
    z_mesh = mesh_whole["zMesh"][30:-30, 5:-5]

    # Load displacement data
    disp = pickle.load(open(disp_path, "rb"))
    u_disp = disp[load_step][0][30:-30, 5:-5]
    v_disp = disp[load_step][1][30:-30, 5:-5]
    w_disp = disp[load_step][2][30:-30, 5:-5]

    # Define the uniform x grid based on the original x range
    x_uniform = np.linspace(x_mesh.min().min(), x_mesh.max().max(), x_mesh.shape[1])

    # Create new meshgrid for the uniform x direction and existing y
    x_new, y_new = np.meshgrid(x_uniform, y_mesh[:, 0])

    # Radius of the cylinder
    radius = np.sqrt(x_mesh**2 + z_mesh**2).max().max()

    # Calculate the new z mesh based on the uniform x and radius
    z_new = np.sqrt(radius**2 - x_new**2)

    # NOTE: too slow
    # # Interpolate displacements onto the new mesh
    # interpolator_u = OrdinaryKriging3D(
    #     x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel(), u_disp.ravel(), variogram_model="linear"
    # )

    # interpolator_v = OrdinaryKriging3D(
    #     x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel(), v_disp.ravel(), variogram_model="linear"
    # )

    # interpolator_w = OrdinaryKriging3D(
    #     x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel(), w_disp.ravel(), variogram_model="linear"
    # )

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

    with open(mesh_save_path, "wb") as fp:  # Pickling
        pickle.dump(new_mesh, fp)

    with open(disp_save_path, "wb") as fp:  # Pickling
        pickle.dump(new_disp, fp)


def main():
    # from differentiate import Kinematics
    from differentiate_cylindrical import Kinematics

    mesh_path = "./data/mesh_hiRes.txt"
    disp_path = "./data/strain_8.0_residualDisp_all.txt"
    load_step = "Z40C_092903b-192"

    # re-mesh path
    mesh_save_path = "./data/mesh_strain_8.0_092903b-192.txt"
    disp_save_path = "./data/disp_strain_8.0_092903b-192.txt"

    # reMesh(
    #     mesh_path=mesh_path,
    #     disp_path=disp_path,
    #     load_step=load_step,
    #     time_step=2,
    #     mesh_save_path=mesh_save_path,
    #     disp_save_path=disp_save_path,
    # )

    kine = Kinematics(
        mesh_path=mesh_save_path,
        disp_path=disp_save_path,
        load_step=load_step,
        time_step=None,
        normalization_flag=False,
        normalization_direction=None,
        average_flag=False,
        average_direction=None,
    )

    # detection_field = compute_rortex(kine)
    detection_field = kine.curl_r
    peaks = find_peaks(abs(detection_field), threshold=3, box_size=5)
    vorticity = kine.curl_r
    # # vortices_counterclockwise, vortices_clockwise = direction_rotation(vorticity, peaks)

    # # surface in paraview
    # surface_to_vtk(
    #     kine.xMesh, kine.yMesh, kine.zMesh, vorticity, filename="./data/vorticity.vtk"
    # )

    # # visualize vortex structure in paraview
    # x_coord = kine.xMesh[peaks[0], peaks[1]].ravel()
    # y_coord = kine.yMesh[peaks[0], peaks[1]].ravel()
    # z_coord = kine.zMesh[peaks[0], peaks[1]].ravel()

    # Fitting vortices to Lamb-Oseen Vortex Model
    vorticies = get_vortices(
        kinematics=kine,
        peaks=peaks,
        vorticity=vorticity,
        rmax=0.4,
        correlation_threshold=0.85,
    )

    # calculate x, y, z coordinates
    x_coord = np.array([sublist[2] for sublist in vorticies])
    y_coord = np.array([sublist[3] for sublist in vorticies])
    z_coord = interp_zMesh(kinematics=kine, x=x_coord, y=y_coord)

    write_vtk_points(
        x_coord, y_coord, z_coord, filename="./data/vortices_reMesh_fit.vtk"
    )

    # # verify detected vorticities in matplotlib
    # from util import plt_grad
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(1.5, 2))
    # ax = fig.add_subplot(111)

    # ax = plt_grad(kinematics=kine, fig=fig, ax=ax)
    # vort_x = kine.xMesh[peaks[0],peaks[1]]
    # vort_y = kine.yMesh[peaks[0],peaks[1]]
    # ax.scatter(vort_x, vort_y, s=3, c='k', zorder=2)
    # plt.tight_layout()
    # plt.savefig("./pics/" + "vort_verify.tiff", dpi=600)

    pass


if __name__ == "__main__":
    main()
