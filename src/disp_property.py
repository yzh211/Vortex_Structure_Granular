import numpy as np
import matplotlib.pyplot as plt
from residuals import residuals
import pickle
import os
import itertools
import more_itertools
from remesh import reMesh_non_regular
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import seaborn as sns
from statannotations.Annotator import Annotator

plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["figure.titlesize"] = 8
plt.rcParams["figure.figsize"] = (3.5, 2)


class Time_series:
    def __init__(self, data_folder, start_frame, end_frame, num_tests=12):
        self.data_folder = data_folder
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.num_tests = num_tests

    def gen_time_series(self):
        # Extract the numeric parts from start and end strains
        start_value = float(self.start_frame.split("_")[1])
        end_value = float(self.end_frame.split("_")[1])
        # Initialize list to hold valid files
        included_files = []

        # Iterate through files in the folder
        for file_name in os.listdir(self.data_folder):
            if file_name.startswith("strain_") and file_name.endswith("_Disp.txt"):
                try:
                    # Extract the first strain value from the filename
                    strain_part = file_name.split("_")[1]
                    strain_value = float(strain_part)

                    # Check if the strain value is within the specified range
                    if start_value <= strain_value <= end_value:
                        included_files.append(file_name)
                except ValueError:
                    # Skip files that don't match the expected pattern
                    continue

        time_series = []
        for file in included_files:
            disp = pickle.load(open(self.data_folder + file, "rb"))
            u_disp = disp["u_disp"]
            v_disp = disp["v_disp"]
            w_disp = disp["w_disp"]
            norm_disp = np.linalg.norm([u_disp, v_disp, w_disp], axis=0)
            time_series.append(norm_disp)

        num_files = len(time_series)
        num_stages = num_files // self.num_tests

        if num_files % self.num_tests != 0:
            raise ValueError("The list length must be divisible by the group size.")

        time_series_nested = [
            [time_series[i + j * self.num_tests] for j in range(num_stages)]
            for i in range(self.num_tests)
        ]

        return time_series_nested

    def gen_disp_vector(self):
        # Extract the numeric parts from start and end strains
        start_value = float(self.start_frame.split("_")[1])
        end_value = float(self.end_frame.split("_")[1])
        # Initialize list to hold valid files
        included_files = []

        # Iterate through files in the folder
        for file_name in os.listdir(self.data_folder):
            if file_name.startswith("strain_") and file_name.endswith(
                "_regularDisp.txt"
            ):
                try:
                    # Extract the first strain value from the filename
                    strain_part = file_name.split("_")[1]
                    strain_value = float(strain_part)

                    # Check if the strain value is within the specified range
                    if start_value <= strain_value <= end_value:
                        included_files.append(file_name)
                except ValueError:
                    # Skip files that don't match the expected pattern
                    continue

        time_series = []
        for file in included_files:
            disp = pickle.load(open(self.data_folder + file, "rb"))
            u_disp = disp["u_disp"].flatten()
            v_disp = disp["v_disp"].flatten()
            w_disp = disp["w_disp"].flatten()
            disp_vector = np.stack((u_disp, v_disp, w_disp), axis=-1)
            time_series.append(disp_vector)

        num_files = len(time_series)
        num_stages = num_files // self.num_tests

        if num_files % self.num_tests != 0:
            raise ValueError("The list length must be divisible by the group size.")

        time_series_nested = [
            [time_series[i + j * self.num_tests] for j in range(num_stages)]
            for i in range(self.num_tests)
        ]

        return time_series_nested

    def avg_disp(self):
        """Calculate the averaged fields for each location across a series of 2D arrays."""
        time_series = self.gen_time_series()
        n_stages = len(time_series[0])

        # Prepare an output list for each group of 5 stages
        output_list = []

        # Process in chunks of 5 stages
        for group_start in range(0, n_stages, 5):
            group_end = group_start + 5

            # Collect arrays for these stages across all tests
            group_data = [test[group_start:group_end] for test in time_series]

            # Convert to numpy array for easier manipulation
            # Resulting shape (n_tests, 5, 126, 63)
            group_data_array = np.array(group_data)

            # Average across tests and stages within the group
            # First average across stages within each test, then across tests
            mean_across_stages = np.mean(
                group_data_array, axis=1
            )  # Shape (n_tests, 126, 63)
            mean_across_tests = np.mean(mean_across_stages, axis=0)  # Shape (126, 63)

            # Append the mean field of this group to the output list
            output_list.append(mean_across_tests.flatten())

        return output_list

    def acf(self, max_lag=None):
        """Calculate the Autocorrelation Function (ACF) for each location across a series of 2D arrays."""
        time_series = self.gen_time_series()
        acf_results_allTests = []

        for time_series_data in time_series:
            n_times, n_rows, n_cols = (
                len(time_series_data),
                time_series_data[0].shape[0],
                time_series_data[0].shape[1],
            )

            batch = []
            # NOTE: Process in chunks of 1% axial strain
            for start in range(0, n_times, 5):
                end = start + 5
                current_batch = time_series_data[start:end]

                # Reorganize data: collect all time points for each spatial location
                reorganized_data = np.transpose(
                    np.array(current_batch), (1, 2, 0)
                )  # Now (n_rows, n_cols, n_times)

                # Prepare to collect ACF
                acf_results = np.zeros((n_rows, n_cols, max_lag + 1))

                for i in range(n_rows):
                    for j in range(n_cols):
                        time_series = reorganized_data[i, j, :]

                        # Compute the ACF for the time series at location (i, j)
                        for lag in range(max_lag + 1):
                            if lag == 0:
                                acf_results[i, j, lag] = 1  # ACF is always 1 at lag 0
                            else:
                                series_past = time_series[:-lag]
                                series_future = time_series[lag:]
                                correlation = np.corrcoef(series_past, series_future)[
                                    0, 1
                                ]
                                acf_results[i, j, lag] = correlation

                batch.append(acf_results)
            acf_results_allTests.append(batch)

        # Assuming a structure of output collection is required similar to Hurst processing:
        output_list = self.process_acf_output(acf_results_allTests)
        return output_list

    def process_acf_output(self, acf_data):
        n_tests = len(acf_data)
        n_stages = len(acf_data[0])
        array_shape = acf_data[0][
            0
        ].shape  # Assumes all arrays have the same shape (n_rows, n_cols, max_lag+1)

        # Prepare an output list for each stage
        output_list = [[] for _ in range(n_stages)]

        # Loop through each stage
        for stage_index in range(n_stages):
            # Collect the arrays for this stage across all tests
            stage_data = [
                acf_data[test_index][stage_index] for test_index in range(n_tests)
            ]

            # Convert list of arrays to a 4D NumPy array for easier manipulation
            stage_data_array = np.array(
                stage_data
            )  # Shape (n_tests, n_rows, n_cols, max_lag+1)

            # Calculate the mean across the first dimension (tests)
            mean_stage_array = np.mean(
                stage_data_array, axis=0
            )  # Shape (n_rows, n_cols, max_lag+1)

            # Append the mean field to the corresponding stage in the output list
            output_list[stage_index].append(mean_stage_array)

        return output_list

    def hurst_exponent(self):
        """Calculate the Hurst Exponent for each location across a series of 2D arrays."""
        # Determine the shape of each 2D array (assuming all arrays have the same shape)
        time_series = self.gen_time_series()
        hurst_exponents_allTests = []
        for time_series_data in time_series:
            n_times, n_rows, n_cols = (
                len(time_series_data),
                time_series_data[0].shape[0],
                time_series_data[0].shape[1],
            )

            batch = []
            # NOTE: Process in chunks of 1% axial strain
            for start in range(0, n_times, 5):
                end = start + 5
                current_batch = time_series_data[start:end]

                # Reorganize data: collect all time points for each spatial location
                reorganized_data = np.array(
                    current_batch
                )  # This will be shape (n_times, n_rows, n_cols)
                reorganized_data = np.transpose(
                    current_batch, (1, 2, 0)
                )  # Now (n_rows, n_cols, n_times)

                # Prepare to collect Hurst exponents
                hurst_exponents = np.zeros((n_rows, n_cols))

                for i in range(n_rows):
                    for j in range(n_cols):
                        series = reorganized_data[i, j, :]

                        # Compute the Hurst Exponent for the time series at location (i, j)
                        N = len(series)
                        T = np.arange(1, N + 1)
                        Y = np.cumsum(series - np.mean(series))
                        Z = Y - np.mean(Y)

                        R = np.max(Z) - np.min(Z)
                        S = np.std(series)

                        hurst_exponents[i, j] = np.log(R / S) / np.log(N)

                batch.append(hurst_exponents)

            hurst_exponents_allTests.append(batch)

        n_tests = len(hurst_exponents_allTests)
        n_stages = len(hurst_exponents_allTests[0])
        array_shape = hurst_exponents_allTests[0][
            0
        ].shape  # Assumes all arrays are the same shape (126, 63)

        # Prepare an output list for each stage with an initialized list of arrays for each stage
        output_list = [[] for _ in range(n_stages)]

        # Loop through each stage
        for stage_index in range(n_stages):
            # Collect the arrays for this stage across all tests
            stage_data = [
                hurst_exponents_allTests[test_index][stage_index]
                for test_index in range(n_tests)
            ]

            # Convert list of arrays to a 3D NumPy array for easier manipulation
            stage_data_array = np.array(stage_data)  # Shape (n_tests, 126, 63)

            # Calculate the mean across the first dimension (tests) to get the mean field for this stage
            mean_stage_array = np.mean(stage_data_array, axis=0)  # Shape (126, 63)

            # Append the mean field to the corresponding stage in the output list
            output_list[stage_index].append(mean_stage_array.flatten())

        # hurst_exponents_stack = np.stack(hurst_exponents_allTests, axis=0)
        # hurst_exponents_avg = np.mean(hurst_exponents_stack, axis=0)

        # return hurst_exponents_avg
        return output_list

    def vertical_diffusion(self):
        pass

    def compute_gradient(self, displacements):
        # Assuming positions and displacements are regularly spaced and structured for simplicity

        # Initialize the gradient tensor
        gradient_tensor = np.zeros((displacements.shape[0], 3, 3))

        # Compute gradients for each component
        for i in range(3):  # Over x, y, z components of displacement
            for j in range(3):  # Over x, y, z directions
                gradient_tensor[:, i, j] = gaussian_filter(
                    displacements[:, i], sigma=1, order=j
                )

        return gradient_tensor

    def compute_second_order_gradients(self, displacements):
        """
        Compute the second-order spatial gradients of the displacement field using finite differences.

        Parameters:
        displacements (np.ndarray): Displacement vectors at each point (n_points, 3).

        Returns:
        np.ndarray: Second-order gradient tensor (n_points, 3, 3, 3).
        """
        from scipy.ndimage import gaussian_filter

        second_order_gradient_tensor = np.zeros((displacements.shape[0], 3, 3, 3))

        # Compute second-order gradients for each displacement component
        for i in range(3):  # Over x, y, z components of displacement
            for j in range(3):  # Over x, y, z first derivatives
                # First derivatives
                first_derivative = gaussian_filter(
                    displacements[:, i], sigma=1, order=j
                )
                for k in range(3):  # Over x, y, z second derivatives
                    # Second derivatives
                    second_derivative = gaussian_filter(
                        first_derivative, sigma=1, order=k
                    )
                    second_order_gradient_tensor[:, i, j, k] = second_derivative

        return second_order_gradient_tensor

    def calculate_non_affine_strain(self, gradient_tensor):
        """
        Calculate the non-affine strain tensor by subtracting the affine part.

        Parameters:
        gradient_tensor (np.ndarray): Gradient tensor (n_points, 3, 3).

        Returns:
        np.ndarray: Non-affine strain tensor (n_points, 3, 3).
        """
        # Compute affine strain tensor (simplified version, assuming small strains)
        affine_strain = 0.5 * (
            gradient_tensor + np.transpose(gradient_tensor, (0, 2, 1))
        )

        # # Subtract from the total gradient to get non-affine strain
        # non_affine_strain = gradient_tensor - affine_strain

        return affine_strain

    def compute_curvature(self, gradient_tensor):
        """
        Calculate a curvature measure from the second-order gradients of the displacement field.

        Parameters:
        second_order_gradients (np.ndarray): Second-order gradient tensor (n_points, 3, 3, 3).

        Returns:
        np.ndarray: Curvature measures (n_points).
        """
        rotation_tensor = 0.5 * (
            gradient_tensor - np.transpose(gradient_tensor, (0, 2, 1))
        )

        return rotation_tensor

    def non_affine_strain(self):
        time_series_nested = self.gen_disp_vector()

        # Number of loading stages
        num_stages = len(time_series_nested[0])

        strain_results = []
        for stage_idx in range(num_stages):
            # Collect the processed results
            stage_results = []

            for test_idx in range(len(time_series_nested)):
                vector_field = time_series_nested[test_idx][stage_idx]
                gradient_tensor = self.compute_gradient(displacements=vector_field)
                strain = self.calculate_non_affine_strain(
                    gradient_tensor=gradient_tensor
                )
                magnitude = np.linalg.norm(strain, axis=(1, 2))
                stage_results.append(magnitude)

            strain_results.append(stage_results)

        return strain_results

    def non_affine_curvature(self):
        time_series_nested = self.gen_disp_vector()

        # Number of loading stages
        num_stages = len(time_series_nested[0])

        strain_results = []
        for stage_idx in range(num_stages):
            # Collect the processed results
            stage_results = []

            for test_idx in range(len(time_series_nested)):
                vector_field = time_series_nested[test_idx][stage_idx]
                gradient_tensor = self.compute_gradient(displacements=vector_field)
                strain = self.compute_curvature(gradient_tensor=gradient_tensor)
                stage_results.append(strain)

            strain_results.append(stage_results)

        return strain_results

    def plot_contours(self, variable, save_path):
        plt.figure(figsize=(10, 8))
        c = plt.pcolormesh(variable, cmap="coolwarm", shading="auto", vmin=0, vmax=1)
        plt.colorbar(c)  # Show color scale
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.savefig(save_path, dpi=1000)

    def plot_contours_f(self, variable, save_path):
        """
        Plot a filled contour plot of the provided variable with customized colormap highlighting deviations from 0.5.
        Values close to 0.5 are shown in white, and deviations are highlighted.

        Args:
        variable (np.ndarray): A 2D array of data points to plot.
        save_path (str): File path where the plot image will be saved.
        """
        plt.figure(figsize=(10, 8))

        # Define levels and midpoint for contours
        levels = np.linspace(
            0.2, 0.8, num=11
        )  # Adjust range and number of levels as needed
        midpoint = 0.5

        # Create a diverging colormap that highlights deviations from the midpoint (0.5)
        colors = [
            "blue",
            "white",
            "red",
        ]  # Blue for low, white for middle, red for high
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        norm = TwoSlopeNorm(vmin=0.2, vcenter=midpoint, vmax=0.8)

        # Create the filled contour plot with the custom colormap
        contourf = plt.contourf(variable, levels=levels, cmap=cmap, norm=norm)
        plt.colorbar(contourf)  # Show color scale

        # Optional: add contour lines to emphasize levels
        contours = plt.contour(
            variable, levels=levels, colors="black", linewidths=0.5, norm=norm
        )
        plt.clabel(
            contours, inline=True, fontsize=8, fmt="%1.2f"
        )  # Label the contour lines

        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.title("Filled Contour Plot of Variable with Custom Colormap")

        # Save the plot to the given path with high resolution
        plt.savefig(save_path, dpi=1000)
        plt.close()  # Close the plot to free up memory

        # plt.figure(figsize=(10, 8))
        # c = plt.contourf(variable, cmap='coolwarm', shading='auto', vmin=0, vmax=1)
        # # Add contour lines to emphasize the levels
        # contours = plt.contour(variable, colors='black', linewidths=0.5)
        # plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')  # Label the contour lines
        # plt.colorbar(c)  # Show color scale
        # plt.xlabel('Column Index')
        # plt.ylabel('Row Index')
        # plt.savefig(save_path, dpi=1000)

    def plot_disp(self, disp, save_path):
        """Plots a boxplot with significance annotations using statannotations."""
        # Colors from grey to black
        colors = [plt.cm.Greys(i / 9) for i in range(8)]

        # Prepare data
        data = [np.array(sublist) for sublist in disp]
        all_data = [item for sublist in data for item in sublist]
        group_labels = [f"{i}-{i+1}" for i in range(1, 9)]
        # group_labels = [f"Interval {i + 1}" for i in range(len(data))]
        repeated_labels = [
            label for i, label in enumerate(group_labels) for _ in range(len(data[i]))
        ]

        # Create the DataFrame for Seaborn
        import pandas as pd

        df = pd.DataFrame(
            {
                "Strain Interval (%)": repeated_labels,
                "Magnitude of Displacements (mm)": all_data,
            }
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
            x="Strain Interval (%)",
            y="Magnitude of Displacements (mm)",
            data=df,
            palette=colors,
            width=0.5,
            showmeans=False,
            meanline=False,
            flierprops=flierprops,
            whis=2.0,  # Extend whiskers to 2.0 * IQR
            medianprops={"color": "black", "lw": 0.4},  # Reduced median line width
            boxprops={"linewidth": 0.4},  # Reduced box line width
            whiskerprops={"linewidth": 0.4},  # Reduced whisker line width
            capprops={"linewidth": 0.4},  # Reduced cap line width
            # meanprops={"color": "red", "ls": "--", "lw": 1},
        )

        # Add annotations
        annotator = Annotator(
            ax,
            comparisons,
            data=df,
            x="Strain Interval (%)",
            y="Magnitude of Displacements (mm)",
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            loc="inside",
            verbose=2,
            fontsize=7,  # Reduced font size for annotator
            line_width=0.5,  # Reduced line width for annotator
        )
        annotator.apply_and_annotate()

        # Customizations
        plt.xlabel("Strain Intervals (%)", fontsize=8)
        plt.ylabel("Displacements (mm)", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=1000)

    def plot_hurst(self, hurst, save_path):
        """Plots a boxplot with significance annotations using statannotations."""
        # Colors from grey to black
        colors = [plt.cm.Greys(i / 9) for i in range(8)]

        # Prepare data
        data = [np.array(sublist[0]) for sublist in hurst]
        all_data = list(more_itertools.collapse(data))
        group_labels = [f"{i}-{i+1}" for i in range(1, 9)]
        # group_labels = [f"Interval {i + 1}" for i in range(len(data))]
        repeated_labels = [
            label for i, label in enumerate(group_labels) for _ in range(len(data[i]))
        ]

        # Create the DataFrame for Seaborn
        import pandas as pd

        df = pd.DataFrame(
            {"Strain interval (%)": repeated_labels, "Hurst exponent": all_data}
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
            y="Hurst exponent",
            data=df,
            palette=colors,
            width=0.5,
            showmeans=False,
            meanline=False,
            flierprops=flierprops,
            whis=1.5,
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
        plt.ylabel("Hurst exponent", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=1000)

    def plot_non_affine_strain(self, nested_list, save_path):
        # Calculate means and standard deviations
        means = []
        stds = []
        x = range(len(nested_list))

        for i, sublist in enumerate(nested_list):
            flattened = [
                item for inner in sublist for item in inner
            ]  # Flatten the inner list
            # Calculate mean and standard deviation
            means.append(np.mean(flattened))
            stds.append(np.std(flattened) / 4)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            x,
            means,
            yerr=stds,
            fmt="o-",
            capsize=5,
            label="Mean ± Std Dev",
            color="blue",
        )

        # Customizations
        plt.title("Mean and Standard Deviation Across Sublists", fontsize=16)
        plt.xlabel("Sublist Index", fontsize=14)
        plt.ylabel("Values", fontsize=14)
        plt.xticks(range(len(nested_list)), fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=12)
        plt.savefig(save_path, dpi=1000)

    def plot_non_affine_curvature(self, nested_list, save_path):
        # Calculate means and standard deviations
        means = []
        stds = []
        x = range(len(nested_list))

        for i, sublist in enumerate(nested_list):
            flattened = [
                item for inner in sublist for item in inner
            ]  # Flatten the inner list
            # Calculate mean and standard deviation
            means.append(np.mean(flattened))
            stds.append(np.std(flattened) / 4)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            x,
            means,
            yerr=stds,
            fmt="o-",
            capsize=5,
            label="Mean ± Std Dev",
            color="blue",
        )

        # Customizations
        plt.title("Mean and Standard Deviation Across Sublists", fontsize=16)
        plt.xlabel("Sublist Index", fontsize=14)
        plt.ylabel("Values", fontsize=14)
        plt.xticks(range(len(nested_list)), fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=12)
        plt.savefig(save_path, dpi=1000)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def reMesh_disp(disp_folder):
    radius = residuals.radius

    mesh_path = "./data/mesh_hiRes.txt"
    keys = [
        "Z40C_092903b-192",
        "Z40C_093003b-192",
        "Z40C_100103a-192",
        "Z40C_100103b-192",
        "Z40C_100103d-192",
        "Z40C_100203a-192",
        "Z40C_100203b-192",
        "Z40C_100303b-192",
        "Z40C_101204a-192",
        "Z40C_120604a-192",
        "Z40C_120604c-192",
        "Z40C_120904e-192",
    ]

    filenames = [x for x in os.listdir(disp_folder)]
    filenames.sort()

    for start_frame, end_frame in pairwise(filenames):
        for key in keys:
            residual_disp = residuals.from_pickle(
                mesh_path=mesh_path,
                start_path="./data/fine_disp_data/" + start_frame,
                end_path="./data/fine_disp_data/" + end_frame,
                key=key,
            )

            xMesh_N, yMesh_N, zMesh_N = residual_disp.calc_coordinates()
            xMesh = xMesh_N * radius * 2
            yMesh = yMesh_N * radius * 2
            zMesh = zMesh_N * radius * 2

            # calculate displacement data
            u_rsdl, v_rsdl, w_rsdl = residual_disp.calc_residuals()

            # remesh the displacement data
            mesh_save_path = (
                "./data/fine_reMesh_mesh_v2/"
                + start_frame.split("_extrap")[0]
                + "_"
                + end_frame.split("_extrap")[0]
                + "_"
                + key
                + "_"
                + "Mesh.txt"
            )
            disp_save_path = (
                "./data/fine_reMesh_disp_v2/"
                + start_frame.split("_extrap")[0]
                + "_"
                + end_frame.split("_extrap")[0]
                + "_"
                + key
                + "_"
                + "Disp.txt"
            )

            new_mesh, new_disp = reMesh_non_regular(
                xMesh=xMesh,
                yMesh=yMesh,
                zMesh=zMesh,
                uDisp=u_rsdl,
                vDisp=v_rsdl,
                wDisp=w_rsdl,
                time_step=2,
            )

            # Save to pickle
            save2pickle(new_mesh, mesh_save_path)
            save2pickle(new_disp, disp_save_path)


def run_reMesh():
    disp_folder = "./data/fine_disp_data/"
    reMesh_disp(disp_folder=disp_folder)


def run_disp(disp_folder, start_frame, end_frame, save_path):
    displacements = Time_series(
        data_folder=disp_folder, start_frame=start_frame, end_frame=end_frame
    )
    avg_disp = displacements.avg_disp()

    displacements.plot_disp(disp=avg_disp, save_path=save_path)


def run_hurst(disp_folder, start_frame, end_frame, save_path):
    displacements = Time_series(
        data_folder=disp_folder, start_frame=start_frame, end_frame=end_frame
    )
    hurst_exponent = displacements.hurst_exponent()

    displacements.plot_hurst(hurst=hurst_exponent, save_path=save_path)


def run_hurst_strain(disp_folder, start_frame, end_frame, save_path):
    displacements = Time_series(
        data_folder=disp_folder, start_frame=start_frame, end_frame=end_frame
    )
    hurst_exponent = displacements.hurst_exponent()
    # non_affine_strain = displacements.non_affine_strain()

    displacements.plot_hurst(hurst=hurst_exponent, save_path=save_path)


def run_acf(disp_folder, start_frame, end_frame, save_path):
    displacements = Time_series(
        data_folder=disp_folder, start_frame=start_frame, end_frame=end_frame
    )
    acf_data = displacements.acf(max_lag=5)
    acf_list = displacements.process_acf_output(acf_data=acf_data)
    # non_affine_strain = displacements.non_affine_strain()

    displacements.plot_hurst(hurst=acf_list, save_path=save_path)


def run_non_affine_strain(disp_folder, start_frame, end_frame, save_path):
    displacements = Time_series(
        data_folder=disp_folder,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    non_affine_strain = displacements.non_affine_strain()

    displacements.plot_non_affine_strain(
        nested_list=non_affine_strain, save_path=save_path
    )


def run_non_affine_curvature(disp_folder, start_frame, end_frame, save_path):
    displacements = Time_series(
        data_folder=disp_folder,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    non_affine_strain = displacements.non_affine_curvature()

    displacements.plot_non_affine_curvature(
        nested_list=non_affine_strain, save_path=save_path
    )


def save2pickle(file, filePath):
    with open(filePath, "wb") as fp:  # Pickling
        pickle.dump(file, fp)


def main():
    # run_reMesh()

    run_hurst(
        disp_folder="./data/fine_reMesh_disp_v2/",
        start_frame="strain_1.0",
        end_frame="strain_9.0",
        save_path="./pics/non_affine_strain/hurst_1.0_9.0.pdf",
    )

    # run_hurst_strain(
    #     disp_folder="./data/fine_reMesh_disp_v2/",
    #     start_frame="strain_1.0",
    #     end_frame="strain_9.0",
    #     save_path="./pics/non_affine_strain/hurst_strain_1.0_9.0_f.pdf",
    # )

    # run_disp(
    #     disp_folder="./data/fine_reMesh_disp_v2/",
    #     start_frame="strain_1.0",
    #     end_frame="strain_9.0",
    #     save_path="./pics/non_affine_strain/disp_1.0_9.0.pdf",
    # )

    # run_acf(
    #     disp_folder="./data/fine_reMesh_disp_v2/",
    #     start_frame="strain_1.0",
    #     end_frame="strain_9.0",
    #     save_path="./pics/non_affine_strain/acf_1.0_9.0.pdf",
    # )

    # run_non_affine_strain(
    #     disp_folder="./data/fine_reMesh_disp/",
    #     start_frame = "strain_1.0",
    #     end_frame = "strain_9.0",
    #     save_path = "./pics/non_affine_strain/strain.jpg"
    # )

    # run_non_affine_curvature(
    #     disp_folder="./data/fine_reMesh_disp/",
    #     start_frame = "strain_1.0",
    #     end_frame = "strain_9.0",
    #     save_path = "./pics/non_affine_strain/curvature.jpg"
    # )


if __name__ == "__main__":
    main()
