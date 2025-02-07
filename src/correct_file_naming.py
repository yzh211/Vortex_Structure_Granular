import os


def add_underscore_to_files(folder_path):
    """
    Adds an underscore after the first strain value in filenames like
    "strain_1.0strain_2.0_Z40C_092903b-192_Disp", transforming them to
    "strain_1.0_strain_2.0_Z40C_092903b-192_Disp".

    Args:
        folder_path (str): Path to the folder containing the files.
    """
    for file_name in os.listdir(folder_path):
        # Check if the file name contains the pattern "strain_x.xstrain_y.y"
        if "strain_" in file_name:
            parts = file_name.split("strain_", 1)  # Split only at the first occurrence
            strain_part = parts[1]  # This contains the remaining part of the file
            if "strain_" in strain_part:
                # Add the underscore after the first strain value
                new_file_name = (
                    f"{parts[0]}strain_{strain_part.replace('strain_', '_strain_', 1)}"
                )

                # Rename the file
                old_path = os.path.join(folder_path, file_name)
                new_path = os.path.join(folder_path, new_file_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {file_name} -> {new_file_name}")


# Example usage
folder_path = "./data/fine_reMesh_mesh/"  # Replace with your actual folder path
add_underscore_to_files(folder_path)
