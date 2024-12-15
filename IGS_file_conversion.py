from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopoDS import TopoDS_Shape
import os

def convert_igs_to_stl(input_folder, output_folder):
    """
    Convert IGS files to STL and ensure geometry validation using bounding box checks.
    """
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".igs") or file_name.endswith(".IGS"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".igs", ".stl").replace(".IGS", ".stl"))

            try:
                # Read the IGS file
                reader = IGESControl_Reader()
                status = reader.ReadFile(input_path)

                if status == 0:
                    print(f"Failed to read IGS file: {file_name}")
                    continue

                reader.TransferRoots()
                shape = reader.OneShape()

                # Check if the shape is valid
                if shape.IsNull():
                    print(f"No valid geometry in file: {file_name}")
                    continue

                # Compute the bounding box of the shape
                bounding_box = Bnd_Box()
                brepbndlib.Add(shape, bounding_box)  # Updated method call

                # Check if the bounding box is valid
                if bounding_box.IsVoid():
                    print(f"File {file_name} contains no valid solid geometry.")
                    continue

                # Write to STL
                writer = StlAPI_Writer()
                writer.Write(shape, output_path)

                # Confirm file creation
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"Converted: {file_name} -> {output_path}")
                else:
                    print(f"Failed to write output for file: {file_name}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

# Define input and output folders
input_folder = r"C:\Users\Otso\Desktop\Directory\LUT opinnot\Dippa\codes\CAD_models"
output_folder = r"C:\Users\Otso\Desktop\Directory\LUT opinnot\Dippa\codes\converted_models"

convert_igs_to_stl(input_folder, output_folder)
