import FreeCAD
import Part
import Mesh
import os

def convert_step_to_stl(input_folder, output_folder):
    """
    Convert STEP/IGES files to STL using FreeCAD.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".step") or file_name.endswith(".stp") or file_name.endswith(".igs") or file_name.endswith(".iges") or file_name.endswith(".IGS"):
            input_path = os.path.join(input_folder, file_name)

            # Ensure the output file has the correct STL extension
            output_file_base = file_name.replace(".step", ".stl").replace(".stp", ".stl").replace(".igs", ".stl").replace(".iges", ".stl").replace(".IGS", ".stl")
            output_path = os.path.join(output_folder, output_file_base)

            try:
                # Load the file as a Part object
                doc = FreeCAD.newDocument()
                part = None

                if file_name.endswith(".step") or file_name.endswith(".stp"):
                    part = Part.read(input_path)
                elif file_name.endswith(".igs") or file_name.endswith(".iges"):
                    part = Part.read(input_path)

                if part is None:
                    print(f"Failed to load geometry for file: {input_path}")
                    continue

                # Add the part to the FreeCAD document
                part_feature = doc.addObject("Part::Feature", "Part")
                part_feature.Shape = part

                # Export as STL
                Mesh.export([part_feature], output_path)
                print(f"Converted: {input_path} -> {output_path}")

                # Close the FreeCAD document
                FreeCAD.closeDocument(doc.Name)

            except Exception as e:
                print(f"Error processing file {input_path}: {e}")

# Define input and output folders
input_folder = r"C:\Users\Otso\Desktop\Directory\LUT opinnot\Dippa\codes\CAD_models"
output_folder = r"C:\Users\Otso\Desktop\Directory\LUT opinnot\Dippa\codes\converted_models"

# Run the conversion
convert_step_to_stl(input_folder, output_folder)
