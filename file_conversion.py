import FreeCAD
import Part
import Mesh
import os

def convert_step_to_stl_recursive(input_folder, output_folder):
    """
    Recursively convert STEP/IGES files to STL and preserve directory structure.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Walk through all directories and files
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.lower().endswith((".step", ".stp", ".igs", ".iges")):
                input_path = os.path.join(root, file_name)

                # Build the relative path and output path
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Generate output file path with .stl extension
                output_file_base = os.path.splitext(file_name)[0] + ".stl"
                output_path = os.path.join(output_dir, output_file_base)

                try:
                    print(f"Processing: {input_path}")
                    # Create a new FreeCAD document
                    doc = FreeCAD.newDocument()
                    part = None

                    # Load the file as a Part object
                    if file_name.lower().endswith((".step", ".stp")):
                        part = Part.read(input_path)
                    elif file_name.lower().endswith((".igs", ".iges")):
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
input_folder = r"C:\AI2CAD"  # Parent folder containing STEP/IGES files
output_folder = r"C:\AI2CAD\stl_conversion_output"  # Output folder for converted STL files

# Run the conversion
convert_step_to_stl_recursive(input_folder, output_folder)
