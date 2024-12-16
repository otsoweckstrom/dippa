import os
import csv
import Mesh  # FreeCAD's built-in Mesh module

def extract_features(parent_folder, output_csv):
    """
    Extract geometric features from STL files in all subfolders and save to a CSV file.
    """
    features = []

    # Walk through each subfolder and file
    for root, dirs, files in os.walk(parent_folder):
        print(f"Scanning folder: {root}")  # Debug: print folder being scanned

        for file_name in files:
            if file_name.lower().endswith(".stl"):  # Case-insensitive check
                input_path = os.path.join(root, file_name)
                print(f"Processing file: {input_path}")  # Debug: print full file path

                try:
                    # Load the STL mesh using FreeCAD's Mesh module
                    mesh = Mesh.Mesh(input_path)

                    # Extract basic geometric features
                    volume = mesh.Volume
                    surface_area = mesh.Area
                    bounds = mesh.BoundBox
                    bounding_box_dimensions = (bounds.XLength, bounds.YLength, bounds.ZLength)
                    bounding_box_volume = bounds.XLength * bounds.YLength * bounds.ZLength

                    # Additional calculations
                    compactness = volume / surface_area if surface_area > 0 else 0
                    triangle_count = len(mesh.Facets)
                    edge_count = triangle_count * 3 // 2  # Approximate edge count for triangular meshes
                    vertex_count = len(mesh.Points)

                    # Append extracted features
                    features.append({
                        "file_name": file_name,
                        "volume": volume,
                        "surface_area": surface_area,
                        "bounding_box_x": bounding_box_dimensions[0],
                        "bounding_box_y": bounding_box_dimensions[1],
                        "bounding_box_z": bounding_box_dimensions[2],
                        "bounding_box_volume": bounding_box_volume,
                        "compactness": compactness,
                        "triangle_count": triangle_count,
                        "edge_count": edge_count,
                        "vertex_count": vertex_count,
                    })

                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")

    # Save features to a CSV file
    if features:
        with open(output_csv, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=features[0].keys())
            writer.writeheader()
            writer.writerows(features)

        print(f"Features saved to {output_csv}")
    else:
        print("No STL files found!")

# Define parent folder and output CSV file
parent_folder = r"C:\AI2CAD"  # r"C:\AI2CAD\stl_conversion_output" for converted stp_files
output_csv = r"C:\AI2CAD\output\features.csv" # r"C:\AI2CAD\output\stp_features.csv" for outputting the converted stp_files data

# Run feature extraction
extract_features(parent_folder, output_csv)
