import Mesh
import os
import csv

def extract_features(input_folder, output_csv):
    """
    Extract geometric features from STL files, including edge and vertex counts, and save to a CSV file.
    """
    features = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".stl"):
            input_path = os.path.join(input_folder, file_name)
            print(f"Processing: {file_name}")

            try:
                # Load the STL mesh
                mesh = Mesh.Mesh(input_path)

                # Extract basic geometric features
                volume = mesh.Volume
                surface_area = mesh.Area
                bounds = mesh.BoundBox
                bounding_box_dimensions = (bounds.XLength, bounds.YLength, bounds.ZLength)
                bounding_box_volume = bounds.XLength * bounds.YLength * bounds.ZLength

                # Calculate additional features
                compactness = volume / surface_area if surface_area > 0 else 0
                aspect_ratio_xy = bounds.XLength / bounds.YLength if bounds.YLength > 0 else 0
                aspect_ratio_xz = bounds.XLength / bounds.ZLength if bounds.ZLength > 0 else 0
                aspect_ratio_yz = bounds.YLength / bounds.ZLength if bounds.ZLength > 0 else 0

                # Triangle count
                triangle_count = len(mesh.Facets)

                # Mesh density
                mesh_density = triangle_count / surface_area if surface_area > 0 else 0

                # Vertex count
                vertex_count = len(mesh.Points)

                # Estimate edge count (approximation for triangular meshes)
                edge_count = triangle_count * 3 // 2  # Approximation for closed triangular meshes

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
                    "aspect_ratio_xy": aspect_ratio_xy,
                    "aspect_ratio_xz": aspect_ratio_xz,
                    "aspect_ratio_yz": aspect_ratio_yz,
                    "triangle_count": triangle_count,
                    "mesh_density": mesh_density,
                    "edge_count": edge_count,
                    "vertex_count": vertex_count,
                })

            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

    # Save features to a CSV file
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "file_name", "volume", "surface_area",
            "bounding_box_x", "bounding_box_y", "bounding_box_z",
            "bounding_box_volume", "compactness",
            "aspect_ratio_xy", "aspect_ratio_xz", "aspect_ratio_yz",
            "triangle_count", "mesh_density", "edge_count", "vertex_count"
        ])
        writer.writeheader()
        writer.writerows(features)

    print(f"Features saved to {output_csv}")

# Define input folder and output CSV file
input_folder = r"C:\Users\Otso\Desktop\Directory\LUT opinnot\Dippa\codes\converted_models"
output_csv = r"C:\Users\Otso\Desktop\Directory\LUT opinnot\Dippa\codes\features.csv"

# Run feature extraction
extract_features(input_folder, output_csv)
