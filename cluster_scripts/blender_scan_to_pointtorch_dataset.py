"""
CSV to LAS Converter for Blender Forest Simulator

This script converts CSV files containing point cloud data to LAS format.
The script maps the following columns:
- x, y, z: coordinates
- semclassId: maps to instance_id in LAS
- MaterialID: maps to semantic_class in LAS

Usage:
    python blender_scan_to_pointtorch_dataset.py input.csv output.las
"""

import sys
import pandas as pd
import laspy
import numpy as np
from pathlib import Path


def csv_to_las(csv_path, las_path):
    """
    Convert CSV file to LAS format with specified column mappings.
    
    Args:
        csv_path (str): Path to input CSV file
        las_path (str): Path to output LAS file
    """
    try:
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        required_columns = ['x', 'y', 'z', 'SemClassID', 'MaterialID']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        instance_id = df['SemClassID'].values.astype(np.int32)
        semantic_class_raw = df['MaterialID'].values.astype(np.int32)
        
        # Map semantic classes: 1->0, 2->1, 10->2
        def map_semantic_class(original_class):
            mapping = {1: 0, 2: 1, 10: 2}
            return mapping.get(original_class, original_class)
        
        # Apply mapping to all semantic classes
        semantic_class = np.array([map_semantic_class(cls) for cls in semantic_class_raw], dtype=np.int32)
        
        print(f"Processing {len(df)} points...")
        
        header = laspy.LasHeader(point_format=2, version="1.2")
        header.offsets = [np.min(x), np.min(y), np.min(z)]
        header.scales = [0.001, 0.001, 0.001] 

        header.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.int32))
        # header.add_extra_dim(laspy.ExtraBytesParams(name="semantic_class", type=np.int32))

        las = laspy.LasData(header)
        las.x = x
        las.y = y
        las.z = z
        
        las.instance_id = instance_id
        las.classification = semantic_class
        
        print(f"Writing LAS file: {las_path}")
        las.write(las_path)
        
        print(f"Successfully converted {len(df)} points to LAS format")
        print(f"Output file: {las_path}")
        
        print("\nSummary:")
        print(f"  Total points: {len(df)}")
        print(f"  X range: {np.min(x):.3f} to {np.max(x):.3f}")
        print(f"  Y range: {np.min(y):.3f} to {np.max(y):.3f}")
        print(f"  Z range: {np.min(z):.3f} to {np.max(z):.3f}")
        print(f"  Unique instance IDs: {len(np.unique(instance_id))}")
        print(f"  Unique semantic classes: {len(np.unique(semantic_class))}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and execute conversion."""
    
    # Hardcoded file paths - modify these as needed
    DEFAULT_CSV_PATHS = [
        r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\datasets\combined_plot1.csv",
        r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\datasets\combined_plot2.csv",
        r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\datasets\combined_plot3.csv",
        r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\datasets\combined_plot4.csv",
        r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\datasets\combined_plot5.csv",
    ]
    DEFAULT_LAS_PATHS = [
        r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\datasets\finished_plot1.las",
        r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\datasets\finished_plot2.las",
        r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\datasets\finished_plot3.las",
        r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\datasets\finished_plot4.las",
        r"C:\Users\anton\Documents\Uni\Spatial_Data_Analysis\datasets\finished_plot5.las",
    ]

    # Check if command line arguments are provided
    if len(sys.argv) == 3:
        # Use command line arguments
        csv_paths = [sys.argv[1]]
        las_paths = [sys.argv[2]]
        print("Using command line arguments:")
    elif len(sys.argv) == 1:
        # Use hardcoded paths
        csv_paths = DEFAULT_CSV_PATHS
        las_paths = DEFAULT_LAS_PATHS   
        print("Using hardcoded file paths:")
    else:
        print("Usage: python blender_scan_to_pointtorch_dataset.py [input.csv output.las]")
        print("\nOptions:")
        print("  1. Run with no arguments to use hardcoded paths:")
        print(f"     CSV: {DEFAULT_CSV_PATHS}")
        print(f"     LAS: {DEFAULT_LAS_PATHS}")
        print("  2. Run with two arguments:")
        print("     python script.py input.csv output.las")
        sys.exit(1)

    print(f"  Input CSV: {csv_paths}")
    print(f"  Output LAS: {las_paths}")
    print()

    for csv_path, las_path in zip(csv_paths, las_paths):
        if not Path(csv_path).exists():
            print(f"Error: Input file '{csv_path}' does not exist")
            sys.exit(1)
    
    output_dir = Path(las_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for csv_path, las_path in zip(csv_paths, las_paths):
        print(f"Converting {csv_path} to {las_path}")
        csv_to_las(csv_path, las_path)


if __name__ == "__main__":
    main()
