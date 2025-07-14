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
        semantic_class = df['MaterialID'].values.astype(np.int32)
        
        print(f"Processing {len(df)} points...")
        
        header = laspy.LasHeader(point_format=2, version="1.2")
        header.offsets = [np.min(x), np.min(y), np.min(z)]
        header.scales = [0.001, 0.001, 0.001] 

        header.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.int32))
        header.add_extra_dim(laspy.ExtraBytesParams(name="semantic_class", type=np.int32))

        las = laspy.LasData(header)
        las.x = x
        las.y = y
        las.z = z
        
        las.instance_id = instance_id
        las.semantic_class = semantic_class
        
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
    if len(sys.argv) != 3:
        print("Usage: python blender_scan_to_pointtorch_dataset.py input.csv output.las")
        print("\nArguments:")
        print("  input.csv  - Path to input CSV file")
        print("  output.las - Path to output LAS file")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    las_path = sys.argv[2]
    
    if not Path(csv_path).exists():
        print(f"Error: Input file '{csv_path}' does not exist")
        sys.exit(1)
    
    output_dir = Path(las_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_to_las(csv_path, las_path)


if __name__ == "__main__":
    main()
