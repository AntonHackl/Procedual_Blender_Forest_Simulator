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
import csv


def csv_to_las(csv_path, las_path):
    """
    Convert CSV file to LAS format with specified column mappings.
    
    Args:
        csv_path (str): Path to input CSV file
        las_path (str): Path to output LAS file
    """
    try:
        print(f"Reading CSV file: {csv_path}")
        # Fast C-engine with robust options (no Python engine fallback)
        required_columns = ['x', 'y', 'z', 'ObjectID', 'MaterialID']
        df = pd.read_csv(
            csv_path,
            engine='c',
            sep=',',
            on_bad_lines='skip',  # skip any rare malformed lines while staying fast
            skipinitialspace=True,
            memory_map=True,
            usecols=lambda c: c in required_columns
        )

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            # Try case-insensitive recovery for common casing variations
            lower_map = {c.lower(): c for c in df.columns}
            rename_map = {}
            for need in required_columns:
                if need not in df.columns and need.lower() in lower_map:
                    rename_map[lower_map[need.lower()]] = need
            if rename_map:
                df = df.rename(columns=rename_map)
                missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

        # Coerce numeric types and drop invalid rows efficiently
        for col in ['x', 'y', 'z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in ['ObjectID', 'MaterialID']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        before_count = len(df)
        df = df.dropna(subset=['x', 'y', 'z', 'ObjectID', 'MaterialID'])
        dropped = before_count - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with non-numeric or missing required values")

        x = df['x'].to_numpy(dtype=np.float64)
        y = df['y'].to_numpy(dtype=np.float64)
        z = df['z'].to_numpy(dtype=np.float64)
        instance_id = df['ObjectID'].to_numpy(dtype=np.int32)
        semantic_class_raw = df['MaterialID'].to_numpy(dtype=np.int32)

        # New mapping logic (updated again per user request):
        # Special rule: for original class 10 points, check if their instance contains ANY original class 2 or 3 point.
        #   If rule satisfied -> mapped to 3
        #   If rule NOT satisfied -> 0
        # Base mappings:
        #   original 0 -> 1
        #   original 2 -> 2
        #   original 3 -> 4
        #   original 10 -> 3 (rule satisfied) else 0 (rule unsatisfied)
        # Resulting allowed semantic_id set: {0,1,2,3,4}

        mask0 = semantic_class_raw == 0
        mask2 = semantic_class_raw == 2
        mask3 = semantic_class_raw == 3
        mask10 = semantic_class_raw == 10

        # Determine per-instance presence of any original 2 or 3 (for special rule)
        df_tmp = pd.DataFrame({
            'instance_id': instance_id,
            'has_2_or_3': ((mask2 | mask3).astype(np.uint8))
        })
        has_2_or_3_instance = df_tmp.groupby('instance_id')['has_2_or_3'].transform('max').astype(bool).values

        semantic_class = np.empty_like(semantic_class_raw, dtype=np.int32)
        semantic_class.fill(-1)

        # Final mapping rules now:
        #   original 0  -> 1
        #   original 2  -> 2
        #   original 3  -> 4
        #   original 10 -> 3 (if rule satisfied) else 0 (if rule NOT satisfied)
        #   (unsatisfied special rule keeps value 0 to distinguish from original 0 which became 1)
        # Allowed semantic set: {0,1,2,3,4}

        semantic_class[mask0] = 1
        semantic_class[mask2] = 2
        semantic_class[mask3] = 4
        semantic_class[mask10 & has_2_or_3_instance] = 3
        semantic_class[mask10 & (~has_2_or_3_instance)] = 0

        # Keep only semantic classes in {0,1,2,3,4}
        valid_classes = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        valid_mask = np.isin(semantic_class, valid_classes)
        dropped_invalid = len(semantic_class) - int(np.count_nonzero(valid_mask))
        if dropped_invalid > 0:
            print(f"Dropped {dropped_invalid} points outside allowed semantic classes {valid_classes.tolist()}")

        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]
        instance_id = instance_id[valid_mask]
        semantic_class = semantic_class[valid_mask]

        # Instance ID policy: retain only for semantic classes 2,3,4 (requested), else set to -1
        keep_inst_mask = np.isin(semantic_class, [2, 3, 4])
        # Set to -1 where not in 2,3,4
        instance_id[~keep_inst_mask] = -1
        # Normalize instance_ids for kept classes only
        unique_ids = np.unique(instance_id[keep_inst_mask])
        id_map = {old: new for new, old in enumerate(unique_ids)}
        # Only update where not -1
        for idx in np.where(keep_inst_mask)[0]:
            instance_id[idx] = id_map[instance_id[idx]]

        print(f"Processing {len(df)} points...")

        header = laspy.LasHeader(point_format=2, version="1.2")
        header.offsets = [np.min(x), np.min(y), np.min(z)]
        header.scales = [0.001, 0.001, 0.001]

        header.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.int32))

        las = laspy.LasData(header)
        las.x = x
        las.y = y
        las.z = z

        las.instance_id = instance_id
        las.classification = semantic_class.astype(np.uint8)

        print(f"Writing LAS file: {las_path}")
        las.write(las_path)

        print(f"Successfully converted {len(df)} points to LAS format")
        print(f"Output file: {las_path}")

        print("\nSummary:")
        print(f"  Total input points: {len(df)}")
        print(f"  Total output points: {len(x)}")
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
    
    DEFAULT_CSV_PATHS = [
        f"C:\\Users\\anton\\Documents\\Uni\\Spatial_Data_Analysis\\datasets\\full_plot{i}.csv"
        for i in range(1, 16)
    ]
    DEFAULT_LAS_PATHS = [
        f"C:\\Users\\anton\\Documents\\Uni\\Spatial_Data_Analysis\\datasets\\finished_plot{i}.las"
        for i in range(1, 16)
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
