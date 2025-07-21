#!/usr/bin/env python3
"""
Blender Script: Print class_id attributes for all objects in scene

This script loops through all objects in the current Blender scene
and prints their class_id attribute if it exists.
"""

import bpy
import numpy as np

def print_class_ids():
    """Print class_id attribute for all objects in the scene."""
    print("=== Scanning objects for class_id attributes ===")
    
    # Get all objects in the current scene
    objects = bpy.context.scene.objects
    
    if not objects:
        print("No objects found in the scene.")
        return
    
    found_objects = []
    
    for obj in objects:
        # Check if the object has a class_id attribute
        if 'class_id' in obj:
            class_id = obj['class_id']
            print(f"Object: '{obj.name}' | class_id: {class_id}")
            found_objects.append((obj.name, class_id))

    
    print(f"\n=== Summary ===")
    print(f"Total objects in scene: {len(objects)}")
    print(f"Objects with class_id: {len(found_objects)}")
    print(f"class ids: {np.unique([class_id for _, class_id in found_objects])}")
    # if found_objects:
    #     print("\nObjects with class_id values:")
    #     for name, class_id in found_objects:
    #         print(f"  - {name}: {class_id}")

# Execute the function
if __name__ == "__main__":
    print_class_ids()
