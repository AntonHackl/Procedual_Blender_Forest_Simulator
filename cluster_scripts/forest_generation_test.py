import bpy
import sys
import os
import argparse
import time

def run_forest_generation(tree_config_path, surface_path, output_path=None):
    """Execute the forest generation process."""
    
    print("=== Starting Forest Generation Test ===")
    print(f"Tree config: {tree_config_path}")
    print(f"Surface file: {surface_path}")
    if output_path:
        print(f"Output file: {output_path}")
    
    # Validate input files
    if not os.path.exists(tree_config_path):
        print(f"✗ Tree config file not found: {tree_config_path}")
        return False
    
    if not os.path.exists(surface_path):
        print(f"✗ Surface file not found: {surface_path}")
        return False
    
    # Reset cursor to origin
    bpy.context.scene.cursor.location = (0, 0, 0)
    
    try:
        # Generate forest
        result = bpy.ops.mesh.forest_generator(
            updateForest=True,
            surface=surface_path,
            treeConfigurationCount=1,
            tree_configurations=[{"name": "tree_config_1", "path": tree_config_path, "weight": 1.0}]
        )
        print(f"✓ Forest generation operator result: {result}")
        
        # Count generated objects
        tree_count = len([obj for obj in bpy.context.scene.objects if obj.name.startswith("Tree_")])
        leaf_count = len([obj for obj in bpy.context.scene.objects if obj.name.startswith("Leaf_")])
        
        print(f"✓ Generated {tree_count} trees and {leaf_count} leaf objects")
        
        # Save the file if output path is specified
        if output_path:
            success = save_scene(output_path)
            if success:
                print(f"✓ Scene saved to: {output_path}")
                return True
            else:
                print(f"✗ Failed to save scene to: {output_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error during forest generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_scene(output_path):
    """Save the current Blender scene to the specified path."""
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Determine file format based on extension
        file_ext = os.path.splitext(output_path)[1].lower()
        
        if file_ext == '.blend':
            # Save as Blender file
            bpy.ops.wm.save_as_mainfile(filepath=output_path)
            print(f"✓ Saved Blender file: {output_path}")
        else:
            print(f"✗ Unsupported file format: {file_ext}")
            print("Supported formats: .blend, .obj, .ply, .stl, .fbx, .gltf, .glb")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        import traceback
        traceback.print_exc()
        return False

def list_available_operators():
    """List all available mesh operators for debugging."""
    print("\n=== Available mesh operators ===")
    for op_name in dir(bpy.ops.mesh):
        if not op_name.startswith("_"):
            print(f"  bpy.ops.mesh.{op_name}")

def list_available_addon_modules():
    """List all available addon modules for debugging."""
    print("\n=== Available Addon Modules ===")
    
    import addon_utils
    available_modules = []
    
    for module in addon_utils.modules():
        module_name = module.__name__
        if hasattr(module, 'bl_info'):
            bl_info = module.bl_info
            name = bl_info.get('name', 'Unknown')
            enabled = module_name in bpy.context.preferences.addons
            
            available_modules.append(module_name)
            status = "✓ ENABLED" if enabled else "○ Available"
            print(f"{status} | {module_name} - {name}")
    
    print(f"\nTotal available modules: {len(available_modules)}")
    return available_modules

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Forest Generator with specified configuration files')
    parser.add_argument('tree_config', nargs='?', help='Path to tree configuration JSON file')
    parser.add_argument('surface_file', nargs='?', help='Path to surface CSV file')
    parser.add_argument('--output', '-o', help='Output file path (.blend, .obj, .ply, .stl, .fbx, .gltf, .glb)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--list-addons', '-l', action='store_true', help='List available addon modules and exit')
    
    # Parse args from sys.argv, but handle Blender's arguments
    # Blender passes script arguments after '--'
    if '--' in sys.argv:
        script_args = sys.argv[sys.argv.index('--') + 1:]
    else:
        script_args = sys.argv[1:]
    
    return parser.parse_args(script_args)

if __name__ == "__main__":
    # List enabled addons
    print("Enabled Addons:")
    addon_list = []
    for addon in bpy.context.preferences.addons.keys():
        print(f"  {addon}")
        addon_list.append(addon)
    
    # List all available addon modules
    available_modules = list_available_addon_modules()
    
    # Check if forest generator addon is available
    forest_addon_found = False
    if "Procedual_Blender_Forest_Simulator" not in addon_list:
        bpy.ops.preferences.addon_enable(module="Procedual_Blender_Forest_Simulator")
        print("Addon Procedual_Blender_Forest_Simulator installed and enabled.")
    
    # List available mesh operations
    print("\n=== Available mesh operators ===")
    for op in dir(bpy.ops.mesh):
        if not op.startswith("_") and "forest" in op.lower():
            print(f"  bpy.ops.mesh.{op}")
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Handle list-addons option
        if args.list_addons:
            print("Exiting after listing available addons.")
            sys.exit(0)
        
        if args.verbose:
            list_available_operators()
        
        # Validate required arguments if not just listing addons
        if not args.tree_config or not args.surface_file:
            print("✗ Error: tree_config and surface_file are required when not using --list-addons")
            print("Use --help for usage information")
            sys.exit(1)
        
        # Run the forest generation test with provided files
        success = run_forest_generation(args.tree_config, args.surface_file, args.output)
        
        if success:
            print("✓ Forest generation completed successfully")
            sys.exit(0)
        else:
            print("✗ Forest generation failed")
            sys.exit(1)
        
    except SystemExit:
        # Handle argparse help/error exits
        pass
    except Exception as e:
        print(f"✗ Script execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)