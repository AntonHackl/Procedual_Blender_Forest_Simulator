import bpy
import sys
import os
import argparse
import time

def enable_forest_generator_addon():
    """Enable the Forest Generator addon."""
    try:
        # The addon should be available as a module, not necessarily in preferences
        # Let's try to import it directly
        try:
            # Add the parent directory to sys.path if needed
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            
            # Try importing the module
            import __init__ as forest_module
            print("✓ Forest Generator module imported successfully")
            return True
        except ImportError as e:
            print(f"✗ Failed to import Forest Generator module: {e}")
            
            # Try enabling as addon if it's installed
            try:
                bpy.ops.preferences.addon_enable(module="PROCEDUAL_BLENDER_FOREST_SIMULATOR")
                print("✓ Forest Generator addon enabled")
                return True
            except Exception as enable_error:
                print(f"✗ Failed to enable Forest Generator addon: {enable_error}")
                return False
                
    except Exception as e:
        print(f"✗ Failed to check/enable Forest Generator addon: {e}")
        return False

def run_forest_generation(tree_config_path, surface_path):
    """Execute the forest generation process."""
    
    print("=== Starting Forest Generation Test ===")
    print(f"Tree config: {tree_config_path}")
    print(f"Surface file: {surface_path}")
    
    # Validate input files
    if not os.path.exists(tree_config_path):
        print(f"✗ Tree config file not found: {tree_config_path}")
        return
    
    if not os.path.exists(surface_path):
        print(f"✗ Surface file not found: {surface_path}")
        return
     
    # Clear existing mesh objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    
    # Reset cursor to origin
    bpy.context.scene.cursor.location = (0, 0, 0)
    
    # Enable addon
    if not enable_forest_generator_addon():
        print("Failed to enable Forest Generator addon")
        return
    
    try:
        # Alternative: Direct execution approach
        print("Executing forest generation directly...")
        
        # Import the forest generator class from the parent module
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        # Import and register the forest generator
        import __init__ as forest_module
        
        # Register the classes if not already registered
        try:
            forest_module.register()
            print("✓ Forest Generator classes registered")
        except Exception as reg_error:
            print(f"Note: Registration error (classes may already be registered): {reg_error}")
        
        # Now use the operator directly
        try:
            # Use the operator through bpy.ops
            result = bpy.ops.mesh.forest_generator(
                updateForest=True,
                surface=surface_path,
                treeConfigurationCount=1
            )
            print(f"✓ Forest generation operator result: {result}")
            
        except Exception as op_error:
            print(f"Operator approach failed: {op_error}")
            
            # Fallback: Direct class instantiation
            print("Trying direct class instantiation...")
            forest_gen = forest_module.ForestGenerator()
            
            # Set properties
            forest_gen.updateForest = True
            forest_gen.surface = surface_path
            forest_gen.treeConfigurationCount = 1
            
            # Add a tree configuration
            forest_gen.tree_configurations.add()
            forest_gen.tree_configurations[0].path = tree_config_path
            forest_gen.tree_configurations[0].weight = 1.0
            
            # Execute with timing
            start_time = time.time()
            result = forest_gen.execute(bpy.context)
            end_time = time.time()
            
            print(f"✓ Forest generation completed in {end_time - start_time:.2f} seconds")
            print(f"✓ Execution result: {result}")
            
            # Enable timing output for future runs
            forest_gen.timePerformance = True
        
        # Count generated objects
        tree_count = len([obj for obj in bpy.context.scene.objects if obj.name.startswith("Tree_")])
        leaf_count = len([obj for obj in bpy.context.scene.objects if obj.name.startswith("Leaf_")])
        
        print(f"✓ Generated {tree_count} trees and {leaf_count} leaf objects")
        
    except Exception as e:
        print(f"✗ Error during forest generation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("✓ Forest generation process completed")

def list_available_operators():
    """List all available mesh operators for debugging."""
    print("\n=== Available mesh operators ===")
    for op_name in dir(bpy.ops.mesh):
        if not op_name.startswith("_"):
            print(f"  bpy.ops.mesh.{op_name}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Forest Generator with specified configuration files')
    parser.add_argument('tree_config', help='Path to tree configuration JSON file')
    parser.add_argument('surface_file', help='Path to surface CSV file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    # Parse args from sys.argv, but handle Blender's arguments
    # Blender passes script arguments after '--'
    if '--' in sys.argv:
        script_args = sys.argv[sys.argv.index('--') + 1:]
    else:
        script_args = sys.argv[1:]
    
    return parser.parse_args(script_args)

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        if args.verbose:
            # List enabled addons
            print("Enabled Addons:")
            addon_list = []
            for addon in bpy.context.preferences.addons.keys():
                print(f"  {addon}")
                addon_list.append(addon)
            
            print(f'\nIs Forest Generator enabled: {"PROCEDUAL_BLENDER_FOREST_SIMULATOR" in addon_list}')
            list_available_operators()
        
        # Run the forest generation test with provided files
        run_forest_generation(args.tree_config, args.surface_file)
        
    except SystemExit:
        # Handle argparse help/error exits
        pass
    except Exception as e:
        print(f"✗ Script execution failed: {e}")
        import traceback
        traceback.print_exc()