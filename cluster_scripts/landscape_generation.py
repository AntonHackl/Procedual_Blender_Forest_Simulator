import bpy
import sys
import os
import argparse
import time
import random

def clear_scene():
    """Clear all objects from the current scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear any remaining mesh data
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

def add_landscape_with_params(noise_type, seed, mesh_size_x=15, mesh_size_y=15, height=0.75, max_height=1.25):
    """Add a landscape using the landscape addon with specified parameters."""
    try:
        # Clear the scene first
        clear_scene()
        
        # Set the random seed for reproducibility
        random.seed(seed)
        
        # Check if the landscape addon is available
        addon_name = "add_mesh_extra_objects"  # This is the common name for landscape addon
        if addon_name not in bpy.context.preferences.addons:
            try:
                bpy.ops.preferences.addon_enable(module=addon_name)
                print(f"✓ Enabled addon: {addon_name}")
            except:
                print(f"✗ Could not enable addon: {addon_name}")
                return False
        
        # Add landscape mesh
        # The exact operator name might vary depending on the addon version
        # Common names are: add_mesh_landscape, add_landscape, landscape_add
        landscape_ops = [
            'add_mesh_landscape',
            'landscape_add', 
            'add_landscape',
            'mesh_landscape_add'
        ]
        
        landscape_added = False
        for op_name in landscape_ops:
            if hasattr(bpy.ops.mesh, op_name):
                try:
                    # Get the operator
                    op = getattr(bpy.ops.mesh, op_name)
                    
                    # Prepare parameters based on noise type
                    landscape_params = {
                        'mesh_size_x': mesh_size_x,
                        'mesh_size_y': mesh_size_y,
                        'height': height,
                        'maximum': max_height,
                        'random_seed': seed
                    }
                    
                    # Set noise type specific parameters
                    if noise_type == "ANoise":
                        landscape_params['noise_type'] = 'another_noise'
                    elif noise_type == "MultiFractal":
                        landscape_params['noise_type'] = 'multi_fractal'
                    elif noise_type == "HeteroTerrain":
                        landscape_params['noise_type'] = 'hetero_terrain'
                    
                    # Call the operator with parameters
                    result = op(**landscape_params)
                    print(f"✓ Added landscape using {op_name} with noise type: {noise_type}, seed: {seed}")
                    landscape_added = True
                    break
                    
                except Exception as e:
                    print(f"Failed to use {op_name}: {e}")
                    continue
        
        if not landscape_added:
            # Try a more basic approach - add a plane and modify it
            print("Trying alternative landscape generation approach...")
            bpy.ops.mesh.primitive_plane_add(size=max(mesh_size_x, mesh_size_y))
            
            # Get the active object (the plane we just added)
            obj = bpy.context.active_object
            if obj and obj.type == 'MESH':
                # Add subdivision surface modifier
                bpy.ops.object.modifier_add(type='SUBSURF')
                obj.modifiers["Subdivision Surface"].levels = 3
                
                # Add displacement modifier for terrain effect
                bpy.ops.object.modifier_add(type='DISPLACE')
                displace_mod = obj.modifiers["Displace"]
                
                # Create a new texture for displacement
                texture = bpy.data.textures.new(f"Landscape_Texture_{seed}", type='CLOUDS')
                texture.noise_scale = 0.5
                texture.noise_depth = 4
                texture.cloud_type = 'GRAYSCALE'
                
                # Set texture based on noise type
                if noise_type == "ANoise":
                    texture.noise_basis = 'BLENDER_ORIGINAL'
                elif noise_type == "MultiFractal":
                    texture.noise_basis = 'PERLIN_ORIGINAL'
                elif noise_type == "HeteroTerrain":
                    texture.noise_basis = 'VORONOI_F1'
                
                displace_mod.texture = texture
                displace_mod.strength = height
                displace_mod.mid_level = 0.5
                
                # Apply random seed by moving the texture coordinates
                obj.location = (random.uniform(-10, 10), random.uniform(-10, 10), 0)
                
                print(f"✓ Created alternative landscape with noise type: {noise_type}, seed: {seed}")
                landscape_added = True
        
        return landscape_added
        
    except Exception as e:
        print(f"✗ Error adding landscape: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_landscape_scenes(output_dir="C:/temp/landscapes"):
    """Generate 15 landscape scenes with different noise types and seeds."""
    
    print("=== Starting Landscape Scene Generation ===")
    
    # Define noise types and seeds
    noise_types = ["ANoise", "MultiFractal", "HeteroTerrain"]
    seeds_per_type = 5
    
    # Parameters for landscape generation
    mesh_size_x = 15
    mesh_size_y = 15
    height = 0.75
    max_height = 1.25
    
    scenes_generated = []
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    scene_counter = 1
    
    for noise_type in noise_types:
        print(f"\n--- Generating scenes with {noise_type} noise ---")
        
        for seed_num in range(1, seeds_per_type + 1):
            seed = seed_num * 100 + hash(noise_type) % 1000  # Create varied seeds
            
            print(f"Generating scene {scene_counter}/15: {noise_type} with seed {seed}")
            
            # Add landscape to current scene
            success = add_landscape_with_params(
                noise_type=noise_type,
                seed=seed,
                mesh_size_x=mesh_size_x,
                mesh_size_y=mesh_size_y,
                height=height,
                max_height=max_height
            )
            
            if success:
                # Save the scene
                scene_name = f"landscape_{noise_type}_{seed}.blend"
                scene_path = os.path.join(output_dir, scene_name)
                
                try:
                    bpy.ops.wm.save_as_mainfile(filepath=scene_path)
                    print(f"✓ Saved scene: {scene_path}")
                    scenes_generated.append(scene_path)
                except Exception as e:
                    print(f"✗ Failed to save scene {scene_path}: {e}")
            else:
                print(f"✗ Failed to generate landscape for {noise_type} with seed {seed}")
            
            scene_counter += 1
    
    print(f"\n=== Landscape Generation Complete ===")
    print(f"Successfully generated {len(scenes_generated)} out of 15 scenes")
    print("Generated scenes:")
    for scene_path in scenes_generated:
        print(f"  {scene_path}")
    
    return scenes_generated

def list_available_landscape_operators():
    """List available landscape-related operators for debugging."""
    print("\n=== Available landscape operators ===")
    landscape_related = []
    
    # Check mesh operators
    for op_name in dir(bpy.ops.mesh):
        if not op_name.startswith("_") and any(keyword in op_name.lower() for keyword in ['landscape', 'terrain', 'heightfield', 'displacement']):
            landscape_related.append(f"bpy.ops.mesh.{op_name}")
    
    # Check add operators
    for op_name in dir(bpy.ops.mesh):
        if not op_name.startswith("_") and 'add' in op_name.lower():
            landscape_related.append(f"bpy.ops.mesh.{op_name}")
    
    if landscape_related:
        for op in landscape_related:
            print(f"  {op}")
    else:
        print("  No landscape-specific operators found")
    
    return landscape_related

def list_available_addons():
    """List all available addons, focusing on those that might contain landscape functionality."""
    print("\n=== Available Addons (focusing on landscape/terrain related) ===")
    
    import addon_utils
    landscape_addons = []
    
    for module in addon_utils.modules():
        module_name = module.__name__
        if hasattr(module, 'bl_info'):
            bl_info = module.bl_info
            name = bl_info.get('name', 'Unknown')
            enabled = module_name in bpy.context.preferences.addons
            
            # Check if addon might be landscape related
            landscape_keywords = ['landscape', 'terrain', 'heightfield', 'extra', 'object', 'mesh']
            if any(keyword in name.lower() or keyword in module_name.lower() for keyword in landscape_keywords):
                status = "✓ ENABLED" if enabled else "○ Available"
                print(f"{status} | {module_name} - {name}")
                landscape_addons.append(module_name)
    
    return landscape_addons

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate landscape scenes using Blender landscape addon')
    parser.add_argument('--output-dir', '-o', default="C:/temp/landscapes", 
                       help='Output directory for generated scenes (default: C:/temp/landscapes)')
    parser.add_argument('--list-operators', '-l', action='store_true', 
                       help='List available landscape operators and exit')
    parser.add_argument('--list-addons', '-a', action='store_true', 
                       help='List available addons and exit')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    
    # Parse args from sys.argv, but handle Blender's arguments
    if '--' in sys.argv:
        script_args = sys.argv[sys.argv.index('--') + 1:]
    else:
        script_args = sys.argv[1:]
    
    return parser.parse_args(script_args)

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Handle listing options
        if args.list_addons:
            list_available_addons()
            sys.exit(0)
        
        if args.list_operators:
            list_available_landscape_operators()
            sys.exit(0)
        
        if args.verbose:
            print("Enabled Addons:")
            for addon in bpy.context.preferences.addons.keys():
                print(f"  {addon}")
            
            list_available_addons()
            list_available_landscape_operators()
        
        # Try to enable common landscape addons
        landscape_addons = [
            "add_mesh_extra_objects",
            "landscape_addon", 
            "add_landscape",
            "mesh_landscape"
        ]
        
        for addon in landscape_addons:
            if addon not in bpy.context.preferences.addons:
                try:
                    bpy.ops.preferences.addon_enable(module=addon)
                    print(f"✓ Enabled addon: {addon}")
                except:
                    if args.verbose:
                        print(f"Could not enable addon: {addon}")
        
        # Generate landscape scenes
        scenes = generate_landscape_scenes(args.output_dir)
        
        if scenes:
            print(f"✓ Successfully generated {len(scenes)} landscape scenes")
            sys.exit(0)
        else:
            print("✗ No scenes were generated successfully")
            sys.exit(1)
        
    except SystemExit:
        # Handle argparse help/error exits
        pass
    except Exception as e:
        print(f"✗ Script execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

