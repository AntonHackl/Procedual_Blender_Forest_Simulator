import bpy
import subprocess
import sys

def install_lidar_if_necessary():
    """Check if laspy is installed, install if not."""
    try:
        # Try to import laspy
        import laspy
        print("✓ laspy is already installed")
        return True
    except ImportError:
        print("✗ laspy is not installed")
        
        # Try to install it
        if install_package("laspy"):
            # Try importing again
            try:
                import laspy
                print("✓ laspy imported successfully after installation")
                return True
            except ImportError as e:
                print(f"✗ Still cannot import laspy after installation: {e}")
                return False
        else:
            return False
        
# def install_openexr_if_necessary():
#     """Check if openexr is installed, install if not."""
#     try:
#         import openexr
#         print("✓ openexr is already installed")
#         return True
#     except ImportError:
#         print("✗ openexr is not installed")
#         if install_package("openexr"):
#             import openexr
#             print("✓ openexr imported successfully after installation")
#             return True
#         else:
#             return False

def install_package(package_name):
    """Install a Python package using pip."""
    try:
        print(f"Attempting to install {package_name}...")
        
        # Try to install using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_name}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error installing {package_name}: {e}")
        return False

def check_available_backends():
    """Check and list all available scanning backends."""
    
    print("=== Checking Available Scanning Backends ===")
    
    # Enable the addon first
    try:
        bpy.ops.preferences.addon_enable(module="pointCloudRender")
        print("✓ pointCloudRender addon enabled successfully")
    except Exception as e:
        print(f"✗ Failed to enable addon: {e}")
        return
    
    # Get addon preferences
    try:
        preferences = bpy.context.preferences.addons["pointCloudRender"].preferences
        print("✓ Addon preferences accessed successfully")
    except Exception as e:
        print(f"✗ Failed to access preferences: {e}")
        return
    
    # Check available backend selector items
    try:
        from pointCloudRender.scanning_backends import available_scanning_backend_selector_items
        available_items = available_scanning_backend_selector_items()
        
        print(f"\nAvailable backends ({len(available_items)}):")
        for i, (backend_id, name, description, icon, id_num) in enumerate(available_items):
            print(f"  {i+1}. {backend_id} - {name}")
            print(f"     Description: {description}")
            print(f"     Icon: {icon}, ID: {id_num}")
            print()
    except Exception as e:
        print(f"✗ Failed to check available backends: {e}")
        return
    
    import importlib
    openexr_loader = importlib.find_loader('openexr_numpy')
    if openexr_loader is not None:
        print("✓ openexr_numpy is installed and available")
    else:
        print("✗ openexr_numpy is NOT installed")
        print("  Install with: pip install openexr_numpy")

def render_point_cloud():
    root_props = bpy.context.scene.pointCloudRenderProperties
    selected_index = root_props.selected_scanner
    scanner = root_props.laser_scanners[selected_index]
    
    print(f"Selected scanner: {scanner.name}")
    scanner.noise_generator.enabled = False
    scanner.noise_generator.standard_deviation = 0.0
    scanner.noise_generator.mean = 0.0

    bpy.ops.render.render_point_cloud()

def set_gpu_backend():
    """Set the scanning backend to GPU."""
    
    bpy.ops.preferences.addon_enable(module="pointCloudRender")
    
    preferences = bpy.context.preferences.addons["pointCloudRender"].preferences
    
    preferences.scanning_backend_type = "GPUScanningBackend"
    
    gpu_settings = preferences.GPUScanningBackendSettings
    gpu_settings.camera_type = "PanoramaGPUCamera"
    
    print(f"Backend set to: {preferences.scanning_backend_type}")
    print(f"Camera type: {gpu_settings.camera_type}")

def set_cpu_backend():
    """Set the scanning backend to CPU."""
    
    bpy.ops.preferences.addon_enable(module="pointCloudRender")
    
    preferences = bpy.context.preferences.addons["pointCloudRender"].preferences
    
    preferences.scanning_backend_type = "CPUScanningBackend"
    
    cpu_settings = preferences.CPUScanningBackendSettings
    # You can set the sampler type if you want, e.g.:
    # cpu_settings.sampler_type = "SceneBVHSampler"
    cpu_settings.sampler_type = "ObjectBVHSampler" 
    
    print(f"Backend set to: {preferences.scanning_backend_type}")
    print(f"Sampler type: {cpu_settings.sampler_type}")

def set_csv_writer():
    """Set the output format to CSV."""
    
    bpy.ops.preferences.addon_enable(module="pointCloudRender")
    
    preferences = bpy.context.preferences.addons["pointCloudRender"].preferences

    preferences.writer_type = "CSVSampleWriter"
    print(f"✓ Writer type set to: {preferences.writer_type}")
    return True

def set_las_writer():
    """Set the output format to LAS."""
    
    bpy.ops.preferences.addon_enable(module="pointCloudRender")
    
    preferences = bpy.context.preferences.addons["pointCloudRender"].preferences
    
    # Check if LAS writer is available
    try:
        from pointCloudRender.writers.las_sample_writer import LASSampleWriter
        if LASSampleWriter.available():
            # Set the writer type to LAS
            preferences.writer_type = "LASSampleWriter"
            print(f"✓ Writer type set to: {preferences.writer_type}")
            return True
        else:
            print("✗ LAS writer not available (laspy not installed)")
            return False
    except Exception as e:
        print(f"✗ Error setting LAS writer: {e}")
        return False

if __name__ == "__main__": 
    # List enabled addons
    print("Enabled Addons:")
    addonList = []
    for addon in bpy.context.preferences.addons.keys():
        print(f"  {addon}")
        addonList.append(addon)
    
    print(f'Is pointCloudRenderEnabled: {"pointCloudRenderer" in addonList}')
    if "pointCloudRenderer" not in addonList:
        # print("Installing pointCloudRenderer addon...")
        # addon_path = r"/root/media/data/pointCloudRender.zip"

        # bpy.ops.preferences.addon_install(filepath=addon_path)

        bpy.ops.preferences.addon_enable(module="pointCloudRender")
        print("Addon pointCloudRenderer installed and enabled.")

    # List operations under bpy.ops.render
    print("\nOperations under bpy.ops.render:")
    for op in dir(bpy.ops.render):
        if not op.startswith("_"):
            print(f"  {op}")

    # if not install_openexr_if_necessary():
    #     print("Failed to install openexr")
    #     exit()
    if not install_lidar_if_necessary():
        print("Failed to install lidar")
        exit()

    check_available_backends()
    # set_gpu_backend()
    set_cpu_backend()
    # if not set_las_writer():
    #     print("Failed to set LAS writer")
    #     exit()
    if not set_csv_writer():
        print("Failed to set CSV writer")
        exit()
    render_point_cloud()
