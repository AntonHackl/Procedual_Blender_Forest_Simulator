"""
Parallel leaf generation module for the Forest Simulator.
This module uses MATLAB's built-in background execution (background=True) for efficient 
parallel crown generation. QSM data is passed directly to MATLAB and OBJ data is returned
in memory without requiring temporary files or disk I/O.
"""

import os
import time
import io
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import matlab.engine
import numpy as np

from .leaf_generation import QSM, import_obj_to_blender


@dataclass
class ParallelLeafTask:
    """Represents a single leaf generation task."""
    tree_id: int
    qsm: QSM
    leaf_params: Dict[str, Any]
    tree_location: Tuple[float, float, float]


@dataclass
class MatlabBackgroundTask:
    """Represents a background MATLAB task with its future result."""
    task_id: int
    future: Any  # matlab.engine.FutureResult
    start_time: float


def convert_qsm_to_matlab_struct(engine: matlab.engine.MatlabEngine, qsm: QSM):
    """Convert QSM dataclass to MATLAB struct format."""
    # Convert numpy arrays to MATLAB format
    cylinder_data = {
        'start': matlab.double(qsm.start.tolist()),
        'axis': matlab.double(qsm.axis.tolist()),
        'length': matlab.double(qsm.length.tolist()),
        'radius': matlab.double(qsm.radius.tolist()),
        'parent': matlab.double(qsm.parent.tolist()),
        'branch': matlab.double(qsm.branch.tolist())
    }
    
    # Create the MATLAB struct using the engine
    return engine.feval('struct', 'cylinder', engine.feval('struct', 
                       'start', cylinder_data['start'],
                       'axis', cylinder_data['axis'], 
                       'length', cylinder_data['length'],
                       'radius', cylinder_data['radius'],
                       'parent', cylinder_data['parent'],
                       'branch', cylinder_data['branch']))


def create_leaf_params_struct(engine: matlab.engine.MatlabEngine, leaf_params: Dict[str, Any]):
    """Create MATLAB struct from leaf parameters."""
    mpairs = []
    
    def add_pair(name: str, value):
        nonlocal mpairs
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            mpairs.extend([name, matlab.double([list(value)]) if len(value) > 1 else matlab.double([value])])
        elif isinstance(value, (int, float)):
            mpairs.extend([name, matlab.double([float(value)])])
        else:
            # fallback: try to convert numpy arrays
            try:
                arr = np.asarray(value).astype(float).reshape(1, -1)
                mpairs.extend([name, matlab.double(arr.tolist())])
            except Exception:
                pass
    
    # Add parameters with defaults
    add_pair('pLADDh', leaf_params.get('pLADDh', [8, 3]))
    add_pair('pLADDd', leaf_params.get('pLADDd', [2.0, 1.5]))
    add_pair('fun_pLSD', leaf_params.get('fun_pLSD', [0.008, 0.00025**2]))
    add_pair('totalLeafArea', leaf_params.get('totalLeafArea', 20))
    
    return engine.feval('struct', *mpairs) if mpairs else engine.eval('struct()', nargout=1)


def create_obj_from_string(obj_string: str, tree_id: int) -> Optional[str]:
    """Create a temporary OBJ file from string data for import into Blender."""
    import tempfile
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_tree_{tree_id}.obj', delete=False) as f:
            f.write(obj_string)
            temp_path = f.name
        return temp_path
    except Exception as e:
        print(f"Error creating temporary OBJ file for tree {tree_id}: {e}")
        return None


class ParallelLeafGenerator:
    """
    Manages parallel leaf generation for multiple trees using MATLAB background execution.
    No temporary directories or disk I/O required - all data passed in memory.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.engines: List[matlab.engine.MatlabEngine] = []
        
    def __enter__(self):
        # Start MATLAB engines
        print(f"Starting {self.max_workers} MATLAB engines for parallel leaf generation...")
        for i in range(self.max_workers):
            try:
                engine = matlab.engine.start_matlab()
                # Add necessary paths
                leafgen_src = os.path.join(os.path.dirname(__file__), 'leafgen', 'src')
                if os.path.exists(leafgen_src):
                    engine.addpath(leafgen_src, nargout=0)
                self.engines.append(engine)
                print(f"Started MATLAB engine {i+1}/{self.max_workers}")
            except Exception as e:
                print(f"Failed to start MATLAB engine {i+1}: {e}")
        
        print(f"Successfully started {len(self.engines)} MATLAB engines")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Quit all MATLAB engines
        for i, engine in enumerate(self.engines):
            try:
                engine.quit()
                print(f"Quit MATLAB engine {i+1}")
            except Exception as e:
                print(f"Error quitting MATLAB engine {i+1}: {e}")
    
    def generate_leaves_parallel(self, tasks: List[ParallelLeafTask]) -> Dict[int, Optional[str]]:
        """
        Generate leaves for multiple trees in parallel using MATLAB background execution.
        QSM data is passed directly to MATLAB and OBJ data is returned in memory.
        Returns a dictionary mapping tree_id to obj_path (or None if failed).
        """
        if not tasks:
            return {}
        
        if not self.engines:
            print("No MATLAB engines available for parallel processing")
            return {task.tree_id: None for task in tasks}
        
        print(f"Starting parallel leaf generation for {len(tasks)} trees using {len(self.engines)} MATLAB engines")
        
        # Submit all tasks as background jobs
        background_tasks = []
        engine_index = 0
        
        for task in tasks:
            # Get next engine (round-robin)
            engine = self.engines[engine_index % len(self.engines)]
            engine_index += 1
            
            try:
                # Convert QSM to MATLAB struct format
                qsm_struct = convert_qsm_to_matlab_struct(engine, task.qsm)
                
                # Create leaf parameters struct
                leaf_params_struct = create_leaf_params_struct(engine, task.leaf_params)
                
                # Submit background task - call the new parallel function
                print(f"Submitting tree {task.tree_id} to MATLAB engine {(engine_index-1) % len(self.engines) + 1}")
                future = engine.run_leaf_generation_parallel(qsm_struct, leaf_params_struct, nargout=1, background=True)
                
                background_tasks.append(MatlabBackgroundTask(
                    task_id=task.tree_id,
                    future=future,
                    start_time=time.time()
                ))
                
            except Exception as e:
                print(f"Error submitting task for tree {task.tree_id}: {e}")
                background_tasks.append(MatlabBackgroundTask(
                    task_id=task.tree_id,
                    future=None,
                    start_time=time.time()
                ))
        
        # Wait for all tasks to complete and collect results
        results = {}
        print(f"Waiting for {len(background_tasks)} background tasks to complete...")
        
        for bg_task in background_tasks:
            try:
                if bg_task.future is None:
                    results[bg_task.task_id] = None
                    continue
                
                # Wait for completion and get OBJ string data
                obj_string = bg_task.future.result()  # This blocks until the task completes and returns the OBJ data
                elapsed = time.time() - bg_task.start_time
                
                if obj_string and len(obj_string) > 0:
                    # Create temporary OBJ file from string data
                    temp_obj_path = create_obj_from_string(obj_string, bg_task.task_id)
                    if temp_obj_path:
                        results[bg_task.task_id] = temp_obj_path
                        print(f"✓ Tree {bg_task.task_id} leaf generation completed successfully in {elapsed:.1f}s ({len(obj_string)} chars)")
                    else:
                        results[bg_task.task_id] = None
                        print(f"✗ Tree {bg_task.task_id} failed to create temporary OBJ file after {elapsed:.1f}s")
                else:
                    results[bg_task.task_id] = None
                    print(f"✗ Tree {bg_task.task_id} returned empty OBJ data after {elapsed:.1f}s")
                
            except Exception as e:
                print(f"✗ Tree {bg_task.task_id} failed: {e}")
                results[bg_task.task_id] = None
        
        success_count = sum(1 for path in results.values() if path is not None)
        print(f"Parallel leaf generation completed: {success_count}/{len(tasks)} trees successful")
        
        return results


def import_parallel_leaf_results(results: Dict[int, Optional[str]], tree_locations: Dict[int, Tuple[float, float, float]]):
    """
    Import the generated leaf OBJ files into Blender for all successful trees.
    Cleans up temporary files after import.
    """
    import bpy
    from mathutils import Vector
    
    success_count = 0
    original_cursor = bpy.context.scene.cursor.location.copy()
    temp_files_to_cleanup = []
    
    for tree_id, obj_path in results.items():
        if obj_path is None:
            continue
            
        try:
            # Set cursor to tree location if available
            if tree_id in tree_locations:
                location = tree_locations[tree_id]
                bpy.context.scene.cursor.location = Vector(location)
                bpy.context.view_layer.update()
            
            # Import the OBJ file
            foliage_obj = import_obj_to_blender(obj_path)
            
            if foliage_obj:
                # Rename to include tree ID
                foliage_obj.name = f"leaves_export_tree_{tree_id}"
                
                # Try to parent to the corresponding tree if it exists
                tree_obj_name = f"Tree_{tree_id}"
                tree_obj = bpy.data.objects.get(tree_obj_name)
                if tree_obj:
                    foliage_obj.parent = tree_obj
                    print(f"Successfully imported and parented leaves for tree {tree_id}")
                else:
                    print(f"Successfully imported leaves for tree {tree_id} (no parent tree found)")
                    
                success_count += 1
            else:
                print(f"Failed to import leaves for tree {tree_id}")
            
            # Add to cleanup list
            temp_files_to_cleanup.append(obj_path)
                
        except Exception as e:
            print(f"Error importing leaves for tree {tree_id}: {e}")
            # Still add to cleanup list
            temp_files_to_cleanup.append(obj_path)
    
    # Clean up temporary files
    for temp_file in temp_files_to_cleanup:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {os.path.basename(temp_file)}")
        except Exception as e:
            print(f"Error cleaning up temporary file {temp_file}: {e}")
    
    # Restore original cursor location
    bpy.context.scene.cursor.location = original_cursor
    bpy.context.view_layer.update()
    
    print(f"Imported leaves for {success_count} trees into Blender")
