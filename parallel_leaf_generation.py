"""
Parallel leaf generation module for the Forest Simulator.
This module uses MATLAB's built-in background execution (background=True) for efficient 
parallel crown generation. QSM data is passed directly to MATLAB and OBJ data is returned
in memory without requiring temporary files or disk I/O.
"""

import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import matlab.engine
import numpy as np
from scipy.io import savemat
from mathutils import Vector

from .utils import create_inverse_graph
from .sca import SCA


@dataclass(frozen=True)
class QSM:
    start: np.ndarray
    axis: np.ndarray
    length: np.ndarray
    radius: np.ndarray
    parent: np.ndarray
    branch: np.ndarray


@dataclass(frozen=True)
class ConversionNode:
    sca_index: int
    qsm_parent: int
    qsm_branch: int


class MatlabEngineProvider:
    """
    Singleton class to manage a single MATLAB engine instance.
    """
    _instance: Optional['MatlabEngineProvider'] = None
    _engine: Optional[matlab.engine.MatlabEngine] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_engine(self) -> matlab.engine.MatlabEngine:
        """
        Returns the MATLAB engine instance, creating it if it doesn't exist.
        
        :return: MATLAB engine instance
        :raises: Exception if engine cannot be started
        """
        if self._engine is None:
            try:
                print("Starting MATLAB engine...")
                self._engine = matlab.engine.start_matlab()
                print("MATLAB engine started successfully")
            except Exception as e:
                print(f"Failed to start MATLAB engine: {e}")
                raise
        return self._engine
    
    def quit_engine(self):
        """
        Quits the MATLAB engine and resets the singleton state.
        """
        if self._engine is not None:
            try:
                self._engine.quit()
                print("MATLAB engine quit successfully")
            except Exception as e:
                print(f"Error quitting MATLAB engine: {e}")
            finally:
                self._engine = None
    
    def is_engine_running(self) -> bool:
        """
        Check if the MATLAB engine is currently running.
        
        :return: True if engine is running, False otherwise
        """
        return self._engine is not None


def convert_qsm_to_matlab_struct(engine: matlab.engine.MatlabEngine, qsm: QSM):
    """Convert QSM dataclass to MATLAB struct format."""
    cylinder_data = {
        'start': matlab.double(qsm.start.tolist()),
        'axis': matlab.double(qsm.axis.tolist()),
        'length': matlab.double(qsm.length.tolist()),
        'radius': matlab.double(qsm.radius.tolist()),
        'parent': matlab.double(qsm.parent.tolist()),
        'branch': matlab.double(qsm.branch.tolist())
    }
    
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
            try:
                arr = np.asarray(value).astype(float).reshape(1, -1)
                mpairs.extend([name, matlab.double(arr.tolist())])
            except Exception:
                pass
    
    add_pair('pLADDh', leaf_params.get('pLADDh', [8, 3]))
    add_pair('pLADDd', leaf_params.get('pLADDd', [2.0, 1.5]))
    add_pair('fun_pLSD', leaf_params.get('fun_pLSD', [0.008, 0.00025**2]))
    add_pair('totalLeafArea', leaf_params.get('totalLeafArea', 20))
    
    return engine.feval('struct', *mpairs) if mpairs else engine.eval('struct()', nargout=1)


def import_obj_to_blender(obj_path: str):
    import bpy
    try:
        if not os.path.exists(obj_path):
            print(f"OBJ file not found: {obj_path}")
            return []
        
        active_object = bpy.context.active_object
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='Y', up_axis='Z')
        foliage_obj = bpy.context.view_layer.objects.active
        foliage_obj.parent = active_object
        
        material_name = "Foliage_Green"
        if material_name not in bpy.data.materials:
            mat = bpy.data.materials.new(material_name)
            mat.diffuse_color = (0.1, 0.6, 0.1, 1.0)
        else:
            mat = bpy.data.materials[material_name]
            mat.diffuse_color = (0.1, 0.6, 0.1, 1.0)

        if foliage_obj.data.materials:
            foliage_obj.data.materials[0] = mat
        else:
            foliage_obj.data.materials.append(mat)
        
        bpy.context.view_layer.objects.active = active_object
        
        print(f"Successfully imported the foliage object from: {obj_path} with green material")
        return foliage_obj

    except Exception as e:
        print(f"Error importing OBJ file: {e}")
        return None


def execute_leaf_generation_with_params(leaf_params: Optional[Dict[str, Any]] = None, quit_after: bool = False) -> bool:
    """Execute MATLAB leaf generation using the parameterized function run_leaf_generation_with_params.
    Builds a MATLAB struct from the provided parameters and calls the function.

    Parameters expected (all optional, defaults applied in MATLAB code if omitted):
    - pLADDh: [alpha, beta]
    - pLADDd: [k, lambda]
    - fun_pLSD: [mu, sigma2]
    - totalLeafArea: float
    """
    try:
        matlab_singleton = MatlabEngineProvider()
        eng = matlab_singleton.get_engine()

        leafgen_src = os.path.join(os.path.dirname(__file__), 'leafgen', 'src')
        if os.path.exists(leafgen_src):
            eng.addpath(leafgen_src, nargout=0)

        params = leaf_params or {}
        leaf_params_struct = create_leaf_params_struct(eng, params)

        print("Executing MATLAB function: run_leaf_generation_with_params")
        eng.run_leaf_generation_with_params(leaf_params_struct, nargout=0)

        if quit_after:
            matlab_singleton.quit_engine()

        print("MATLAB parameterized leaf generation finished")
        return True

    except Exception as e:
        print(f"Error executing parameterized MATLAB leaf generation: {e}")
        return False


def generate_foliage(
    qsm: QSM,
    mat_path: str,
    execute_matlab: bool = False,
    matlab_script_path: str | None = None,
    import_result: bool = True,
    leaf_params: Optional[Dict[str, Any]] = None,
):
    qsm_dict = asdict(qsm)
    for key, value in qsm_dict.items():
        arr = np.asarray(value)
        if arr.ndim == 1:
            qsm_dict[key] = arr.reshape(-1, 1)
    script_dir = os.path.join(os.path.dirname(__file__), 'leafgen', 'src')
    mat_out = os.path.join(script_dir, 'example-data', 'generated_tree.mat')
    os.makedirs(os.path.dirname(mat_out), exist_ok=True)
    savemat(mat_out, {'qsm': {'cylinder': qsm_dict}})
    
    print(f"QSM saved to: {mat_path}")
    
    if execute_matlab:
        success = execute_leaf_generation_with_params(leaf_params)
        if success and import_result:
            obj_path = os.path.join(os.path.dirname(__file__), 'leafgen', 'src', 'example-data', 'leaves_export.obj')
            import_obj_to_blender(obj_path)


def convert_sca_skeleton_to_qsm(sca_tree: SCA, radii: np.ndarray):
    branchpoints = sca_tree.branchpoints

    start: list[Vector] = []
    axis: list[Vector] = []
    length: list[float] = []
    radius: list[float] = []
    parent: list[int] = []
    branch: list[int] = []

    active_list: List[ConversionNode] = [ConversionNode(0, 0, 1)]

    inverse_graph = create_inverse_graph(branchpoints)
    while len(active_list) > 0:
        current_node = active_list.pop(0)
        current_branchpoint = branchpoints[current_node.sca_index]
        current_position = current_branchpoint.v
        
        children = inverse_graph[current_node.sca_index]
        branch_index = current_node.qsm_branch
        qsm_parent_index = len(start)
        for sca_child_index in children:
            start.append(current_branchpoint.v)
            
            child_position = branchpoints[sca_child_index].v
            current_to_child = child_position - current_position
            axis.append(current_to_child.normalized())
            length.append(current_to_child.length)
            radius.append(radii[sca_child_index])
            parent.append(qsm_parent_index)
            branch.append(branch_index)

            branch_index += 1 if len(children) > 1 else 0
            active_list.append(ConversionNode(
                sca_index=sca_child_index,
                qsm_parent=len(start),
                qsm_branch=branch_index,
            ))

    start_arr = np.array(start)
    axis_arr = np.array(axis)
    length_arr = np.array(length).reshape(-1, 1)
    radius_arr = np.array(radius).reshape(-1, 1)
    parent_arr = np.array(parent).reshape(-1, 1)
    branch_arr = np.array(branch).reshape(-1, 1)

    return QSM(start_arr, axis_arr, length_arr, radius_arr, parent_arr, branch_arr)


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


def parse_obj_string(obj_string: str) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    """Parse OBJ string data and return vertices and faces."""
    vertices = []
    faces = []
    
    for line in obj_string.strip().split('\n'):
        line = line.strip()
        if line.startswith('v '):
            # Parse vertex: "v x y z"
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append((x, y, z))
                except ValueError:
                    continue
        elif line.startswith('f '):
            # Parse face: "f v1 v2 v3" (OBJ uses 1-based indexing)
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Convert to 0-based indexing for Blender
                    face_indices = []
                    for i in range(1, len(parts)):
                        # Handle "v/vt/vn" format by taking only vertex index
                        vertex_data = parts[i].split('/')[0]
                        face_indices.append(int(vertex_data) - 1)
                    
                    # For triangles, add directly
                    if len(face_indices) == 3:
                        faces.append(tuple(face_indices))
                    # For quads, split into two triangles
                    elif len(face_indices) == 4:
                        faces.append((face_indices[0], face_indices[1], face_indices[2]))
                        faces.append((face_indices[0], face_indices[2], face_indices[3]))
                except (ValueError, IndexError):
                    continue
    
    return vertices, faces


def create_blender_mesh_from_obj_data(obj_string: str, tree_id: int, tree_location: Tuple[float, float, float]) -> Optional[object]:
    """Create a Blender mesh object directly from OBJ string data at the specified location."""
    import bpy
    from mathutils import Vector
    
    try:
        # Parse OBJ data
        vertices, faces = parse_obj_string(obj_string)
        
        if not vertices or not faces:
            print(f"No valid geometry found in OBJ data for tree {tree_id}")
            return None
        
        # Store original active object
        original_active = bpy.context.view_layer.objects.active
        
        # Find the tree object to set as active (like the original import does)
        tree_obj_name = f"Tree_{tree_id}"
        tree_obj = bpy.data.objects.get(tree_obj_name)
        
        if tree_obj:
            # Set tree as active object (like original import_obj_to_blender)
            bpy.context.view_layer.objects.active = tree_obj
            
            # Set cursor to tree location 
            original_cursor = bpy.context.scene.cursor.location.copy()
            bpy.context.scene.cursor.location = Vector(tree_location)
            bpy.context.view_layer.update()
        
        # Create new mesh
        mesh_name = f"leaves_tree_{tree_id}"
        mesh = bpy.data.meshes.new(mesh_name)
        
        # Convert vertices to Vector objects
        verts = [Vector(v) for v in vertices]
        
        # Create mesh from vertices and faces
        mesh.from_pydata(verts, [], faces)
        mesh.update(calc_edges=True)
        
        # Create object from mesh 
        obj_name = f"leaves_export_tree_{tree_id}"
        obj = bpy.data.objects.new(obj_name, mesh)
        
        # Add to scene
        bpy.context.collection.objects.link(obj)
        
        # Set as active object (like after OBJ import)
        bpy.context.view_layer.objects.active = obj
        
        if tree_obj:
            # Parent to tree (like original import_obj_to_blender)
            obj.parent = tree_obj
            
            # Restore original cursor
            bpy.context.scene.cursor.location = original_cursor
            bpy.context.view_layer.update()
            
            # Restore original active object
            bpy.context.view_layer.objects.active = original_active
        
        # Create and assign green material
        material_name = "Foliage_Green"
        if material_name not in bpy.data.materials:
            mat = bpy.data.materials.new(material_name)
            mat.diffuse_color = (0.1, 0.6, 0.1, 1.0)
        else:
            mat = bpy.data.materials[material_name]
        
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        
        print(f"Created Blender mesh for tree {tree_id} at {tree_location}: {len(vertices)} vertices, {len(faces)} faces")
        return obj
        
    except Exception as e:
        print(f"Error creating Blender mesh for tree {tree_id}: {e}")
        return None


class ParallelLeafGenerator:
    """
    Manages parallel leaf generation for multiple trees using MATLAB background execution.
    No temporary directories or disk I/O required - all data passed in memory.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.engines: List[matlab.engine.MatlabEngine] = []
        self.struct_engine: Optional[matlab.engine.MatlabEngine] = None
        
    def __enter__(self):
        # Start one dedicated engine for struct creation (non-blocking operations)
        try:
            print("Starting dedicated MATLAB engine for struct creation...")
            self.struct_engine = matlab.engine.start_matlab()
            leafgen_src = os.path.join(os.path.dirname(__file__), 'leafgen', 'src')
            if os.path.exists(leafgen_src):
                self.struct_engine.addpath(leafgen_src, nargout=0)
            print("Struct engine started successfully")
        except Exception as e:
            print(f"Failed to start struct engine: {e}")
            self.struct_engine = None
        
        # Start MATLAB engines for background tasks
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
        
        print(f"Successfully started {len(self.engines)} MATLAB engines + 1 struct engine")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Quit struct engine
        if self.struct_engine:
            try:
                self.struct_engine.quit()
                print("Quit struct engine")
            except Exception as e:
                print(f"Error quitting struct engine: {e}")
        
        # Quit all MATLAB engines
        for i, engine in enumerate(self.engines):
            try:
                engine.quit()
                print(f"Quit MATLAB engine {i+1}")
            except Exception as e:
                print(f"Error quitting MATLAB engine {i+1}: {e}")
    
    def generate_leaves_parallel(self, tasks: List[ParallelLeafTask]) -> Dict[int, Optional[object]]:
        """
        Generate leaves for multiple trees in parallel using MATLAB background execution.
        QSM data is passed directly to MATLAB and OBJ data is returned in memory.
        Creates Blender mesh objects directly from OBJ string data.
        Returns a dictionary mapping tree_id to Blender object (or None if failed).
        """
        if not tasks:
            return {}
        
        if not self.engines:
            print("No MATLAB engines available for parallel processing")
            return {task.tree_id: None for task in tasks}
        
        if not self.struct_engine:
            print("No struct engine available - falling back to blocking struct creation")
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
                # Convert QSM to MATLAB struct format using dedicated struct engine
                qsm_struct = convert_qsm_to_matlab_struct(self.struct_engine, task.qsm)
                
                # Create leaf parameters struct using dedicated struct engine
                leaf_params_struct = create_leaf_params_struct(self.struct_engine, task.leaf_params)
                
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
                    # Get tree location for this task
                    tree_location = None
                    for task in tasks:
                        if task.tree_id == bg_task.task_id:
                            tree_location = task.tree_location
                            break
                    
                    if tree_location:
                        # Create Blender mesh object directly from OBJ string data at correct location
                        blender_obj = create_blender_mesh_from_obj_data(obj_string, bg_task.task_id, tree_location)
                        if blender_obj:
                            results[bg_task.task_id] = blender_obj
                            print(f"✓ Tree {bg_task.task_id} leaf generation completed successfully in {elapsed:.1f}s ({len(obj_string)} chars)")
                        else:
                            results[bg_task.task_id] = None
                            print(f"✗ Tree {bg_task.task_id} failed to create Blender mesh after {elapsed:.1f}s")
                    else:
                        results[bg_task.task_id] = None
                        print(f"✗ Tree {bg_task.task_id} no location found")
                else:
                    results[bg_task.task_id] = None
                    print(f"✗ Tree {bg_task.task_id} returned empty OBJ data after {elapsed:.1f}s")
                
            except Exception as e:
                print(f"✗ Tree {bg_task.task_id} failed: {e}")
                results[bg_task.task_id] = None
        
        success_count = sum(1 for obj in results.values() if obj is not None)
        print(f"Parallel leaf generation completed: {success_count}/{len(tasks)} trees successful")
        
        return results


def finalize_parallel_leaf_results(results: Dict[int, Optional[object]], tree_locations: Dict[int, Tuple[float, float, float]]):
    """
    Finalize the generated leaf objects.
    The objects have already been created, positioned, and parented during mesh creation.
    This function mainly reports the final status.
    """
    success_count = sum(1 for obj in results.values() if obj is not None)
    print(f"Finalized leaves for {success_count} trees in Blender - all objects already positioned and parented")
