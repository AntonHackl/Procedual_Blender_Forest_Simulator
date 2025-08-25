from dataclasses import dataclass, asdict
import numpy as np
from scipy.io import savemat
from typing import List, Tuple, Optional, Dict, Any
from mathutils import Vector
import os
import bpy
import matlab.engine

from .utils import create_inverse_graph
from .sca import SCA


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

def import_obj_to_blender(obj_path: str):
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

def execute_matlab_script(script_path: str, quit_after: bool = False):
    """
    Executes a MATLAB script using the singleton engine instance.
    
    :param script_path: Path to the MATLAB script to execute
    :param quit_after: Whether to quit the engine after execution (default: False)
    :return: True if successful, False otherwise
    """
    try:
        matlab_singleton = MatlabEngineProvider()
        eng = matlab_singleton.get_engine()
        
        script_dir = os.path.dirname(script_path)
        eng.addpath(script_dir, nargout=0)
        
        leafgen_src = os.path.join(os.path.dirname(__file__), 'leafgen', 'src')
        if os.path.exists(leafgen_src):
            eng.addpath(leafgen_src, nargout=0)
        
        print(f"Executing MATLAB script: {script_path}")
        
        eng.run(script_path, nargout=0)
        
        if quit_after:
            matlab_singleton.quit_engine()
        
        print(f"MATLAB script executed successfully: {script_path}")
        return True
        
    except Exception as e:
        print(f"Error executing MATLAB script: {e}")
        return False

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

        # Build MATLAB struct
        mpairs = []
        params = leaf_params or {}
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

        add_pair('pLADDh', params.get('pLADDh', [8, 3]))
        add_pair('pLADDd', params.get('pLADDd', [2.0, 1.5]))
        add_pair('fun_pLSD', params.get('fun_pLSD', [0.008, 0.00025**2]))
        add_pair('totalLeafArea', params.get('totalLeafArea', 20))

        leaf_params_struct = eng.feval('struct', *mpairs) if mpairs else eng.eval('struct()', nargout=1)

        # Call the MATLAB function
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
    # Ensure we save to the MATLAB script's expected folder
    script_dir = os.path.join(os.path.dirname(__file__), 'leafgen', 'src')
    mat_out = os.path.join(script_dir, 'example-data', 'generated_tree.mat')
    os.makedirs(os.path.dirname(mat_out), exist_ok=True)
    savemat(mat_out, {'qsm': {'cylinder': qsm_dict}})
    
    print(f"QSM saved to: {mat_path}")
    
    if execute_matlab:
        # Always use parameterized MATLAB function now
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