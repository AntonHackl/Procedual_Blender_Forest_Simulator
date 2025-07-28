from dataclasses import dataclass, asdict
import numpy as np
from scipy.io import savemat
from typing import List, Tuple
from mathutils import Vector
import os
import bpy
import matlab.engine

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

def import_obj_to_blender(obj_path: str, collection_name: str = "Generated_Leaves"):
    try:
        if not os.path.exists(obj_path):
            print(f"OBJ file not found: {obj_path}")
            return []
        
        if collection_name not in bpy.data.collections:
            collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(collection)
        else:
            collection = bpy.data.collections[collection_name]
        
        active_object = bpy.context.active_object
        bpy.ops.wm.obj_import(filepath=obj_path)
        bpy.context.view_layer.objects.active = active_object
        
        imported_objects = []
        for obj in bpy.context.selected_objects:
            bpy.context.scene.collection.objects.unlink(obj)
            collection.objects.link(obj)
            imported_objects.append(obj)
        
        print(f"Successfully imported {len(imported_objects)} objects from: {obj_path}")
        return imported_objects
        
    except Exception as e:
        print(f"Error importing OBJ file: {e}")
        return []

def execute_matlab_script(script_path: str):
    try:
        
        print("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        
        script_dir = os.path.dirname(script_path)
        eng.addpath(script_dir, nargout=0)
        
        leafgen_src = os.path.join(os.path.dirname(__file__), 'leafgen', 'src')
        if os.path.exists(leafgen_src):
            eng.addpath(leafgen_src, nargout=0)
        
        print(f"Executing MATLAB script: {script_path}")
        
        eng.run(script_path, nargout=0)
        
        eng.quit()
        
        print(f"MATLAB script executed successfully: {script_path}")
        return True
        
    except Exception as e:
        print(f"Error executing MATLAB script: {e}")
        return False

def generate_foliage(qsm: QSM, mat_path: str, execute_matlab: bool = False, matlab_script_path: str = None, import_result: bool = True):
    qsm_dict = asdict(qsm)
    for key, value in qsm_dict.items():
        arr = np.asarray(value)
        if arr.ndim == 1:
            qsm_dict[key] = arr.reshape(-1, 1)
    savemat(mat_path, {'qsm': {'cylinder': qsm_dict}})
    
    print(f"QSM saved to: {mat_path}")
    
    if execute_matlab:
        if matlab_script_path is None:
            script_dir = os.path.join(os.path.dirname(__file__), 'leafgen', 'src')
            matlab_script_path = os.path.join(script_dir, 'main_qsm_direct.m')
        
        if os.path.exists(matlab_script_path):
            # success = execute_matlab_script(matlab_script_path)
            success = True
            if success and import_result:
                obj_path = os.path.join(os.path.dirname(__file__), 'leafgen', 'src', 'leaves_export.obj')
                import_obj_to_blender(obj_path)
        else:
            print(f"MATLAB script not found: {matlab_script_path}")

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