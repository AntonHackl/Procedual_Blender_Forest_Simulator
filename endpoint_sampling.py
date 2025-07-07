from typing import Any, Dict, List
import bpy
import random
import bmesh
from mathutils import Vector

def sample_mesh_group_surface_points(group_name, num_points, seed=0):
  """
  Generate points on the surface of all meshes in a group.
  Uses face area-weighted sampling for uniform distribution.
  
  Args:
      group_name: Name of the Blender collection/group
      num_points: Number of points to generate
      seed: Random seed for reproducibility
  
  Returns:
      List of Vector points on the mesh surface
  """
  
  random.seed(seed)
  
  if group_name not in bpy.data.collections:
      return []
  
  points: List[Vector] = []
  all_faces: List[Dict[str, Any]] = []
  total_area = 0.0

  temporary_meshes: List[bmesh.types.BMesh] = []

  # Collect all faces from all meshes in the group
  for obj in bpy.data.collections[group_name].objects:
    if obj.type != 'MESH':
      continue
          
    # Get the mesh in world space
    mesh = obj.data
    world_matrix = obj.matrix_world
    
    # Create bmesh for easier face access
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # bm.transform(world_matrix)
    
    # Collect faces with their areas
    for face in bm.faces:
      area = face.calc_area()
      all_faces.append({
        'face': face,
        'area': area,
        'object': obj
      })
      total_area += area

    temporary_meshes.append(bm)

  if total_area == 0:
      return []
  
  # Generate points using area-weighted sampling
  for _ in range(num_points):
    # Pick a random face based on area
    target_area = random.uniform(0, total_area)
    cumulative_area = 0.0
    selected_face_data = None
    
    for face_data in all_faces:
      cumulative_area += face_data['area']
      if cumulative_area >= target_area:
        selected_face_data = face_data
        break
    
    if selected_face_data:
      # Generate random point on the selected face
      point = generate_point_on_face(selected_face_data['face'])
      points.append(point)

  for bm in temporary_meshes:
      bm.free()

  return points

def generate_point_on_face(face):
  """
  Generate a random point on a triangular face using barycentric coordinates.
  
  Args:
      face: bmesh face (assumed to be triangular)
  
  Returns:
      Vector point on the face
  """
  # Get the three vertices of the face
  verts = list(face.verts)
  
  if len(verts) != 3:
    # Handle non-triangular faces by triangulating
    # For simplicity, we'll just use the first three vertices
    # In practice, you might want to properly triangulate
    verts = verts[:3]
  
  r1 = random.random()
  r2 = random.random()
  
  # Ensure r1 + r2 <= 1 (barycentric constraint)
  if r1 + r2 > 1:
    r1 = 1 - r1
    r2 = 1 - r2
  
  r3 = 1 - r1 - r2
  
  point = (r1 * verts[0].co + r2 * verts[1].co + r3 * verts[2].co)
  
  return Vector(point)
