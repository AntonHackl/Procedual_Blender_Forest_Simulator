import bpy
import numpy as np
import bmesh
import random
from skimage.measure import marching_cubes
from typing import Tuple, List, Dict, Set

class VoxelGrid:
  def __init__(self):
    self.grid = np.zeros((50, 50, 50), dtype=int)
    
    # for _ in range(int(20*20*20*0.8)):
    #    self.grid[random.randint(0, 19)][random.randint(0, 19)][random.randint(0, 19)] = 1

    # for x in range(3, 15):
    #     for y in range(3, 15):
    #         for z in range(3, 15):
    #           self.grid[x][y][z] = 1
    # self.grid[2][2][3] = 1
    # self.grid[2][2][4] = 1
    # self.grid[1][2][3] = 1
    # self.grid[3][2][3] = 1

    self.cube_size = 0.5

  def generate_mesh(self, index):
    mesh = bpy.data.meshes.new("VoxelMesh")
    obj = bpy.data.objects.new("VoxelObject", mesh)

    # verts, faces, _, _ = marching_cubes(self.grid, level=0.9999)    

    # mesh.from_pydata(verts.tolist(), [], faces.tolist())
    # mesh.update()

    # Prepare bmesh for geometry creation
    bm = bmesh.new()

    for x in range(len(self.grid)):
        for y in range(len(self.grid[x])):
            for z in range(len(self.grid[x][y])):
                if self.grid[x][y][z] == index:
                    self.add_voxel_to_bmesh(bm, x, y, z, index, self.cube_size)

    bm.to_mesh(mesh)
    bm.free()

    return obj
  
  def add_voxel_to_bmesh(self, bm, x, y, z, index, size):
    voxel_pos = (x * size, y * size, z * size)
    
    # Check neighbors and add faces only where necessary
    if not self.is_filled(x-1, y, z, index):
        self.add_face_to_bmesh(bm, voxel_pos, size, "left")
    if not self.is_filled(x+1, y, z, index): 
        self.add_face_to_bmesh(bm, voxel_pos, size, "right")
    if not self.is_filled(x, y-1, z, index): 
        self.add_face_to_bmesh(bm, voxel_pos, size, "front")
    if not self.is_filled(x, y+1, z, index): 
        self.add_face_to_bmesh(bm, voxel_pos, size, "back")
    if not self.is_filled(x, y, z-1, index): 
        self.add_face_to_bmesh(bm, voxel_pos, size, "bottom")
    if not self.is_filled(x, y, z+1, index): 
        self.add_face_to_bmesh(bm, voxel_pos, size, "top")

  def add_face_to_bmesh(self, bm, position, size, face):
    x, y, z = position
    hs = size / 2.0  # Half-size of the cube
    
    # Define cube vertices
    if face == "left":
        verts = [(x - hs, y - hs, z - hs), (x - hs, y - hs, z + hs), (x - hs, y + hs, z + hs), (x - hs, y + hs, z - hs)]
    elif face == "right":
        verts = [(x + hs, y - hs, z - hs), (x + hs, y - hs, z + hs), (x + hs, y + hs, z + hs), (x + hs, y + hs, z - hs)]
    elif face == "front":
        verts = [(x - hs, y - hs, z - hs), (x + hs, y - hs, z - hs), (x + hs, y - hs, z + hs), (x - hs, y - hs, z + hs)]
    elif face == "back":
        verts = [(x - hs, y + hs, z - hs), (x + hs, y + hs, z - hs), (x + hs, y + hs, z + hs), (x - hs, y + hs, z + hs)]
    elif face == "bottom":
        verts = [(x - hs, y - hs, z - hs), (x + hs, y - hs, z - hs), (x + hs, y + hs, z - hs), (x - hs, y + hs, z - hs)]
    elif face == "top":
        verts = [(x - hs, y - hs, z + hs), (x + hs, y - hs, z + hs), (x + hs, y + hs, z + hs), (x - hs, y + hs, z + hs)]

    bm_verts = [bm.verts.new(v) for v in verts]
    bm.faces.new(bm_verts)
  
  def is_filled(self, x, y, z, index):
    if not (0 <= x < len(self.grid) and 0 <= y < len(self.grid[x]) and 0 <= z < len(self.grid[x][y])):
      return False 
    return self.grid[x][y][z] == index
  
  def add_tree(self, position: Tuple[int, int, int], stem_diameter: float, stem_height: float, crown_diameter: float):
    x, y, z = position
    x = int(x / self.cube_size)
    y = int(y / self.cube_size)
    z = int(z / self.cube_size)
    stem_radius = int(stem_diameter / 2 / self.cube_size)
    crown_radius = int(crown_diameter / 2 / self.cube_size)

    #Add stem
    stem_height_range = np.arange(int(stem_height / self.cube_size))
    stem_diameter_range = np.arange(-int(stem_diameter / self.cube_size), int(stem_diameter / self.cube_size))
    j, k = np.meshgrid(stem_diameter_range, stem_diameter_range, indexing='ij')
    mask = j**2 + k**2 <= stem_radius**2

    for i in stem_height_range:
        self.grid[x + j[mask], y + k[mask], z + i] = 1

    # Add crown
    crown_range = np.arange(-int(crown_diameter / self.cube_size), int(crown_diameter / self.cube_size))
    i, j, k = np.meshgrid(crown_range, crown_range, crown_range, indexing='ij')
    mask = i**2 + j**2 + k**2 <= crown_radius**2

    self.grid[x + j[mask], y + k[mask], z + i[mask] + int((stem_height+crown_diameter/2) / self.cube_size)] = 1
    # # Add stem
    # for i in range(-int(stem_height / self.cube_size), int(stem_height / self.cube_size)):
    #   for j in range(-int(stem_diameter / self.cube_size), int(stem_diameter / self.cube_size)):
    #     for k in range(-int(stem_diameter / self.cube_size), int(stem_diameter / self.cube_size)):
    #         if j*j + k*k <= stem_radius*stem_radius:
    #             self.grid[x + j][y + k][z + i] = 1
    
    

    # # Add crown
    # for i in range(-int(crown_diameter / self.cube_size), int(crown_diameter / self.cube_size)):
    #   for j in range(-int(crown_diameter / self.cube_size), int(crown_diameter / self.cube_size)):
    #     for k in range(-int(crown_diameter / self.cube_size), int(crown_diameter / self.cube_size)):
    #       if i*i+j*j+k*k <= crown_radius*crown_radius*crown_radius:
    #         self.grid[x + j][y + k][z + i + int(stem_height + crown_radius / self.cube_size)] = 1
  
  def greedy_meshing(self, index: int):
    quads = self.capture_quads(index)
    
    mesh = bpy.data.meshes.new("VoxelMesh")
    obj = bpy.data.objects.new("VoxelObject", mesh)

    # Prepare bmesh for geometry creation
    bm = bmesh.new()
    
    for quad in quads:
      x_start, y_start, z_start, x_end, y_end, z_end = quad
      x_end += 1
      y_end += 1
      z_end += 1 
      
      verts = [
        (x_start * self.cube_size, y_start * self.cube_size, z_start * self.cube_size),
        (x_end * self.cube_size, y_start * self.cube_size, z_start * self.cube_size),
        (x_end * self.cube_size, y_end * self.cube_size, z_start * self.cube_size),
        (x_start * self.cube_size, y_end * self.cube_size, z_start * self.cube_size),
        (x_start * self.cube_size, y_start * self.cube_size, z_end * self.cube_size),
        (x_end * self.cube_size, y_start * self.cube_size, z_end * self.cube_size),
        (x_end * self.cube_size, y_end * self.cube_size, z_end * self.cube_size),
        (x_start * self.cube_size, y_end * self.cube_size, z_end * self.cube_size)
      ]
      bm_verts = [bm.verts.new(v) for v in verts]

      # Create faces for the quad
      bm.faces.new([bm_verts[i] for i in [0, 1, 2, 3]])  # Bottom face
      bm.faces.new([bm_verts[i] for i in [4, 5, 6, 7]])  # Top face
      bm.faces.new([bm_verts[i] for i in [0, 1, 5, 4]])  # Front face
      bm.faces.new([bm_verts[i] for i in [2, 3, 7, 6]])  # Back face
      bm.faces.new([bm_verts[i] for i in [0, 3, 7, 4]])  # Left face
      bm.faces.new([bm_verts[i] for i in [1, 2, 6, 5]])  # Right face
    
    bm.to_mesh(mesh)
    bm.free()  
    return obj
  
  def capture_quads(self, index: int):
    instance_matrix = np.copy(self.grid)
    
    instance_matrix[instance_matrix != index] = 0
    
    planes = self.capture_planes(instance_matrix)
    
    quads: List[Tuple[int, int, int, int, int, int]] = []
    while len(planes) > 0:
      z_position, plane_set = next(iter(planes.items()))
      x_start, y_start, x_end, y_end = next(iter(plane_set))
      quads.append(self.capture_quad(z_position, x_start, y_start, x_end, y_end, planes))
      
    return quads
  
  def capture_quad(self, z_position: int, x_start: int, y_start: int, x_end: int, y_end: int, planes: Dict[int, Set[Tuple[int, int, int, int]]]):
    offset_minus = 0
    while self.plane_matches_segment_length(z_position + offset_minus, x_start, y_start, y_end, planes): 
      offset_minus -= 1
    offset_minus += 1
    
    offset_plus = 1
    while self.plane_matches_segment_length(z_position + offset_plus, x_start, y_start, y_end, planes): 
      offset_plus += 1
    offset_plus -= 1
    
    return x_start, y_start, z_position + offset_minus, x_end, y_end, z_position + offset_plus
  
  def plane_matches_segment_length(self, z_position: int, x_start: int, y_start: int, y_end: int, planes: Dict[int, Set[Tuple[int, int, int, int]]]):
    if z_position not in planes:
      return False 
    segments = planes[z_position]
    
    for (seg_x_start, seg_y_start, seg_x_end, seg_y_end) in segments:
      if seg_x_start == x_start and seg_y_start == y_start and seg_y_end == y_end:
        planes[z_position].remove((seg_x_start, seg_y_start, seg_x_end, seg_y_end))
        if (len(planes[z_position]) == 0):
          del planes[z_position]
        return True
    
    return False
  
  def capture_planes(self, instance_matrix: np.array):
    rows = self.capture_rows(instance_matrix)
    
    planes: Dict[int, Set[Tuple[int, int, int, int]]] = {}
    while len(rows) > 0: 
      (y_position, z_position), row_set = next(iter(rows.items()))
      x_start, x_end = next(iter(row_set))
      plane = self.capture_plane(y_position, z_position, x_start, x_end, rows)
      if z_position in planes:
        planes[z_position].add(plane)
      else:
        planes[z_position] = {plane}
        
    return planes

  def capture_plane(self, y_position: int, z_position: int, x_start: int, x_end: int, rows: Dict[Tuple[int, int], Set[Tuple[int, int]]]):
    # start with zero so the original row gets deleted as well.
    offset_minus = 0
    while self.row_matches_segment_length(y_position + offset_minus, z_position, x_start, x_end, rows): 
      offset_minus -= 1
    offset_minus += 1
    if offset_minus != 0:
      print('yea')
    offset_plus = 1
    while self.row_matches_segment_length(y_position + offset_plus, z_position, x_start, x_end, rows): 
      offset_plus += 1
    offset_plus -= 1
    
    return x_start, y_position + offset_minus, x_end, y_position + offset_plus
  
  def row_matches_segment_length(self, y_position: int, z_position: int, x_start: int, x_end: int, rows: Dict[Tuple[int, int], Set[Tuple[int, int]]]):
    if (y_position, z_position) not in rows:
      return False 
    segments = rows[(y_position, z_position)]
    
    for (seg_x_start, seg_x_end) in segments:
      if seg_x_start == x_start and seg_x_end == x_end:
        rows[(y_position, z_position)].remove((seg_x_start, seg_x_end))
        if (len(rows[(y_position, z_position)]) == 0):
          del rows[(y_position, z_position)]
        return True
    
    return False
   
  def capture_rows(self, instance_matrix: np.array):
    diff_x = np.diff(instance_matrix, axis=0, append=0, prepend=0)
    
    start_x, start_y, start_z = np.where(diff_x > 0)
    end_x, end_y, end_z = np.where(diff_x < 0)
    
    border_positions = list(zip(start_x, start_y, start_z, np.zeros(len(start_x))))
    border_positions.extend(zip(end_x, end_y, end_z, np.ones(len(end_x))))
    
    sorted_start_and_end = sorted(border_positions, key=lambda x: x[0])
    
    start_map: Dict[Tuple[int, int], int] = {}
    
    rows: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    for begin_or_end in sorted_start_and_end:
      if begin_or_end[3] == 0:
        start_map[(begin_or_end[1], begin_or_end[2])] = begin_or_end[0]
      else:
        start = start_map[(begin_or_end[1], begin_or_end[2])]
        if (begin_or_end[1], begin_or_end[2]) in rows:
          rows[(begin_or_end[1], begin_or_end[2])].add((start, begin_or_end[0]))
        else:
          rows[(begin_or_end[1], begin_or_end[2])] = {(start, begin_or_end[0])}
          
    return rows
    