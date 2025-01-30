import bpy
import numpy as np
import bmesh
import random
from skimage.measure import marching_cubes
from typing import Tuple, List, Dict, Set
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

class VoxelGrid:
  def __init__(self):
    self.evaluated_forest = False
    
    # The first three elements of this tuple are the position of the tree, with the position of the tree being the position of the stem.
    # This position is in the middle of the grid (4th element). This does not need to be true when the crown is asymmetrical.
    self.trees: List[Tuple[int, int, int, np.ndarray]] = []
    
    self.unique_grid = np.zeros((50, 50, 50), dtype=int)
    
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
    if not self.evaluated_forest:
      self.evaluate_forest()
    
    mesh = bpy.data.meshes.new("VoxelMesh")
    obj = bpy.data.objects.new(f"VoxelObject_{index}_n", mesh)

    # verts, faces, _, _ = marching_cubes(self.grid, level=0.9999)    

    # mesh.from_pydata(verts.tolist(), [], faces.tolist())
    # mesh.update()

    # Prepare bmesh for geometry creation
    bm = bmesh.new()
    
    tree_grid = self.trees[index][3]

    for x in range(len(tree_grid)):
        for y in range(len(tree_grid[x])):
            for z in range(len(tree_grid[x][y])):
                if tree_grid[x][y][z] == 1:
                    self.add_voxel_to_bmesh(bm, x, y, z, tree_grid, self.cube_size)

    bm.to_mesh(mesh)
    bm.free()
    obj.location = tuple(np.array(self.trees[index][:3]) * self.cube_size)
    return obj
  
  def add_voxel_to_bmesh(self, bm, x, y, z, tree_grid, size):
    voxel_pos = (x * size, y * size, z * size)
    
    # Check neighbors and add faces only where necessary
    if True or not self.is_filled(x-1, y, z, tree_grid):
        self.add_face_to_bmesh(bm, voxel_pos, size, "left")
    if True or not self.is_filled(x+1, y, z, tree_grid): 
        self.add_face_to_bmesh(bm, voxel_pos, size, "right")
    if True or not self.is_filled(x, y-1, z, tree_grid): 
        self.add_face_to_bmesh(bm, voxel_pos, size, "front")
    if True or not self.is_filled(x, y+1, z, tree_grid): 
        self.add_face_to_bmesh(bm, voxel_pos, size, "back")
    if True or not self.is_filled(x, y, z-1, tree_grid): 
        self.add_face_to_bmesh(bm, voxel_pos, size, "bottom")
    if True or not self.is_filled(x, y, z+1, tree_grid): 
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
  
  def is_filled(self, x, y, z, tree_grid):
    if not (0 <= x < len(tree_grid) and 0 <= y < len(tree_grid[x]) and 0 <= z < len(tree_grid[x][y])):
      return False 
    return self.unique_grid[x][y][z] == 1
  
  def add_tree(self, position: Tuple[int, int, int], stem_diameter: float, stem_height: float, crown_diameter: float):
    self.evaluated_forest = False
    
    tree_grid = np.zeros((int(crown_diameter / self.cube_size + 1), int(crown_diameter / self.cube_size + 1), int((stem_height + crown_diameter) / self.cube_size + 1)), dtype=np.int8)
    
    stem_radius = int(stem_diameter / 2 / self.cube_size)
    crown_radius = int(crown_diameter / 2 / self.cube_size)

    #Add stem
    stem_height_range = np.arange(int(stem_height / self.cube_size))
    stem_diameter_range = np.arange(-int(stem_diameter / self.cube_size), int(stem_diameter / self.cube_size))
    j, k = np.meshgrid(stem_diameter_range, stem_diameter_range, indexing='ij')
    mask = j**2 + k**2 <= stem_radius**2

    for i in stem_height_range:
      tree_grid[j[mask]+tree_grid.shape[0]//2, k[mask]+tree_grid.shape[1]//2, i] = 1

    # Add crown
    crown_range = np.arange(-int(crown_diameter / self.cube_size), int(crown_diameter / self.cube_size))
    i, j, k = np.meshgrid(crown_range, crown_range, crown_range, indexing='ij')
    mask = i**2 + j**2 + k**2 <= crown_radius**2

    tree_grid[j[mask]+tree_grid.shape[0]//2, k[mask]+tree_grid.shape[1]//2, i[mask] + int((stem_height+crown_diameter/2) / self.cube_size)] = 1
    # self.unique_grid = np.zeros((50, 50, 50), dtype=int)
    # for index in np.argwhere(tree_grid == 1):
    #   self.unique_grid[index[0], index[1], index[2]] = 1
    
    # self.trees.append((int(position[0] / self.cube_size) - len(tree_grid)//2, int(position[1] / self.cube_size) - len(tree_grid[0])//2, int(position[2] / self.cube_size), tree_grid))
    self.trees.append((int(position[0] / self.cube_size), int(position[1] / self.cube_size), int(position[2] / self.cube_size), tree_grid))
    
    
  def evaluate_forest(self):
    self.evaluated_forest = True
    
    max_range = np.max([t[:2] for t in self.trees])
    
    tree_positions = KDTree([t[:3] for t in self.trees])
    
    evaluated_pairs: Set[Tuple[int, int]] = set()
    
    for i, tree in enumerate(self.trees):
      potential_collisions = tree_positions.query_ball_point(tree[:3], max_range)
      
      for collision_index in potential_collisions:
        if collision_index == i:
          continue
        pair = (i if i < collision_index else collision_index, collision_index if i < collision_index else i) 
        if pair in evaluated_pairs:
          continue
        evaluated_pairs.add(pair)
        
        other_tree = self.trees[collision_index]
        
        self.resolve_collision(tree, other_tree)
        
  def resolve_collision(self, tree1: Tuple[int, int, int, np.ndarray], tree2: Tuple[int, int, int, np.ndarray]):
    x1, y1, z1, tree1_grid = tree1
    x2, y2, z2, tree2_grid = tree2
    translation = np.array([x1 - x2, y1 - y2, z1 - z2])
    
    tree1_filled_cells = np.argwhere(tree1_grid == 1)
    
    # translate to tree2 coordinate space
    tree1_filled_cells = tree1_filled_cells + translation
    
    tree2_collision_cells = self.get_colliding_cells(tree2_grid, tree1_filled_cells)
    tree1_collision_cells = tree2_collision_cells - translation
    
    if len(tree2_collision_cells) == 0:
      return
    
    tree1_collision_edge_cells = self.get_collision_edge_cells(tree1_grid, tree2_collision_cells - translation)
    tree_2_collision_edge_cells = self.get_collision_edge_cells(tree2_grid, tree2_collision_cells)
    
    # set cells to 2 to distinguish them from other filled cells
    tree1_grid[tree1_collision_cells[:, 0], tree1_collision_cells[:, 1], tree1_collision_cells[:, 2]] = 1
    tree2_grid[tree2_collision_cells[:, 0], tree2_collision_cells[:, 1], tree2_collision_cells[:, 2]] = 0
    # return 
    # trimmed_mask = self.trim_mask(tree2_grid, tree1_filled_cells)
    # print(np.sum(tree2_grid[trimmed_mask[:, 0], trimmed_mask[:, 1], trimmed_mask[:, 2]]))
    # tree1_set = set(map(tuple, np.argwhere(tree1_grid == 1) + np.array([x1, y1, z1])))
    # tree2_set = set(map(tuple, np.argwhere(tree2_grid == 1) + np.array([x2, y2, z2])))
    # print(tree1_set.intersection(tree2_set))
    
    self.assign_collision_cells(tree1_grid, tree2_grid, tree1_collision_edge_cells, tree_2_collision_edge_cells, translation)
  
  def get_colliding_cells(self, tree_grid: np.ndarray, filled_translated_cells: np.ndarray):
    contained_cells = self.trim_mask(tree_grid, filled_translated_cells)
    
    collision_indices = np.argwhere(tree_grid[contained_cells[:, 0], contained_cells[:, 1], contained_cells[:, 2]] == 1)
    collision_cells = contained_cells[collision_indices]
    
    return collision_cells.reshape(-1, 3)
  
  def trim_mask(self, tree_grid: np.ndarray, mask: np.ndarray):
    # get only the indices that are within the tree
    lower_limit = np.array([0, 0, 0])
    upper_limit = np.array([len(tree_grid), len(tree_grid[0]), len(tree_grid[0][0])])
    tree_contains_cell = np.all(mask >= lower_limit, axis=1) & np.all(mask < upper_limit, axis=1)
    
    return mask[np.array(tree_contains_cell)]
    
  def get_collision_edge_cells(self, tree_grid: np.ndarray, collision_cells: set[np.ndarray]) -> np.ndarray:
    # Define neighbor offsets (6-connectivity)
    neighbor_offsets = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    # Get all neighbors
    neighbors = collision_cells[:, None, :] + neighbor_offsets[None, :, :]
    neighbors = neighbors.reshape(-1, 3)
    
    # Filter out neighbors that are out of bounds
    valid_mask = (
        (neighbors[:, 0] >= 0) & (neighbors[:, 0] < tree_grid.shape[0]) &
        (neighbors[:, 1] >= 0) & (neighbors[:, 1] < tree_grid.shape[1]) &
        (neighbors[:, 2] >= 0) & (neighbors[:, 2] < tree_grid.shape[2])
    )
    valid_neighbors = neighbors[valid_mask]
    
    # Filter out neighbors that are not edge cells
    edge_mask = tree_grid[valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]] == 1
    edge_cells = valid_neighbors[edge_mask]
    
    return edge_cells
  
  def assign_collision_cells(self, 
                             tree1_grid: np.ndarray, 
                             tree2_grid: np.ndarray, 
                             tree1_collision_edge_cells: np.ndarray, 
                             tree2_collision_edge_cells: np.ndarray,
                             translation: np.ndarray):
    rounds = 5
    min_radius = 1
    max_radius = 3
    
    if len(tree1_collision_edge_cells) == 0 or len(tree2_collision_edge_cells) == 0:
      return
  
    random.seed(7e4)
    for round in range(rounds):
      sphere_radius = random.randint(min_radius, max_radius)
      sphere_cells = self.get_cells_for_sphere(sphere_radius)
      
      collision_edge_cell = random.choice(tree1_collision_edge_cells)
      self.add_collision_cells_to_tree(tree1_grid, sphere_cells + collision_edge_cell)
      self.subtract_collision_cells_from_tree(tree2_grid, sphere_cells + (collision_edge_cell + translation))
      
      collision_edge_cell = random.choice(tree2_collision_edge_cells)
      self.add_collision_cells_to_tree(tree2_grid, sphere_cells + collision_edge_cell)
      self.subtract_collision_cells_from_tree(tree1_grid, sphere_cells + (collision_edge_cell - translation))
      
    self.assign_rest_of_collision_cells(tree1_grid, tree2_grid, translation)
      
  def get_cells_for_sphere(self, radius: int):
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    z = np.arange(-radius, radius + 1)
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    
    distances = np.sum(grid**2, axis=1)
    
    inside_sphere = grid[distances <= radius**2]
    
    return inside_sphere
  
  def add_collision_cells_to_tree(self, tree_grid: np.ndarray, selected_cells: np.ndarray):
    contained_cells = self.trim_mask(tree_grid, selected_cells)
    conflicted_contained_cells = contained_cells[tree_grid[contained_cells[:, 0], contained_cells[:, 1], contained_cells[:, 2]] == 2]
    tree_grid[conflicted_contained_cells[:, 0], conflicted_contained_cells[:, 1], conflicted_contained_cells[:, 2]] = 1
          
  def subtract_collision_cells_from_tree(self, tree_grid: np.ndarray, selected_cells: np.ndarray):
    contained_cells = self.trim_mask(tree_grid, selected_cells)
    conflicted_contained_cells = contained_cells[tree_grid[contained_cells[:, 0], contained_cells[:, 1], contained_cells[:, 2]] == 2]
    tree_grid[conflicted_contained_cells[:, 0], conflicted_contained_cells[:, 1], conflicted_contained_cells[:, 2]] = 0
    
  def assign_rest_of_collision_cells(self, 
                                     tree1_grid: np.ndarray, 
                                     tree2_grid: np.ndarray,
                                     translation: np.ndarray):
    tree1_collision_cells = np.argwhere(tree1_grid == 2)
    mask = (tree1_grid != 1)
    tree1_distances = distance_transform_edt(mask)
    tree1_conflicted_distances = tree1_distances[tree1_collision_cells[:, 0], tree1_collision_cells[:, 1], tree1_collision_cells[:, 2]]
    
    tree2_collision_cells = np.argwhere(tree2_grid == 2)
    mask = (tree2_grid != 1)
    tree2_distances = distance_transform_edt(mask)
    tree2_conflicted_distances = tree2_distances[tree2_collision_cells[:, 0], tree2_collision_cells[:, 1], tree2_collision_cells[:, 2]]
    
    tree1_cells_closer = tree1_collision_cells[tree1_conflicted_distances <= tree2_conflicted_distances]
    tree1_cells_farther = tree1_collision_cells[tree1_conflicted_distances > tree2_conflicted_distances]
    
    tree1_grid[tree1_cells_closer[:, 0], tree1_cells_closer[:, 1], tree1_cells_closer[:, 2]] = 1
    tree1_grid[tree1_cells_farther[:, 0], tree1_cells_farther[:, 1], tree1_cells_farther[:, 2]] = 0
    
    tree2_cells_closer = tree2_collision_cells[tree2_conflicted_distances < tree1_conflicted_distances]
    tree2_cells_farther = tree2_collision_cells[tree2_conflicted_distances >= tree1_conflicted_distances]
    
    tree2_grid[tree2_cells_closer[:, 0], tree2_cells_closer[:, 1], tree2_cells_closer[:, 2]] = 1
    tree2_grid[tree2_cells_farther[:, 0], tree2_cells_farther[:, 1], tree2_cells_farther[:, 2]] = 0
    
  def greedy_meshing(self, index: int):
    if not self.evaluated_forest:
      self.evaluate_forest()
    
    quads = self.capture_quads(index)
    
    mesh = bpy.data.meshes.new(f"VoxelMesh")
    obj = bpy.data.objects.new(f"VoxelObject_{index}", mesh)

    # Prepare bmesh for geometry creation
    bm = bmesh.new()
    
    for quad in quads:
      x_start, y_start, z_start, x_end, y_end, z_end = quad
      # x_end += 1
      # y_end += 1
      # z_end += 1 
      x_start_position = x_start * self.cube_size - self.cube_size / 2
      y_start_postion = y_start * self.cube_size - self.cube_size / 2
      z_start_position = z_start * self.cube_size - self.cube_size / 2
      x_end_position = x_end * self.cube_size + self.cube_size / 2
      y_end_position = y_end * self.cube_size + self.cube_size / 2
      z_end_position = z_end * self.cube_size + self.cube_size / 2
      
      # verts = [
      #   (x_start_position, y_start * self.cube_size - self.cube_size / 2, z_start * self.cube_size - self.cube_size / 2),
      #   (x_end * self.cube_size, y_start * self.cube_size, z_start * self.cube_size),
      #   (x_end * self.cube_size, y_end * self.cube_size, z_start * self.cube_size),
      #   (x_start * self.cube_size, y_end * self.cube_size, z_start * self.cube_size),
      #   (x_start * self.cube_size, y_start * self.cube_size, z_end * self.cube_size),
      #   (x_end * self.cube_size, y_start * self.cube_size, z_end * self.cube_size),
      #   (x_end * self.cube_size, y_end * self.cube_size, z_end * self.cube_size),
      #   (x_start * self.cube_size, y_end * self.cube_size, z_end * self.cube_size)
      # ]
      
      verts = [
        (x_start_position, y_start_postion, z_start_position),
        (x_end_position, y_start_postion, z_start_position),
        (x_end_position, y_end_position, z_start_position),
        (x_start_position, y_end_position, z_start_position),
        (x_start_position, y_start_postion, z_end_position),
        (x_end_position, y_start_postion, z_end_position),
        (x_end_position, y_end_position, z_end_position),
        (x_start_position, y_end_position, z_end_position)
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
    
    obj.location = tuple(np.array(self.trees[index][:3]) * self.cube_size)
    return obj
  
  def capture_quads(self, index: int):
    instance_matrix = self.trees[index][3]
    
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
    border_positions.extend(zip(end_x - 1, end_y, end_z, np.ones(len(end_x))))
    
    sorted_start_and_end = sorted(border_positions, key=lambda x: (x[0], x[3]))
    
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
    