import sys
sys.path.append("C:\\users\\anton\\appdata\\roaming\\python\\python39\\site-packages")

bl_info = {
    "name": "SCA Tree Generator",
    "author": "michel anders (varkenvarken)",
    "version": (0, 2, 14),
    "blender": (2, 93, 0),
    "location": "View3D > Add > Mesh",
    "description": "Adds a tree created with the space colonization algorithm starting at the 3D cursor",
    "warning": "",
    "wiki_url": "https://github.com/varkenvarken/spacetree/wiki",
    "tracker_url": "",
    "category": "Add Mesh"}

from time import time
import random
from functools import partial
from math import sin,cos
import numpy as np
import csv

import bpy
from bpy.props import FloatProperty, IntProperty, BoolProperty, EnumProperty
from mathutils import Vector,Euler,Matrix,Quaternion
from .voxel_grid import VoxelGrid
from .sca_not_init import SCATree
import bmesh

bl_info = {
  "name": "Forest Generator",
  "author": "Anton Hackl",
  "version": (0, 2, 14),
  "blender": (2, 93, 0),
  "location": "View3D > Add > Mesh",
  "description": "Adds a forest of trees created with the space colonization algorithm starting at the 3D cursor",
  "warning": "",
  "wiki_url": "https://github.com/varkenvarken/spacetree/wiki",
  "tracker_url": "",
  "category": "Add Mesh"}

def poisson_disk_sampling(width, height, radius, k=30):
  def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

  def in_circle(point, radius, points):
    for p in points:
      if distance(point, p) < radius:
        return True
    return False

  def generate_random_point_around(point, radius):
    r1 = random.random()
    r2 = random.random()
    radius = radius * (r1 + 1)
    angle = 2 * np.pi * r2
    new_x = point[0] + radius * np.cos(angle)
    new_y = point[1] + radius * np.sin(angle)
    return (new_x, new_y)

  grid = [[None for _ in range(height)] for _ in range(width)]
  cell_size = radius / np.sqrt(2)
  active_list = []
  points = []

  initial_point = (random.uniform(0, width), random.uniform(0, height))
  points.append(initial_point)
  active_list.append(initial_point)
  grid[int(initial_point[0] // cell_size)][int(initial_point[1] // cell_size)] = initial_point

  while active_list:
      idx = random.randint(0, len(active_list) - 1)
      point = active_list[idx]
      found = False
      for _ in range(k):
          new_point = generate_random_point_around(point, radius)
          if 0 <= new_point[0] < width and 0 <= new_point[1] < height and not in_circle(new_point, radius, points):
              points.append(new_point)
              active_list.append(new_point)
              grid[int(new_point[0] // cell_size)][int(new_point[1] // cell_size)] = new_point
              found = True
              break
      if not found:
          active_list.pop(idx)

  return points

class ForestGenerator(bpy.types.Operator):
  bl_idname = "mesh.forest_generator"
  bl_label = "Forest Generator"
  bl_options = {'REGISTER', 'UNDO'}

  forestXExtend: bpy.props.IntProperty(
    name="Forest X Extend",
    description="forestXExtend in Blender units",
    default=30,
    min=1,
  )

  forestYExtend: bpy.props.IntProperty(
    name="Forest Y Extend",
    description="forestYExtend in Blender units",
    default=30,
    min=1,
  )

  surface: bpy.props.StringProperty(name="Surface File", description="Path to the file", subtype='FILE_PATH')
  
  treeCount: bpy.props.IntProperty(
    name="Tree Count",
    description="Number of trees to generate",
    default=1,
    min=1,
  )

  updateTree: bpy.props.BoolProperty(name="Update Tree", default=False)
  samePosUpdate: bpy.props.BoolProperty(name="Same Position Update Tree", default=False)

  @classmethod
  def poll(self, context):
    # Check if we are in object mode
    return context.mode == 'OBJECT'
  
  def draw(self, context):
    layout = self.layout

    # layout.prop(self, 'updateTree', icon='MESH_DATA')

    columns=layout.row()
    col1=columns.column()
    col2=columns.column()
    
    box = col1.box()
    box.prop(self, 'updateTree', icon='MESH_DATA')
    box.prop(self, 'samePosUpdate', icon='MESH_DATA')
    box.label(text="Generation Settings:")
    box.prop(self, 'forestXExtend')
    box.prop(self, 'forestYExtend')
    box.prop(self, 'treeCount')
    box.prop(self, 'surface')
    
  def execute(self, context):
    if not (self.updateTree or self.samePosUpdate):
      return {'FINISHED'}
    
    csv_data = []
    if '.csv' in self.surface:
      with open(self.surface) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            csv_data.append([int(value) for value in row])
    
    min_x = min([point[0] for point in csv_data])
    max_x = max([point[0] for point in csv_data])
    min_y = min([point[1] for point in csv_data])
    max_y = max([point[1] for point in csv_data])

    # Perform Poisson disk sampling within the bounding box
    sampled_points = poisson_disk_sampling(max_x - min_x, max_y - min_y, radius=3)

    # Translate sampled points to the polygon's coordinate space
    translated_points = [(point[0] + min_x, point[1] + min_y) for point in sampled_points]

    voxel_grid = VoxelGrid()
    for position in translated_points:
        voxel_grid.add_tree((position[0], position[1], 0), 1, 4, 6)

    tree_mesh_groups = [voxel_grid.generate_mesh_groups(i) for i in range(len(translated_points))]
    
    rest_collection = bpy.data.collections.get("Rest")
    if not rest_collection:
      rest_collection = bpy.data.collections.new("Rest")
      bpy.context.scene.collection.children.link(rest_collection)
      
    crown_collection = bpy.data.collections.get("Crown")
    if not crown_collection:
      crown_collection = bpy.data.collections.new("Crown")
      bpy.context.scene.collection.children.link(crown_collection)
      
    exlusion_collection = bpy.data.collections.get("Exclusion")
    if not exlusion_collection:
      exlusion_collection = bpy.data.collections.new("Exclusion")
      bpy.context.scene.collection.children.link(exlusion_collection)

    for i, mesh_groups in enumerate(tree_mesh_groups):
      material = self.create_random_material(f"Material_{i}")
      if mesh_groups[1].data.materials:
        mesh_groups[1].data.materials[0] = material
      else:
        mesh_groups[1].data.materials.append(material)
      rest_collection.objects.link(mesh_groups[0])
      # rest_collection.objects.link(mesh_groups[1])
    
    for i, mesh_groups in enumerate(tree_mesh_groups):
      bpy.context.view_layer.update()
      rest_collection.objects.unlink(mesh_groups[0])
      # rest_collection.objects.unlink(mesh_groups[1])
      crown_collection.objects.link(mesh_groups[0])
      # exlusion_collection.objects.link(mesh_groups[1])
      bpy.context.view_layer.update()
      
      tree_location = mesh_groups[0].location.copy()
      # mesh_groups[0].location = mesh_groups[1].location = (-3, -3, 0)
      bpy.context.scene.cursor.location = Vector((tree_location[0] + 3, tree_location[1] + 3, tree_location[2]))
      bpy.context.view_layer.update()
      
      sca_tree = SCATree(
        context,
        numberOfEndpoints=200,
        interNodeLength=0.25,
        killDistance=0.1,
        useGroups=True,
        crownGroup="Crown",
        exclusionGroup="Rest",
        noModifiers=False,
        subSurface=True,
        randomSeed=random.randint(0, 1_000_000),
      )
      tree_mesh = sca_tree.create_tree(context)
      
      tree_mesh.location = bpy.context.scene.cursor.location.copy()
      
      crown_collection.objects.unlink(mesh_groups[0])
      # exlusion_collection.objects.unlink(mesh_groups[1])
      rest_collection.objects.link(mesh_groups[0])
      # rest_collection.objects.link(mesh_groups[1])
      bpy.context.view_layer.update()
      
    self.updateTree = False
    self.samePosUpdate = False
    
    return {'FINISHED'}
        
  def create_random_material(self, name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (random.random(), random.random(), random.random(), 1)
    return mat
            
def menu_func(self, context):
  self.layout.operator(ForestGenerator.bl_idname, text="Generate Forest",
                                          icon='PLUGIN').updateTree = False

def register():
  bpy.utils.register_class(ForestGenerator)
  bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
  bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
  bpy.utils.unregister_class(ForestGenerator)
      
if __name__ == "__main__":
  register()