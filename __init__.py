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
from typing import Any
import random
import csv
import json

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

class ForestGenerator(bpy.types.Operator):
  bl_idname = "mesh.forest_generator"
  bl_label = "Forest Generator"
  bl_options = {'REGISTER', 'UNDO'}

  surface: bpy.props.StringProperty(
    name="Surface File", 
    description="Path to the file", 
    subtype='FILE_PATH',
    default="C:\\Users\\anton\\Desktop\\surface.csv"
  )
  tree_configuration: bpy.props.StringProperty(
    name="Tree Configuration File", 
    description="Path to the file", 
    subtype='FILE_PATH',
    default="C:\\Users\\anton\\Desktop\\sphere_tree.json"  
  )
  treeCount: bpy.props.IntProperty(
    name="Tree Count",
    description="Number of trees to generate",
    default=1,
    min=1,
  )

  updateForest: bpy.props.BoolProperty(name="Update Tree", default=False)

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
    box.prop(self, 'updateForest', icon='MESH_DATA')
    box.label(text="Generation Settings:")
    box.prop(self, 'treeCount')
    box.prop(self, 'surface')
    box.prop(self, 'tree_configuration')
    
  def execute(self, context):
    if not (self.updateForest):
      return {'FINISHED'}
    
    surface_data = []
    if '.csv' in self.surface:
      with open(self.surface) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            surface_data.append([int(value) for value in row])

    with open(self.tree_configuration) as tree_config_json:
      tree_configuration: dict[str, Any] = json.load(tree_config_json)
      
    voxel_grid = VoxelGrid()
    voxel_grid.generate_forest(tree_configuration, surface_data)
    tree_meshes = [voxel_grid.generate_mesh(i) for i in range(len(voxel_grid.trees))]
    
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

    for i, tree_mesh in enumerate(tree_meshes):
      material = self.create_random_material(f"Material_{i}")
      if tree_mesh.data.materials:
        tree_mesh.data.materials[0] = material
      else:
        tree_mesh.data.materials.append(material)
      rest_collection.objects.link(tree_mesh)
      # rest_collection.objects.link(mesh_groups[1])
    
    for i, tree_mesh in enumerate(tree_meshes):
      bpy.context.view_layer.update()
      rest_collection.objects.unlink(tree_mesh)
      crown_collection.objects.link(tree_mesh)
      bpy.context.view_layer.update()
      
      tree_location = tree_mesh.location.copy()
      tree_dimensions = tree_mesh.dimensions
      bpy.context.scene.cursor.location = Vector((
        tree_location[0] + tree_dimensions.x / 2, 
        tree_location[1] + tree_dimensions.y / 2, 
        tree_location[2]
      ))
      bpy.context.view_layer.update()
      
      sca_tree = SCATree(
        context,
        numberOfEndpoints=tree_configuration["numberOfEndpoints"],
        interNodeLength=tree_configuration["interNodeLength"],
        killDistance=0.1,
        useGroups=True,
        crownGroup="Crown",
        exclusionGroup="Rest",
        noModifiers=False,
        subSurface=True,
        randomSeed=random.randint(0, 1_000_000),
      )
      sca_tree_mesh = sca_tree.create_tree(context)
      
      sca_tree_mesh.location = bpy.context.scene.cursor.location.copy()
      
      crown_collection.objects.unlink(tree_mesh)
      rest_collection.objects.link(tree_mesh)
      bpy.context.view_layer.update()
      
    self.updateForest = False
    
    return {'FINISHED'}
        
  def create_random_material(self, name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (random.random(), random.random(), random.random(), 1)
    return mat
            
def menu_func(self, context):
  self.layout.operator(ForestGenerator.bl_idname, text="Generate Forest",
                                          icon='PLUGIN').updateForest = False

def register():
  bpy.utils.register_class(ForestGenerator)
  bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
  bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
  bpy.utils.unregister_class(ForestGenerator)
      
if __name__ == "__main__":
  register()