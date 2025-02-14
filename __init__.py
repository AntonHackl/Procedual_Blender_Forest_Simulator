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

  treeCount: bpy.props.IntProperty(
    name="Tree Count",
    description="Number of trees to generate",
    default=5,
    min=1,
  )

  updateTree: bpy.props.BoolProperty(name="Update Tree", default=False)

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
    box.label(text="Generation Settings:")
    box.prop(self, 'forestXExtend')
    box.prop(self, 'forestYExtend')
    box.prop(self, 'treeCount')
    
  def execute(self, context):
    voxel_grid = VoxelGrid()
    # problem pair: (5, 21)
    for position in [(random.randint(0, self.forestXExtend), random.randint(0, self.forestYExtend), 0) for _ in range(self.treeCount)]:
      voxel_grid.add_tree(position, 1, 4, 6)
    
    tree_objects = [voxel_grid.generate_mesh(i) for i in range(self.treeCount)]
    
    rest_collection = bpy.data.collections.get("Rest")
    if not rest_collection:
      rest_collection = bpy.data.collections.new("Rest")
      bpy.context.scene.collection.children.link(rest_collection)
    
    for i, tree_object in enumerate(tree_objects):
      material = self.create_random_material(f"Material_{i}")
      if tree_object.data.materials:
        tree_object.data.materials[0] = material
      else:
        tree_object.data.materials.append(material)
      rest_collection.objects.link(tree_object)
    
    processing_collection = bpy.data.collections.get("Processing")
    if not processing_collection:
      processing_collection = bpy.data.collections.new("Processing")
      bpy.context.scene.collection.children.link(processing_collection)
    
    for i, tree_object in enumerate(tree_objects):
      bpy.context.view_layer.update()
      rest_collection.objects.unlink(tree_object)
      processing_collection.objects.link(tree_object)
      bpy.context.view_layer.update()
      
      sca_tree = SCATree(
        context,
        numberOfEndpoints=200,
        interNodeLength=0.25,
        killDistance=0.1,
        useGroups=True,
        crownGroup="Processing",
        noModifiers=False,
        subSurface=True,
        randomSeed=random.randint(0, 1_000_000),
      )
      
      tree_location = tree_object.location.copy()
      tree_object.location = (-3, -3, 0)
      tree_mesh = sca_tree.create_tree(context)
      
      tree_object.location = (tree_location[0] - 3, tree_location[1] - 3, tree_location[2])
      tree_mesh.location = tree_location
      bpy.context.view_layer.update()
    
    return {'FINISHED'}
        
  def create_random_material(self, name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (random.random(), random.random(), random.random(), 1)
    return mat
            
def menu_func(self, context):
  self.layout.operator(ForestGenerator.bl_idname, text="Generate Forest",
                                          icon='PLUGIN').updateTree = True

def register():
  bpy.utils.register_class(ForestGenerator)
  bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
  bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
  bpy.utils.unregister_class(ForestGenerator)
      
if __name__ == "__main__":
  register()