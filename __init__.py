import sys

sys.path.append("C:\\users\\anton\\appdata\\roaming\\python\\python311\\site-packages")
from .leaf_generation import MatlabEngineProvider

import time
import gc
from typing import Any, List, Dict, Tuple
from dataclasses import dataclass
import random
import csv
import json
from scipy.spatial import KDTree
import numpy as np

import bpy
from mathutils import Vector
# removed: VoxelGrid (voxelization not used)
from .tree_mesh_generation import SCATree
from .sca import SCA
from .edge_index import EdgeIndex
import bmesh

from .poisson_disk_sampling import poisson_disk_sampling_on_surface

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

@dataclass
class PreparedTree:
    index: int
    pos: Tuple[float, float]
    config_index: int
    location: Vector
    generator: SCATree
    sca: SCA
    start_time: float

class TreeConfiguration(bpy.types.PropertyGroup):
    path: bpy.props.StringProperty(
        name="Tree Configuration File", 
        description="Path to the file", 
        subtype='FILE_PATH',
        default="C:\\Users\\anton\\Documents\\Uni\\Spatial_Data_Analysis\\Procedual_Blender_Forest_Simulator\\tree_configs\\sphere_tree.json"  
    )
    weight: bpy.props.FloatProperty(
        name="Weight",
        description="Weight of the tree configuration",
        default=1,
        min=0,
    )

class ForestGenerator(bpy.types.Operator):
    bl_idname = "mesh.forest_generator"
    bl_label = "Forest Generator"
    bl_options = {'REGISTER', 'UNDO'}

    surface: bpy.props.StringProperty(
        name="Surface", 
        description="Path to the file", 
        subtype='FILE_PATH',
        default="C:\\Users\\anton\\Documents\\Uni\\Spatial_Data_Analysis\\Procedual_Blender_Forest_Simulator\\surface.csv"
    )
    treeConfigurationCount: bpy.props.IntProperty(
        name="Number of tree configurations",
        description="Number of tree configurations",
        default=2,
        min=1,
    )
    tree_configurations: bpy.props.CollectionProperty(type=TreeConfiguration) 
    updateForest: bpy.props.BoolProperty(name="Generate Forest", default=False)

    voxel_model_related_configuration_fields = {
        # "stem_height",
        # "stem_diameter",
    }
    
    @classmethod
    def poll(self, context):
        # Check if we are in object mode
        return context.mode == 'OBJECT'
    
    def update_tree_configurations(self):
        """
        Updates the tree configurations by adding or removing elements to match the desired tree configuration count.
        
        :return: None
        """
        
        current_count = len(self.tree_configurations)
        if self.treeConfigurationCount > current_count:
            for _ in range(self.treeConfigurationCount - current_count):
                self.tree_configurations.add()
        elif self.treeConfigurationCount < current_count:
            for _ in range(current_count - self.treeConfigurationCount):
                self.tree_configurations.remove(len(self.tree_configurations) - 1)
    
    def draw(self, context):
        """
        Draws the UI layout for the add-on, including generation settings and tree configurations.
        
        :param context: The context in which the UI is being drawn.
        :type context: bpy.types.Context
        :return: None
        :rtype: None
        """
        
        layout = self.layout
        col1 = layout.column()
        box = layout.box()
        box.prop(self, 'updateForest', icon='MESH_DATA')
        box.label(text="Generation Settings:")
        box.prop(self, 'surface')
        box.prop(self, 'treeConfigurationCount')

        for i, tree_config in enumerate(self.tree_configurations):
            col = box.column(align=True)
            col.scale_x = 20  # Adjust width scaling
            col.alignment = 'EXPAND'  # Expand to fit available space

            col.prop(tree_config, "path", text="Tree Config")
            col.prop(tree_config, "weight", text="Weight")
            
            col.separator()
        
    def execute(self, context):
        """
        Executes the process of procedurally generating a forest. The forest is generated based on the configuration of the operator.
        
        :param context: The Blender context in which the operator is executed.
        :type context: bpy.types.Context
        :return: A set indicating the execution status of the operator.
        :rtype: Set[str, str]
        """
        random.seed(random.randint(0, 1_000_000))
        self.update_tree_configurations()
        if not self.updateForest:
            return {'FINISHED'}
        
        surface_data = []
        if '.csv' in self.surface:
            with open(self.surface) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                for row in csv_reader:
                    surface_data.append([int(value) for value in row])

        tree_configurations: List[dict[str, Any]] = []
        configuration_weights: List[float] = []
        for tree_config in self.tree_configurations:
            with open(tree_config.path) as tree_config_json:
                tree_configurations.append(json.load(tree_config_json))
                configuration_weights.append(tree_config.weight)
        
        tree_voxel_configurations = [
            {k : v for k, v in tree_configuration.items() 
                if k in self.voxel_model_related_configuration_fields} 
            for tree_configuration in tree_configurations
        ]
        
        tree_mesh_configurations = [
            {k : v for k, v in tree_configuration.items()
                if k not in self.voxel_model_related_configuration_fields}
            for tree_configuration in tree_configurations
        ]

        for cfg in tree_mesh_configurations:
            if 'leaf_params' not in cfg or not isinstance(cfg.get('leaf_params'), dict):
                cfg['leaf_params'] = {
                    'pLADDh': [8, 3],
                    'pLADDd': [2.0, 1.5],
                    'fun_pLSD': [0.008, (0.00025)**2],
                    'totalLeafArea': 20,
                }

        crown_widths = [tree_configuration["crown_width"] for tree_configuration in tree_mesh_configurations]
        tree_positions = poisson_disk_sampling_on_surface(surface_data, configuration_weights, crown_widths)
        
        original_cursor_location = bpy.context.scene.cursor.location.copy()

        prepared_trees: List[PreparedTree] = []

        # Edge index for inter-tree proximity checks
        edge_index = EdgeIndex()
        # 1) Initialize all trees (volume, SCA state), but don't fully generate
        for i, tree_position in enumerate(tree_positions):
            tree_location = (tree_position[0][0], tree_position[0][1], 0)
            bpy.context.scene.cursor.location = tree_location
            bpy.context.view_layer.update()

            cfg = dict(tree_mesh_configurations[tree_position[1]])
            cfg_leaf_params = cfg.pop('leaf_params', None)
            sca_tree_generator = SCATree(
                noModifiers=False,
                subSurface=True,
                randomSeed=random.randint(0, 1_000_000),
                context=context,
                class_id=i,
                **cfg,
            )

            # propagate leaf_params from config to generator for downstream usage
            setattr(sca_tree_generator, 'leaf_params', cfg_leaf_params)

            sca = sca_tree_generator.prepare_growth(context, edge_index)
            prepared_trees.append(PreparedTree(
                index=i,
                pos=(float(tree_position[0][0]), float(tree_position[0][1])),
                config_index=int(tree_position[1]),
                location=bpy.context.scene.cursor.location.copy(),
                generator=sca_tree_generator,
                sca=sca,
                start_time=time.time(),
            ))

        # 2) Grow generation-by-generation across all trees
        if prepared_trees:
            max_generations = max(t.sca.maxiterations for t in prepared_trees)
            for gen in range(max_generations):
                all_finished = True
                for t in prepared_trees:
                    if not t.sca.is_finished():
                        t.sca.step_growth(gen)
                    all_finished = all_finished and t.sca.is_finished()
                if all_finished:
                    break

        # 3) Finalize meshes for all trees
        for t in prepared_trees:
            try: 
                bpy.context.scene.cursor.location = t.location
                bpy.context.view_layer.update()
                sca_tree_mesh = t.generator.finalize_tree(context)
                if sca_tree_mesh is None:
                    continue
                sca_tree_mesh.location = t.location
                elapsed_time = time.time() - t.start_time
                i = t.index
                print(f"{i+1} out of {len(prepared_trees)} trees generated at {t.pos} with configuration index {t.config_index} in {elapsed_time:.2f} seconds")
            except Exception as e:
                print(f"Error generating tree {t.index} at {t.pos}: {e}")

        matlab_engine_provider = MatlabEngineProvider()
        matlab_engine_provider.quit_engine()

        self.updateForest = False
        bpy.context.scene.cursor.location = original_cursor_location

        # Purge unused Blender data-blocks and run Python GC to free memory
        try:
            # Recursively remove all orphan data (meshes, materials, images, etc.)
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except Exception as e:
            print(f"Orphans purge failed: {e}")
        try:
            gc.collect()
        except Exception as e:
            print(f"Python GC failed: {e}")
        
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
    bpy.utils.register_class(TreeConfiguration)
    bpy.utils.register_class(ForestGenerator)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    matlab_engine_provider = MatlabEngineProvider()
    matlab_engine_provider.quit_engine()
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
    bpy.utils.unregister_class(TreeConfiguration)
    bpy.utils.unregister_class(ForestGenerator)
        
if __name__ == "__main__":
    register()