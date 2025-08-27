import sys

sys.path.append("C:\\users\\anton\\appdata\\roaming\\python\\python311\\site-packages")
from .leaf_generation import MatlabEngineProvider

import time
import gc
import os
from typing import Any, List, Tuple
from dataclasses import dataclass
import random
import json
import numpy as np

import bpy
from mathutils import Vector
# removed: VoxelGrid (voxelization not used)
from .tree_mesh_generation import SCATree
from .sca import SCA
from .edge_index import EdgeIndex
import bmesh

from .poisson_disk_sampling import poisson_disk_sampling_on_surface, poisson_disk_sampling_low_vegetation

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
    config: dict
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

    surface_object_name: bpy.props.StringProperty(
        name="Surface Object",
        description="Name of the mesh object to use as terrain surface",
        default=""
    )
    treeConfigurationCount: bpy.props.IntProperty(
        name="Number of tree configurations",
        description="Number of tree configurations",
        default=2,
        min=1,
    )
    tree_configurations: bpy.props.CollectionProperty(type=TreeConfiguration)
    updateForest: bpy.props.BoolProperty(name="Generate Forest", default=False)
    low_vegetation_density: bpy.props.FloatProperty(
        name="Low Vegetation Density",
        description="Density of low vegetation placement",
        default=1.0,
        min=0.1,
        max=10.0
    )

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
        box.prop_search(self, 'surface_object_name', bpy.data, 'objects', text='Surface Object')
        box.prop(self, 'low_vegetation_density', text='Low Vegetation Density')
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
        
        terrain_obj = bpy.data.objects.get(self.surface_object_name) if self.surface_object_name else None
        use_mesh_surface = terrain_obj is not None and getattr(terrain_obj.data, 'polygons', None) is not None

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

        # Pre-sample full configurations in Poisson sampling and receive per-tree sampled configs
        tree_positions = []
        if use_mesh_surface:
            tree_positions = poisson_disk_sampling_on_surface(terrain_obj, configuration_weights, tree_mesh_configurations)
        else:
            print('No surface object set. Aborting forest generation.')
            return {'FINISHED'}
        
        original_cursor_location = bpy.context.scene.cursor.location.copy()

        prepared_trees: List[PreparedTree] = []

        edge_index = EdgeIndex()
        if use_mesh_surface:
            edge_index.set_terrain(terrain_obj)
        else:
            raise RuntimeError("Mesh surface required to build terrain BVH for collision checks.")
        for i, tree_position in enumerate(tree_positions):
            pos = tree_position[0]
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                tree_location = (pos[0], pos[1], pos[2])
            else:
                tree_location = (pos[0], pos[1], 0)
            bpy.context.scene.cursor.location = tree_location
            bpy.context.view_layer.update()
            cfg_sampled = dict(tree_position[1])
            cfg_leaf_params = cfg_sampled.pop('leaf_params', None)

            sca_tree_generator = SCATree(
                noModifiers=False,
                subSurface=True,
                randomSeed=random.randint(0, 1_000_000),
                context=context,
                class_id=i,
                **cfg_sampled,
            )

            setattr(sca_tree_generator, 'leaf_params', cfg_leaf_params)

            sca = sca_tree_generator.prepare_growth(context, edge_index)
            prepared_trees.append(PreparedTree(
                index=i,
                pos=(float(tree_position[0][0]), float(tree_position[0][1]), float(tree_position[0][2])),
                config=cfg_sampled,
                location=bpy.context.scene.cursor.location.copy(),
                generator=sca_tree_generator,
                sca=sca,
                start_time=time.time(),
            ))

        if prepared_trees:
            max_generations = max(t.sca.maxiterations for t in prepared_trees)
            print(f"Starting growth simulation with {max_generations} generations")
            
            # Calculate progress tracking intervals (every 10%)
            progress_interval = max(1, max_generations // 10)
            last_progress_reported = 0
            
            for gen in range(max_generations):
                all_finished = True
                trees_this_generation = prepared_trees.copy()
                random.shuffle(trees_this_generation)
                for t in trees_this_generation:
                    if not t.sca.is_finished():
                        t.sca.step_growth(gen)
                    all_finished = all_finished and t.sca.is_finished()
                
                # Report progress every 10%
                if gen > 0 and (gen % progress_interval == 0 or gen == max_generations - 1):
                    progress_percent = round((gen / max_generations) * 100)
                    if progress_percent > last_progress_reported:
                        print(f"Growth progress: {progress_percent}% of generations completed ({gen}/{max_generations})")
                        last_progress_reported = progress_percent
                    
                if all_finished:
                    final_progress = round(((gen + 1) / max_generations) * 100)
                    print(f"All trees finished growing after {gen + 1} generations ({final_progress}% complete)")
                    break

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
                print(f"{i+1} out of {len(prepared_trees)} trees generated at {t.pos} in {elapsed_time:.2f} seconds")
            except Exception as e:
                print(f"Error generating tree {t.index} at {t.pos}: {e}")

        matlab_engine_provider = MatlabEngineProvider()
        matlab_engine_provider.quit_engine()

        self.updateForest = False
        bpy.context.scene.cursor.location = original_cursor_location

        # Organize trees and leaf exports into collection first
        trees_collection = self.get_or_create_collection("Trees")
        for obj in bpy.context.scene.objects:
            if obj.name.startswith("Tree_") or obj.name.startswith("leaves_export"):
                self.move_to_collection(obj, trees_collection)
        
        # Place low vegetation
        if self.low_vegetation_density > 0:
            self.place_low_vegetation(terrain_obj)

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
    
    def place_low_vegetation(self, terrain_obj):
        """Place low vegetation objects using Poisson disk sampling."""
        print("Placing low vegetation...")
        
        # Get low vegetation positions
        vegetation_positions = poisson_disk_sampling_low_vegetation(
            terrain_obj, 
            self.low_vegetation_density
        )
        
        if not vegetation_positions:
            print("No low vegetation positions generated.")
            return
        
        # Create or get low vegetation collection
        low_vegetation_collection = self.get_or_create_collection("Low Vegetation")
        
        # Load low vegetation models from LowVegetation.blend
        vegetation_models = self.load_low_vegetation_models()
        
        if not vegetation_models:
            print("No low vegetation models found.")
            return
        
        # Calculate all bounding boxes and max height once
        vegetation_bboxes = self.calculate_vegetation_bounding_boxes(vegetation_models)
        max_veg_height = self.get_max_vegetation_height_from_bboxes(vegetation_bboxes)
        
        # Calculate tree bounding boxes once
        tree_bboxes = self.calculate_tree_bounding_boxes(max_veg_height)
        
        # Create low vegetation material
        low_veg_material = self.create_low_vegetation_material()
        
        # Place vegetation objects
        placed_count = 0
        for pos in vegetation_positions:
            # Choose random vegetation model first
            model_name = random.choice(list(vegetation_models.keys()))
            model_obj = vegetation_models[model_name]
            
            # Check if position would cause collision with existing trees
            if self.is_too_close_to_trees(pos, model_name, vegetation_bboxes, tree_bboxes):
                continue
            
            # Create instance
            new_obj = model_obj.copy()
            new_obj.data = model_obj.data.copy()
            new_obj.name = f"LowVegetation_{placed_count}"
            
            # Position on terrain
            new_obj.location = (pos[0], pos[1], pos[2])
            
            # Get terrain normal at this point and orient the object
            normal = self.get_terrain_normal_at_point(terrain_obj, pos)
            if normal:
                # Create rotation matrix to align with terrain normal
                up_vector = Vector((0, 0, 1))
                rotation_matrix = up_vector.rotation_difference(normal).to_matrix()
                new_obj.rotation_euler = rotation_matrix.to_euler()
            
            # Apply low vegetation material
            if new_obj.data.materials:
                new_obj.data.materials[0] = low_veg_material
            else:
                new_obj.data.materials.append(low_veg_material)
            
            # Add to collection
            low_vegetation_collection.objects.link(new_obj)
            
            placed_count += 1
        
        print(f"Placed {placed_count} low vegetation objects.")
    
    def get_or_create_collection(self, collection_name):
        """Get or create a collection with the given name."""
        if collection_name in bpy.data.collections:
            return bpy.data.collections[collection_name]
        else:
            new_collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(new_collection)
            return new_collection
    
    def move_to_collection(self, obj, target_collection):
        """Move an object to the specified collection."""
        # Remove from scene collection if present
        if obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(obj)
        
        # Remove from all other collections
        for collection in bpy.data.collections:
            if obj.name in collection.objects:
                collection.objects.unlink(obj)
        
        # Add to target collection
        target_collection.objects.link(obj)
    
    def load_low_vegetation_models(self):
        """Load low vegetation models from LowVegetation.blend file."""
        models = {}
        
        # Look for LowVegetation.blend in the addon directory
        addon_dir = os.path.dirname(__file__)
        low_veg_path = os.path.join(addon_dir, "low_vegetation", "LowVegetation.blend")
        
        if not os.path.exists(low_veg_path):
            print(f"LowVegetation.blend not found at: {low_veg_path}")
            return models
        
        try:
            # Load the blend file
            with bpy.data.libraries.load(low_veg_path, link=False) as (data_from, data_to):
                # Load all objects and filter by name
                data_to.objects = [name for name in data_from.objects 
                                 if name.startswith(('fern_02_', 'High grass clump'))]
            
            # Store the loaded objects
            for obj in data_to.objects:
                if obj is not None:
                    models[obj.name] = obj
            
            print(f"Loaded {len(models)} low vegetation models: {list(models.keys())}")
            
        except Exception as e:
            print(f"Error loading low vegetation models: {e}")
            import traceback
            traceback.print_exc()
        
        return models
    
    def create_low_vegetation_material(self):
        """Create dark green material for low vegetation."""
        mat_name = "Low Vegetation"
        
        # Check if material already exists
        if mat_name in bpy.data.materials:
            return bpy.data.materials[mat_name]
        
        # Create new material
        mat = bpy.data.materials.new(name=mat_name)
        mat.diffuse_color = (0.1, 0.3, 0.1, 1.0)

        return mat
    
    def is_too_close_to_trees(self, pos, model_name, vegetation_bboxes, tree_bboxes):
        """Check if vegetation object would collide with existing trees using pre-calculated bounding boxes."""
        vegetation_bbox = vegetation_bboxes.get(model_name)
        if not vegetation_bbox:
            return False  # If no bbox data, allow placement
            
        for tree_name, tree_bbox in tree_bboxes.items():
            if self.check_bbox_collision(vegetation_bbox, pos, tree_bbox):
                return True
        return False
    
    def calculate_vegetation_bounding_boxes(self, vegetation_models):
        """Calculate bounding boxes for all vegetation models once."""
        vegetation_bboxes = {}
        
        for model_name, model_obj in vegetation_models.items():
            try:
                # Get bounding box of the model
                bbox_corners = [model_obj.matrix_world @ Vector(corner) for corner in model_obj.bound_box]
                vegetation_bboxes[model_name] = {
                    'min': Vector((
                        min(corner.x for corner in bbox_corners),
                        min(corner.y for corner in bbox_corners),
                        min(corner.z for corner in bbox_corners)
                    )),
                    'max': Vector((
                        max(corner.x for corner in bbox_corners),
                        max(corner.y for corner in bbox_corners),
                        max(corner.z for corner in bbox_corners)
                    )),
                    'height': max(corner.z for corner in bbox_corners) - min(corner.z for corner in bbox_corners)
                }
            except Exception as e:
                print(f"Error calculating bounding box for {model_name}: {e}")
        
        return vegetation_bboxes
    
    def get_max_vegetation_height_from_bboxes(self, vegetation_bboxes):
        """Get maximum height from pre-calculated bounding boxes."""
        max_height = 0.0
        
        for model_name, bbox_data in vegetation_bboxes.items():
            max_height = max(max_height, bbox_data['height'])
        
        # Add some buffer for safety
        return max_height + 0.5
    
    def calculate_tree_bounding_boxes(self, max_veg_height):
        """Calculate height-limited bounding boxes for all trees once."""
        tree_bboxes = {}
        
        for obj in bpy.context.scene.objects:
            if obj.name.startswith("Tree_"):
                try:
                    # Get tree vertices and filter by height
                    tree_mesh = obj.data
                    tree_vertices_world = [obj.matrix_world @ v.co for v in tree_mesh.vertices]
                    
                    # Filter vertices to only those within the height range of low vegetation
                    tree_base_z = min(v.z for v in tree_vertices_world)
                    max_tree_z_for_collision = tree_base_z + max_veg_height
                    
                    # Only consider tree vertices up to the maximum vegetation height
                    filtered_vertices = [v for v in tree_vertices_world if v.z <= max_tree_z_for_collision]
                    
                    if filtered_vertices:
                        # Calculate tree bounding box from filtered vertices
                        tree_bboxes[obj.name] = {
                            'min': Vector((
                                min(v.x for v in filtered_vertices),
                                min(v.y for v in filtered_vertices),
                                min(v.z for v in filtered_vertices)
                            )),
                            'max': Vector((
                                max(v.x for v in filtered_vertices),
                                max(v.y for v in filtered_vertices),
                                max(v.z for v in filtered_vertices)
                            ))
                        }
                except Exception as e:
                    print(f"Error calculating bounding box for tree {obj.name}: {e}")
        
        return tree_bboxes
    
    def check_bbox_collision(self, vegetation_bbox, vegetation_pos, tree_bbox):
        """Check collision between vegetation and pre-calculated tree bounding box."""
        try:
            # Calculate vegetation bounding box at the test position
            # Use the pre-calculated bbox dimensions and translate to the new position
            bbox_size = vegetation_bbox['max'] - vegetation_bbox['min']
            pos_vector = Vector(vegetation_pos)
            
            veg_min = pos_vector - bbox_size * 0.5
            veg_max = pos_vector + bbox_size * 0.5
            
            # Check if bounding boxes overlap
            return not (
                tree_bbox['max'].x < veg_min.x or tree_bbox['min'].x > veg_max.x or
                tree_bbox['max'].y < veg_min.y or tree_bbox['min'].y > veg_max.y or
                tree_bbox['max'].z < veg_min.z or tree_bbox['min'].z > veg_max.z
            )
            
        except Exception as e:
            print(f"Error in bbox collision check: {e}")
            return True  # Conservative: assume collision if error
    

    

    

    
    def get_terrain_normal_at_point(self, terrain_obj, pos):
        """Get terrain normal at a specific point."""
        try:
            # Raycast from above the point
            origin = Vector((pos[0], pos[1], pos[2] + 100.0))
            direction = Vector((0.0, 0.0, -1.0))
            
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = terrain_obj.evaluated_get(depsgraph)
            hit, location, normal, index = eval_obj.ray_cast(origin, direction)
            
            if hit and normal:
                return normal.normalized()
        except Exception as e:
            print(f"Error getting terrain normal: {e}")
        
        return None
        
    def create_random_material(self, name):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Base Color'].default_value = (random.random(), random.random(), random.random(), 1)
        return mat
            
def menu_func(self, context):
    op = self.layout.operator(ForestGenerator.bl_idname, text="Generate Forest", icon='PLUGIN')
    op.updateForest = False

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