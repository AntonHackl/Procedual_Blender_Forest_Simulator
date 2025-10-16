from os import remove
from os.path import exists, join
from zipfile import ZipFile
from typing import TYPE_CHECKING, Dict, List, Optional, DefaultDict
from collections import defaultdict

if TYPE_CHECKING:
    import bpy.types
    from mathutils import Vector
import bpy
from .sca import Branchpoint

def extract(zipfile: str, name: str, dest: str) -> None:
    zf = zipfile + '.zip'
    with ZipFile(zf) as z:
        z.extract(name, dest)
    remove(zf)
    
def load_materials(library: str, material_name: str) -> Dict[str, "bpy.types.Material"]:
    """Given a path to a library .blend file and the name of a material, append that material and return a reference to it.
    
    Note that the name of the material may change if a material with the same name is already present.
    """
    if not TYPE_CHECKING:
        import bpy
    before = set(m.name for m in bpy.data.materials)
    with bpy.data.libraries.load(library) as (data_from, data_to):
        data_to.materials = [m for m in data_from.materials if m.startswith(material_name)]
    after = set(m.name for m in bpy.data.materials)
    new = after - before
    if len(new) < 1:
        raise ValueError("While loading material %s from library %s %d materials were found (%s) instead of just 1" % (
            material_name, library, len(new), str(new)))
    return {m: bpy.data.materials[m] for m in new}
    
def load_particlesettings(library: str, object_name: str) -> Dict[str, "bpy.types.ParticleSettings"]:
    if not TYPE_CHECKING:
        import bpy
    beforep = set(m.name for m in bpy.data.particles)
    before = set(m.name for m in bpy.data.objects)
    with bpy.data.libraries.load(library) as (data_from, data_to):
        data_to.objects = [m for m in data_from.objects if m.startswith(object_name)]
    afterp = set(m.name for m in bpy.data.particles)
    after = set(m.name for m in bpy.data.objects)
    new = after - before
    if len(new) < 1:
        raise ValueError("While loading objects with names starting with %s from library %s 0 objects were found" % (
            object_name, library))
    new = afterp - beforep
    if len(new) < 1:
        raise ValueError("While loading particle settings from objects with names starting with %s from library %s no particle settings were found" % (object_name, library))
    return {p: bpy.data.particles[p] for p in new}
    
def load_materials_from_bundled_lib(script_name: str, library: str, material_name: str) -> Dict[str, "bpy.types.Material"]:
    """Load a material from a library located in the installation directory of a script."""
    if not TYPE_CHECKING:
        import bpy
    for dir in ('addons', 'addons_contrib'):
        for path in bpy.utils.script_paths():
            fullpath = join(path, dir, script_name, library)
            if exists(fullpath):
                return load_materials(fullpath, material_name)
            if exists(fullpath + ".zip"):
                extract(fullpath, library, join(path, dir, script_name))
                return load_materials(fullpath, material_name)
    return load_materials('/root/media/data/Tree_Generation/Procedual_Blender_Forest_Simulator/material_lib.blend', material_name)

def load_particlesettings_from_bundled_lib(script_name: str, library: str, object_name: str) -> Dict[str, "bpy.types.ParticleSettings"]:
    """Load particle settings associated with objects from a library located in the installation directory of a script."""
    if not TYPE_CHECKING:
        import bpy
    for dir in ('addons', 'addons_contrib'):
        for path in bpy.utils.script_paths():
            fullpath = join(path, dir, script_name, library)
            if exists(fullpath):
                return load_particlesettings(fullpath, object_name)
            if exists(fullpath + ".zip"):
                extract(fullpath, library, join(path, dir, script_name))
                return load_particlesettings(fullpath, object_name)
    return load_particlesettings('/root/media/data/Tree_Generation/Procedual_Blender_Forest_Simulator/material_lib.blend', object_name)
 
def get_vertex_group(context: "bpy.types.Context", name: str) -> Optional["bpy.types.VertexGroup"]:
    """Get a reference to the named vertex group of the active object, creating it if necessary."""
    if not TYPE_CHECKING:
        import bpy
    ob = context.view_layer.objects.active
    if ob is None:
        return None

    if ob.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    if name in ob.vertex_groups:
        return ob.vertex_groups[name]
    else:
        bpy.ops.object.vertex_group_add()
        vg = ob.vertex_groups.active
        vg.name = name
        return vg

def create_inverse_graph(branchpoints: List["Branchpoint"]) -> Dict[int, List[int]]:
    node_to_children: DefaultDict[int, List[int]] = defaultdict(list)
    for bp in branchpoints:
        if bp.parent is not None:
            node_to_children[bp.parent].append(bp.index)
    return dict(node_to_children)