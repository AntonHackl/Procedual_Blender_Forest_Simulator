# ##### BEGIN GPL LICENSE BLOCK #####
#
#  SCA Tree Generator, a Blender addon
#  (c) 2013, 2014 Michel J. Anders (varkenvarken)
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# for first time
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
from random import random,gauss
import random as ra
from functools import partial
from math import sin,cos
import numpy as np

import bpy
from bpy.props import FloatProperty, IntProperty, BoolProperty, EnumProperty
from mathutils import Vector,Euler,Matrix,Quaternion
from scipy.spatial import KDTree
import bmesh

from .scanew import SCA, Branchpoint # the core class that implements the space colonization algorithm and the definition of a segment
from .timer import Timer
from .utils import load_materials_from_bundled_lib, load_particlesettings_from_bundled_lib, get_vertex_group
from .voxel_grid import VoxelGrid

def availableGroups(self, context):
    return [(name, name, name, n) for n,name in enumerate(bpy.data.collections.keys())]

def availableGroupsOrNone(self, context):
    groups = [ ('None', 'None', 'None', 0) ]
    return groups + [(name, name, name, n+1) for n,name in enumerate(bpy.data.collections.keys())]

def availableObjects(self, context):
    return [(name, name, name, n+1) for n,name in enumerate(bpy.data.objects.keys())]

particlesettings = None
barkmaterials = None

def availableParticleSettings(self, context):
    global particlesettings
    # im am not sure why self.__class__.particlesettings != bpy.types.MESH_OT_sca_tree ....
    settings = [ ('None', 'None', 'None', 0) ]
    #    return settings + [(name, name, name, n+1) for n,name in enumerate(bpy.types.MESH_OT_sca_tree.particlesettings.keys())]
    # (identifier, name, description, number)
    # note, when we create a new tree the particles settings will be made unique so they can be tweaked individually for
    # each tree. That also means they will  have distinct names, but we manipulate those to be displayed in a consistent way
    return settings + [(name, name.split('.')[0], name, n+1) for n,name in enumerate(particlesettings.keys())]

def availableBarkMaterials(self, context):
    global barkmaterials
    return [(name, name.split('.')[0], name, n) for n,name in enumerate(barkmaterials.keys())]

def ellipsoid(r=5,rz=5,p=Vector((0,0,8)),taper=0):
    r2=r*r
    z2=rz*rz
    if rz>r : r = rz
    while True:
        x = (random()*2-1)*r
        y = (random()*2-1)*r
        z = (random()*2-1)*r
        f = (z+r)/(2*r)
        f = 1 + f*taper if taper>=0 else (1-f)*-taper
        if f*x*x/r2+f*y*y/r2+z*z/z2 <= 1:
            yield p+Vector((x,y,z))

def pointInsideMesh(pointrelativetocursor,ob):
    # adapted from http://blenderartists.org/forum/showthread.php?195605-Detecting-if-a-point-is-inside-a-mesh-2-5-API&p=1691633&viewfull=1#post1691633
    mat = ob.matrix_world.inverted()
    orig = mat@(pointrelativetocursor+bpy.context.scene.cursor.location)
    count = 0
    axis=Vector((0,0,1))
    while True:
        _, location,normal,index = ob.ray_cast(orig,orig+axis*10000.0)
        if index == -1: break
        count += 1
        orig = location + axis*0.00001
    if count%2 == 0:
        return False
    return True
    
def ellipsoid2(rxy=5,rz=5,p=Vector((0,0,8)),surfacebias=1,topbias=1):
    while True:
        phi = 6.283*random()
        theta = 3.1415*(random()-0.5)
        r = random()**((1.0/surfacebias)/2)
        x = r*rxy*cos(theta)*cos(phi)
        y = r*rxy*cos(theta)*sin(phi)
        st=sin(theta)
        st = (((st+1)/2)**(1.0/topbias))*2-1
        z = r*rz*st
        #print(">>>%.2f %.2f %.2f "%(x,y,z))
        m = p+Vector((x,y,z))
# undocumented feature: bombs if any selected object is not a mesh. Use crown and shadow/exclusion groups instead
#        reject = False
#        for ob in bpy.context.selected_objects:
#            # probably we should check if each object is a mesh
#            if pointInsideMesh(m,ob) :
#                reject = True
#                break
#        if not reject:
        yield m

def halton3D(index):
    """
    return a quasi random 3D vector R3 in [0,1].
    each component is based on a halton sequence. 
    quasi random is good enough for our purposes and is 
    more evenly distributed then pseudo random sequences. 
    See en.m.wikipedia.org/wiki/Halton_sequence
    """

    def halton(index, base):
        result=0
        f=1.0/base
        I=index
        while I>0:
            result += f*(I%base)
            I=int(I/base)
            f/=base
        return result
    return Vector((halton(index,2),halton(index,3),halton(index,5)))

def insidegroup(pointrelativetocursor, group):
    if group not in bpy.data.collections : return False
    for ob in bpy.data.collections.get(group).objects:
        if isinstance(ob.data, bpy.types.Mesh) and pointInsideMesh(pointrelativetocursor,ob):
            return True
    return False

def groupdistribution(crowngroup,shadowgroup=None,shadowdensity=0.5, seed=0,size=Vector((1,1,1)),pointrelativetocursor=Vector((0,0,0))):
    if crowngroup == shadowgroup:
        shadowgroup = None # safeguard otherwise every marker would be rejected
    nocrowngroup = crowngroup not in bpy.data.collections
    noshadowgroup = (shadowgroup is None) or (shadowgroup not in bpy.data.collections) or (shadowgroup == 'None')
    index=100+seed
    nmarkers=0
    nyield=0
    while True:
        nmarkers+=1
        v = halton3D(index)
        v[0] *= size[0]
        v[1] *= size[1]
        v[2] *= size[2]
        v+=pointrelativetocursor
        index+=1
        insidecrown = nocrowngroup or insidegroup(v,crowngroup)
        outsideshadow = noshadowgroup # if there's no shadowgroup we're always outside of it
        if not outsideshadow:
            inshadow = insidegroup(v,shadowgroup) # if there is, check if we're inside the group
            if not inshadow:
                outsideshadow = True
            else:
                outsideshadow = random() > shadowdensity  # if inside the group we might still generate a marker if the density is low
        # if shadowgroup overlaps all or a significant part of the crowngroup
        # no markers will be yielded and we would be in an endless loop.
        # so if we yield too few correct markers we start yielding them anyway.
        lowyieldrate = (nmarkers>200) and (nyield/nmarkers < 0.01)
        if (insidecrown and outsideshadow) or lowyieldrate:
            nyield+=1
            yield v
        
def groupExtends(group):
    """
    return a size,minimum tuple both Vector elements, describing the size and position
    of the bounding box in world space that encapsulates all objects in a group.
    """
    bb=[]
    if group in bpy.data.collections:
        for ob in bpy.data.collections[group].objects:
            rot = ob.matrix_world.to_quaternion()
            scale = ob.matrix_world.to_scale()
            translate = ob.matrix_world.translation
            for v in ob.bound_box: # v is not a vector but an array of floats
                p = ob.matrix_world @ Vector(v[0:3])
                bb.extend(p[0:3])
        mx = Vector((max(bb[0::3]), max(bb[1::3]), max(bb[2::3])))
        mn = Vector((min(bb[0::3]), min(bb[1::3]), min(bb[2::3])))
        return mx-mn,mn
    return Vector((2,2,2)),Vector((-1,-1,-1)) # a 2x2x2 cube when the group does not exist
    
def createMarkers(tree,scale=0.05):
    #not used as markers are parented to tree object that is created at the cursor position
    #p=bpy.context.scene.cursor.location
    
    verts=[]
    faces=[]

    tetraeder = [Vector((-1,1,-1)),Vector((1,-1,-1)),Vector((1,1,1)),Vector((-1,-1,1))]
    tetraeder = [v * scale for v in tetraeder]
    tfaces = [(0,1,2),(0,1,3),(1,2,3),(0,3,2)]
    
    for eip,ep in enumerate(tree.endpoints):
        verts.extend([ep + v for v in tetraeder])
        n=len(faces)
        faces.extend([(f1+n,f2+n,f3+n) for f1,f2,f3 in tfaces])
        
    mesh = bpy.data.meshes.new('Markers')
    mesh.from_pydata(verts,[],faces)
    mesh.update(calc_edges=True)
    return mesh

def basictri(bp, verts, radii, power, scale, p):
    v = bp.v + p
    nv = len(verts)
    r=(bp.connections**power)*scale
    a=-r
    b=r*0.5   # cos(60)
    c=r*0.866 # sin(60)
    verts.extend([v+Vector((a,0,0)), v+Vector((b,-c,0)), v+Vector((b,c,0))]) # provisional, should become an optimally rotated triangle
    radii.extend([bp.connections,bp.connections,bp.connections])
    return (nv, nv+1, nv+2)
    
def _simpleskin(bp, loop, verts, faces, radii, power, scale, p):
    newloop = basictri(bp, verts, radii, power, scale, p)
    for i in range(3):
        faces.append((loop[i],loop[(i+1)%3],newloop[(i+1)%3],newloop[i]))
    if bp.apex:
        _simpleskin(bp.apex, newloop, verts, faces, radii, power, scale, p)
    if bp.shoot:
        _simpleskin(bp.shoot, newloop, verts, faces, radii, power, scale, p)
    
def simpleskin(bp, verts, faces, radii, power, scale, p):
    loop = basictri(bp, verts, radii, power, scale, p)
    if bp.apex:
        _simpleskin(bp.apex, loop, verts, faces, radii, power, scale, p)
    if bp.shoot:
        _simpleskin(bp.shoot, loop, verts, faces, radii, power, scale, p)

def leafnode(bp, verts, faces, radii, p1, p2, scale=0.1):
    loop1 = basictri(bp, verts, radii, 0.0, scale, p1)
    loop2 = basictri(bp, verts, radii, 0.0, scale, p2)
    for i in range(3):
        faces.append((loop1[i],loop1[(i+1)%3],loop2[(i+1)%3],loop2[i]))
    if bp.apex:
        leafnode(bp.apex, verts, faces, radii, p1, p2, scale)
    if bp.shoot:
        leafnode(bp.shoot, verts, faces, radii, p1, p2, scale)

def createLeaves2(tree, roots, p, scale):
    verts = []
    faces = []
    radii = []
    for r in roots:
        leafnode(r, verts, faces, radii, p, p++Vector((0,0, scale)), scale)
    mesh = bpy.data.meshes.new('LeafEmitter')
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)
    return mesh, verts, faces, radii

def pruneTree(tree, generation):
    nbp = []
    i2p = {}
    #print()
    for i,bp in enumerate(tree):
        #print(i, bp.v, bp.generation, bp.parent, end='')
        if bp.generation >= generation:
            #print(' keep', end='')
            bp.index = i
            i2p[i] = len(nbp)
            nbp.append(bp)
        #print()
    return nbp, i2p
    
def createGeometry(tree, power=0.5, scale=0.01,
    nomodifiers=True, skinmethod='NATIVE', subsurface=False,
    bleaf=1.0,
    leafParticles='None',
    objectParticles='None',
    emitterscale=0.1,
    timeperf=True,
    prune=0):
    
    timings = Timer()
    start = time()
    p=bpy.context.scene.cursor.location
    verts=[]
    edges=[]
    faces=[]
    radii=[]
    roots=set()
    
    # prune if requested
    tree.branchpoints, index2position = pruneTree(tree.branchpoints, prune)
    if len(tree.branchpoints) < 2:
        return None
    # Loop over all branchpoints and create connected edges
    #print('\ngenerating skeleton')
    
    for n,bp in enumerate(tree.branchpoints):
        #print(n, bp.index, bp.v, bp.generation, bp.parent)
        verts.append(bp.v+p)
        radii.append(bp.connections)
        if not (bp.parent is None) :
            #print(bp.parent,index2position[bp.parent])
            edges.append((len(verts)-1,index2position[bp.parent]))
        else :
            nv=len(verts)
            roots.add(bp)
        bp.index=n
    # radii = np.array(radii)/np.max(radii)
    timings.add('skeleton')
    
    # native skinning method
    if nomodifiers == False and skinmethod == 'NATIVE': 
        # add a quad edge loop to all roots
        for r in roots:
            simpleskin(r, verts, faces, radii, power, scale, p)
            
    # end of native skinning section
    timings.add('nativeskin')
    
    # create the (skinned) tree mesh
    mesh = bpy.data.meshes.new('Tree')
    mesh.from_pydata(verts, edges, faces)
    mesh.update(calc_edges=True)
    
    # create the tree object an make it the only selected and active object in the scene
    obj_new = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.view_layer.active_layer_collection.collection.objects.link(obj_new)
    # bpy.context.collection.objects.link(obj_new)
    for ob in bpy.context.scene.objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active = obj_new
    obj_new.select_set(True)
    # bpy.context.scene.objects.active = obj_new
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    
    # add a leaves vertex group
    leavesgroup = get_vertex_group(bpy.context, 'Leaves')
    
    maxr = max(radii) if len(radii)>0 else 0.03 # pruning might have been so aggressive that there are no radii (NB. python 3.3 does not know the default keyword for the max() fie
    if maxr<=0 : maxr=1.0
    maxr=float(maxr)
    for v,r in zip(mesh.vertices,radii):
        leavesgroup.add([v.index], (1.0-r/maxr)**bleaf, 'REPLACE')
    timings.add('createmesh')
    
    # add a subsurf modifier to smooth the branches 
    if nomodifiers == False:
        if subsurface:
            bpy.ops.object.modifier_add(type='SUBSURF')
            bpy.context.active_object.modifiers[0].levels = 1
            bpy.context.active_object.modifiers[0].render_levels = 1
            bpy.context.active_object.modifiers[0].uv_smooth = 'PRESERVE_CORNERS'

        # add a skin modifier
        if skinmethod == 'BLENDER':
          # sphere_id = 0
          # add spheres to demonstrate trunk nodes
          # for position in trunk_node_positions:
          #   # if bp.connections <= 1 and bp.parent is not None and tree.branchpoints[bp.parent].shoot != bp:
          #   # print(tree.branchpoints[bp.parent].connections)
          #     # bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(bp.v[0], bp.v[1], bp.v[2]))
          #     # sphere = bpy.context.object
          #   mesh_sphere = bpy.data.meshes.new('Basic_Sphere')
          #   bm = bmesh.new()
          #   bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, diameter=0.1)
          #   bm.to_mesh(mesh_sphere)
          #   bm.free()
          #   basic_sphere = bpy.data.objects.new("Basic_Sphere", mesh_sphere)
          #   basic_sphere.location = position
          #   basic_sphere.name = f"Sphere_{sphere_id}"
          #   sphere_id += 1
          #   collection = bpy.data.collections.get("Collection")
          #   collection.objects.link(basic_sphere)
          
          bpy.ops.object.modifier_add(type='SKIN')
          bpy.context.active_object.modifiers[-1].use_smooth_shade=True
          bpy.context.active_object.modifiers[-1].use_x_symmetry=True
          bpy.context.active_object.modifiers[-1].use_y_symmetry=True
          bpy.context.active_object.modifiers[-1].use_z_symmetry=True
          
          skinverts = bpy.context.active_object.data.skin_vertices[0].data

          for i,v in enumerate(skinverts):
            v.radius = [(radii[i]**power)*scale,(radii[i]**power)*scale]
            if i in roots:
                v.use_root = True
          
          # add an extra subsurf modifier to smooth the skin
          bpy.ops.object.modifier_add(type='SUBSURF')
          bpy.context.active_object.modifiers[-1].levels = 1
          bpy.context.active_object.modifiers[-1].render_levels = 2
          #Changed from me
          #bpy.context.active_object.modifiers[-1].use_subsurf_uv = True
          # to (same above)
          # bpy.context.active_object.modifiers[0].uv_smooth = 'PRESERVE_CORNERS'
              
    timings.add('modifiers')

    # create a particles based leaf emitter (if we have leaves and/or objects)
    if leafParticles != 'None' or objectParticles != 'None':
        mesh, verts, faces, radii = createLeaves2(tree, roots, Vector((0,0,0)), emitterscale)
        obj_leaves2 = bpy.data.objects.new(mesh.name, mesh)
        base = bpy.context.collection.objects.link(obj_leaves2)
        obj_leaves2.parent = obj_new
        # bpy.context.scene.objects.active = obj_leaves2
        bpy.context.view_layer.objects.active = obj_leaves2
        obj_leaves2.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        # add a LeafDensity vertex group to the LeafEmitter object
        leavesgroup = get_vertex_group(bpy.context, 'LeafDensity')
        maxr = max(radii)
        if maxr<=0 : maxr=1.0
        maxr=float(maxr)
        for v,r in zip(mesh.vertices,radii):
            leavesgroup.add([v.index], (1.0-r/maxr)**bleaf, 'REPLACE')

        if leafParticles != 'None':
            bpy.ops.object.particle_system_add()
            obj_leaves2.particle_systems.active.settings = particlesettings[leafParticles]
            obj_leaves2.particle_systems.active.settings.count = len(faces)
            obj_leaves2.particle_systems.active.name = 'Leaves'
            obj_leaves2.particle_systems.active.vertex_group_density = leavesgroup.name
        if objectParticles != 'None':
            bpy.ops.object.particle_system_add()
            obj_leaves2.particle_systems.active.settings = particlesettings[objectParticles]
            obj_leaves2.particle_systems.active.settings.count = len(faces)
            obj_leaves2.particle_systems.active.name = 'Objects'
            obj_leaves2.particle_systems.active.vertex_group_density = leavesgroup.name
    end = time()
    print('geometry default time', end-start)  
    # bpy.context.scene.objects.active = obj_new
    start = time()
    obj_processed = segmentIntoTrunkAndBranch(tree, obj_new, (np.array(radii)**power)*scale)
    bpy.ops.object.shade_smooth()
    
    timings.add('leaves')
    
    if timeperf:
        print(timings)
        
    bpy.data.objects.remove(obj_new, do_unlink=True)
    
    end = time()
    print('geometry my time', end-start)
    return obj_processed

def segmentIntoTrunkAndBranch(tree, obj_new, radii):
    top = find_top_of_trunk(tree.branchpoints)
            
    trunk_nodes = [top]
    trunk_indices = [top.index]

    while trunk_nodes[-1].parent is not None:  
        trunk_nodes.append(tree.branchpoints[trunk_nodes[-1].parent])
        trunk_indices.append(trunk_nodes[-1].index)
        

    trunk_node_positions = [trunk_node.v for trunk_node in trunk_nodes]
    branch_node_positions = [bp.v for bp in tree.branchpoints if bp not in trunk_nodes]
    branch_node_indices = [i for i in range(len(tree.branchpoints)) if i not in trunk_indices]

    trunk_material = create_material("TrunkMaterial", (0.77, 0.64, 0.52, 1)) # light brown
    branch_material = create_material("BranchMaterial", (0.36, 0.25, 0.20, 1)) # dark brown
    assign_material(obj_new, trunk_material)
    assign_material(obj_new, branch_material)
    trunk_vertex_indices = []
    branch_vertex_indices = []
    # bpy.ops.object.modifier_apply(modifier="Skin")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj_new.evaluated_get(depsgraph)
    # final_mesh = evaluated_obj.to_mesh()
    final_mesh = bpy.data.meshes.new_from_object(evaluated_obj)
    obj_processed = bpy.data.objects.new('Tree_Processed', final_mesh)
    bpy.context.view_layer.active_layer_collection.collection.objects.link(obj_processed)
    # obj_new.data = final_mesh

    trunk_node_kd_tree = KDTree(trunk_node_positions)
    branch_node_kd_tree = KDTree(branch_node_positions)
    for poly in final_mesh.polygons:
        position = poly.center
        trunk_node_distance, trunk_node_index = trunk_node_kd_tree.query(position, 1)
        branch_node_distance, branch_node_index = branch_node_kd_tree.query(position, 1)
        
        if trunk_node_distance - radii[trunk_indices[trunk_node_index]] < branch_node_distance - radii[branch_node_indices[branch_node_index]]:
            poly.material_index = 0
        else:
            poly.material_index = 1
        
    assign_vertices_to_group(obj_new, "TrunkGroup", trunk_vertex_indices)
    assign_vertices_to_group(obj_new, "BranchGroup", branch_vertex_indices)

    obj_processed.data.update()
    obj_new.data.update()
    
    return obj_processed

def create_inverse_graph(branchpoints):
  node_to_children = {}
  for bp in branchpoints:
    if bp.parent is not None:
      if bp.parent not in node_to_children:
        node_to_children[bp.parent] = []
      node_to_children[bp.parent].append(bp.index)
  return node_to_children

def find_top_of_trunk(branchpoints):
  node_to_children = create_inverse_graph(branchpoints)
  candidate = branchpoints[0]
  queue = node_to_children[0]
  while len(queue) > 0:
    current_index = queue.pop(0)
    current_node = branchpoints[current_index]
    if (current_node.parent is not None 
        and branchpoints[current_node.parent].shoot != current_node):
      if current_node.connections < candidate.connections:
        candidate = current_node
      queue.extend(node_to_children.get(current_index, []))
  return candidate

def create_material(name, color):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.diffuse_color = color
    return mat

def assign_material(obj, mat):
    if mat.name not in obj.data.materials:
        obj.data.materials.append(mat)

def assign_vertices_to_group(obj, group_name, vertex_indices):
    if group_name not in obj.vertex_groups:
        group = obj.vertex_groups.new(name=group_name)
    else:
        group = obj.vertex_groups[group_name]
    group.add(vertex_indices, 1.0, 'ADD')

class SCATree():

  def __init__(self, 
              context,
              interNodeLength=0.25,
              killDistance=0.1,
              influenceRange=15.,
              tropism=0.,
              power=0.3,
              scale=0.01,
              useGroups=False,
              crownGroup='None',
              shadowGroup='None',
              shadowDensity=0.5,
              exclusionGroup='None',
              useTrunkGroup=False,
              trunkGroup=None,
              crownSize=5.,
              crownShape=1.,
              crownOffset=3.,
              surfaceBias=1.,
              topBias=1.,
              randomSeed=0,
              maxIterations=40,
              pruningGen=0,
              numberOfEndpoints=100,
              newEndPointsPer1000=0,
              maxTime=0.0,
              bLeaf=1.0,
              addLeaves=False,
              emitterScale=0.01,
              noModifiers=True,
              subSurface=False,
              showMarkers=False,
              markerScale=0.05,
              timePerformance=False,
              apicalcontrol=0.0,
              apicalcontrolfalloff=1.0,
              apicalcontroltiming=10
              ):
    self.internodeLength = interNodeLength
    self.killDistance = killDistance
    self.influenceRange = influenceRange
    self.tropism = tropism
    self.power = power
    self.scale = scale
    self.useGroups = useGroups
    self.crownGroup = crownGroup
    self.shadowGroup = shadowGroup
    self.shadowDensity = shadowDensity
    self.exclusionGroup = exclusionGroup
    self.useTrunkGroup = useTrunkGroup
    self.trunkGroup = trunkGroup
    self.crownSize = crownSize
    self.crownShape = crownShape
    self.crownOffset = crownOffset
    self.surfaceBias = surfaceBias
    self.topBias = topBias
    self.randomSeed = randomSeed
    self.maxIterations = maxIterations
    self.pruningGen = pruningGen
    self.numberOfEndpoints = numberOfEndpoints
    self.newEndPointsPer1000 = newEndPointsPer1000
    self.maxTime = maxTime
    self.bLeaf = bLeaf
    self.addLeaves = addLeaves
    
    # self.leafParticles = availableParticleSettings(self, context)[0]
    # self.objectParticles = availableParticleSettings(self, context)[0]
    self.emitterScale = emitterScale
    # self.barMaterial = availableBarkMaterials(self, context)[0]
    self.updateTree = False
    self.noModifiers = noModifiers
    self.subSurface = subSurface
    # self.skinMethod = ('NATIVE','Space tree','Spacetrees own skinning method',1)
    self.skinMethod = 'NATIVE'
    self.showMarkers = showMarkers
    self.markerScale = markerScale
    self.timePerformance = timePerformance
    self.apicalcontrol = apicalcontrol   
    self.apicalcontrolfalloff = apicalcontrolfalloff 
    self.apicalcontroltiming = apicalcontroltiming    

  def create_tree(self, context):
      # if not self.updateTree:
      #     return {'PASS_THROUGH'}

      timings=Timer()
      
      
      # necessary otherwise ray casts toward these objects may fail. However if nothing is selected, we get a runtime error ...
      # and if an object is selected that has no edit mode (e.g. an empty) we get a type error
      try:
          bpy.ops.object.mode_set(mode='EDIT', toggle=False)
          bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
      except RuntimeError:
          pass
      except TypeError:
          pass
      

      if self.useGroups:
          size,minp = groupExtends(self.crownGroup)
          volumefie=partial(groupdistribution,self.crownGroup,self.shadowGroup,self.shadowDensity,self.randomSeed,size,minp-bpy.context.scene.cursor.location)
      else:
          volumefie=partial(ellipsoid2,self.crownSize*self.crownShape,self.crownSize,Vector((0,0,self.crownSize+self.crownOffset)),self.surfaceBias,self.topBias)
      
      startingpoints = []
      if self.useTrunkGroup:
          if self.trunkGroup in bpy.data.collections :
              for ob in bpy.data.collection[self.trunkGroup].objects :
                  p = ob.location - context.scene.cursor.location
                  startingpoints.append(Branchpoint(p,None, 0))
      
      timings.add('scastart')
      start = time()
      sca = SCA(NBP = self.maxIterations,
          NENDPOINTS=self.numberOfEndpoints,
          d=self.internodeLength,
          KILLDIST=self.killDistance,
          INFLUENCE=self.influenceRange,
          SEED=self.randomSeed,
          TROPISM=self.tropism,
          volume=volumefie,
          exclude=lambda p: insidegroup(p, self.exclusionGroup),
          startingpoints=startingpoints,
          apicalcontrol=self.apicalcontrol,
          apicalcontrolfalloff=self.apicalcontrolfalloff,
          apicaltiming=self.apicalcontroltiming
          )
      timings.add('sca')
          
      sca.iterate(newendpointsper1000=self.newEndPointsPer1000,maxtime=self.maxTime)
      timings.add('iterate')
      
      if self.showMarkers:
          mesh = createMarkers(sca, self.markerScale)
          obj_markers = bpy.data.objects.new(mesh.name, mesh)
          base = bpy.context.collection.objects.link(obj_markers)
      timings.add('showmarkers')
      end = time()
      print('SCA Time',end-start)
      start = time()
      obj_new=createGeometry(sca,self.power,self.scale,
          self.noModifiers, self.skinMethod, self.subSurface,
          self.bLeaf, 
        #   self.leafParticles if self.addLeaves else 'None', 
        #   self.objectParticles if self.addLeaves else 'None',
            'None',
            'None',
          self.emitterScale,
          self.timePerformance,
          self.pruningGen)
      
      if obj_new is None:
          return None
      
      # bpy.ops.object.material_slot_add()
      # obj_new.material_slots[-1].material = barkmaterials[self.barkMaterial]
      
      if self.showMarkers:
          obj_markers.parent = obj_new
      
      self.updateTree = False
      
      if self.timePerformance or True:
          timings.add('Total')
          print('geometry timings')
          print(timings)
      
      self.timings = timings
      
      return obj_new
