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

import sys
# for first time
from collections import defaultdict
from typing import Dict, List

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

import os
import random
from functools import partial
from math import cos, radians, sin
from time import time

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from mathutils import Euler, Matrix, Quaternion, Vector
from scipy.spatial import KDTree

from .leaf_generation import convert_sca_skeleton_to_qsm, generate_foliage
from .sca import (  # the core class that implements the space colonization algorithm and the definition of a segment
    SCA, Branchpoint)
from .timer import Timer
from .utils import (create_inverse_graph, get_vertex_group,
                    load_materials_from_bundled_lib,
                    load_particlesettings_from_bundled_lib)
from .edge_index import EdgeIndex


def availableGroups(self, context):
    return [(name, name, name, n) for n,name in enumerate(bpy.data.collections.keys())]

def availableGroupsOrNone(self, context):
    groups = [ ('None', 'None', 'None', 0) ]
    return groups + [(name, name, name, n+1) for n,name in enumerate(bpy.data.collections.keys())]

def availableObjects(self, context):
    return [(name, name, name, n+1) for n,name in enumerate(bpy.data.objects.keys())]

barkmaterials = None

def availableParticleSettings(self, context, particlesettings):
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
        x = (random.random()*2-1)*r
        y = (random.random()*2-1)*r
        z = (random.random()*2-1)*r
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
        phi = 2*np.pi*random.random()
        theta = np.pi*(random.random()-0.5)
        r = random.random()**((1.0/surfacebias))
        x = r*rxy*cos(theta)*cos(phi)
        y = r*rxy*cos(theta)*sin(phi)
        st=sin(theta)
        st = (((st+1)/2)**(1.0/topbias))*2-1
        z = r*rz*st
        m = p+Vector((x,y,z+rz))
        yield m

def cylinder_points(radius=1, height=2, center=Vector((0,0,0)), surfacebias=1, topbias=1):
    """
    Generate random points inside a cylinder with surface and top bias.
    - radius: cylinder radius
    - height: cylinder height (along Z)
    - center: center of the cylinder (middle of height)
    - surfacebias: >1 = more points near surface, <1 = more points near center
    - topbias: >1 = more points near top/bottom, <1 = more points near middle Z
    """
    pi = np.pi
    while True:
        theta = random.uniform(0, 2*pi)
        r = radius * (random.random() ** (1.0/surfacebias))
        z_frac = (random.random() ** (1.0/topbias))
        z = (z_frac - 0.5) * height  # Range: -height/2 to +height/2
        x = r * cos(theta)
        y = r * sin(theta)
        yield center + Vector((x, y, z+height/2))

def hemisphere_points(radius=1, center=Vector((0,0,0)), surfacebias=1, topbias=1, direction='up'):
    """
    Generate random points inside a hemisphere with surface and top bias.
    - radius: hemisphere radius
    - center: center of the base of the hemisphere
    - surfacebias: >1 = more points near surface, <1 = more points near center
    - topbias: >1 = more points near top, <1 = more points near base
    - direction: 'up' (default) or 'down' for hemisphere orientation
    """
    pi = np.pi
    while True:
        phi = random.uniform(0, 2*pi)
        # theta: 0 (top) to pi/2 (equator/base)
        theta = (random.random() ** (1.0/topbias)) * (pi/2)
        r = radius * (random.random() ** (1.0/surfacebias))
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi)
        z = r * cos(theta)
        if direction == 'down':
            z = -z
        yield center + Vector((x, y, z))

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
                outsideshadow = random.random() > shadowdensity  # if inside the group we might still generate a marker if the density is low
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

def basictri(bp, verts, radii_abs, p):
    if bp.v is None:
        raise RuntimeError(f"Branchpoint {getattr(bp, 'index', -1)} has no position (v=None)")
    v_bp = bp.v if isinstance(bp.v, Vector) else Vector(bp.v)
    v = v_bp + p
    nv = len(verts)
    r = float(radii_abs[bp.index])
    a = -r
    b = r * 0.5   # cos(60)
    c = r * 0.866 # sin(60)
    verts.extend([
        v + Vector((a, 0, 0)),
        v + Vector((b, -c, 0)),
        v + Vector((b,  c, 0)),
    ])
    return (nv, nv+1, nv+2)

def _simpleskin(bp, loop, verts, faces, radii_abs, p):
    newloop = basictri(bp, verts, radii_abs, p)
    for i in range(3):
        faces.append((loop[i],loop[(i+1)%3],newloop[(i+1)%3],newloop[i]))
    if bp.apex:
        _simpleskin(bp.apex, newloop, verts, faces, radii_abs, p)
    if bp.shoot:
        _simpleskin(bp.shoot, newloop, verts, faces, radii_abs, p)

def simpleskin(bp, verts, faces, radii_abs, p):
    loop = basictri(bp, verts, radii_abs, p)
    if bp.apex:
        _simpleskin(bp.apex, loop, verts, faces, radii_abs, p)
    if bp.shoot:
        _simpleskin(bp.shoot, loop, verts, faces, radii_abs, p)

def _basictri_fixed(bp, verts, scale, p):
    v = bp.v + p
    nv = len(verts)
    r = float(scale)
    a = -r
    b = r * 0.5
    c = r * 0.866
    verts.extend([
        v + Vector((a, 0, 0)),
        v + Vector((b, -c, 0)),
        v + Vector((b,  c, 0)),
    ])
    return (nv, nv+1, nv+2)

#TODO: Make it better than just random
def leafnode(bp, verts, faces, radii_unused, p1, p2, scale=0.0001):
    loop1 = _basictri_fixed(bp, verts, scale, p1)
    loop2 = _basictri_fixed(bp, verts, scale, p2)
    # if random() > random_threshold:
    #   for i in range(3):
    #     faces.append((loop1[i],loop1[(i+1)%3],loop2[(i+1)%3],loop2[i]))
    for i in range(3):
        faces.append((loop1[i],loop1[(i+1)%3],loop2[(i+1)%3],loop2[i]))
    if bp.apex:
        leafnode(bp.apex, verts, faces, radii_unused, p1, p2, scale)
    if bp.shoot:
        leafnode(bp.shoot, verts, faces, radii_unused, p1, p2, scale)

def createLeaves2(tree, roots, p, scale):
    verts = []
    faces = []
    radii = []
    for r in roots:
        leafnode(r, verts, faces, radii, p, p+Vector((0,0, scale)), scale)
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
            new_idx = len(nbp)
            bp.index = new_idx
            i2p[i] = new_idx
            nbp.append(bp)
        #print()
    return nbp, i2p

def compute_radii_da_vinci(tree, trunk_radius, radius_exponent, node_to_children, root_idx):
    """
    Compute radii using weighted Da Vinci's rule: r_i = (w_i * r_p^x)^(1/x)
    where w_i is the subtree weight (number of descendant nodes) normalized to sum to 1.
    This makes tips thicker based on their subtree size.
    """
    x = float(radius_exponent)
    if x <= 0.0:
        raise ValueError("radius_exponent must be positive")
    
    n_nodes = len(tree.branchpoints)
    
    # First pass: compute subtree sizes (number of descendant nodes) for each node
    subtree_sizes = [0] * n_nodes
    
    # Post-order traversal to compute subtree sizes
    stack = [root_idx]
    visit_order = []
    while stack:
        node = stack.pop()
        visit_order.append(node)
        stack.extend(node_to_children.get(node, []))
    
    for node in reversed(visit_order):
        children = node_to_children.get(node, [])
        if not children:
            subtree_sizes[node] = 1  # leaf nodes count as 1
        else:
            subtree_sizes[node] = 1 + sum(subtree_sizes[c] for c in children)
    
    # Second pass: top-down distribution using weighted Da Vinci's rule
    radii_abs = [0.0] * n_nodes
    radii_abs[root_idx] = float(trunk_radius)
    
    stack = [root_idx]
    while stack:
        node = stack.pop()
        children = node_to_children.get(node, [])
        if not children:
            continue
        
        # Compute weights based on subtree sizes, normalized to sum to 1
        total_subtree_size = sum(subtree_sizes[c] for c in children)
        weights = [subtree_sizes[c] / total_subtree_size for c in children]
        
        # Apply weighted Da Vinci's rule: r_i = (w_i * r_p^x)^(1/x)
        parent_radius = radii_abs[node]
        for i, child in enumerate(children):
            w_i = weights[i]
            child_radius = (w_i * (parent_radius ** x)) ** (1.0 / x)
            radii_abs[child] = child_radius
            stack.append(child)
    
    return radii_abs


def smooth_radii_along_unary_chains(tree, radii_abs, node_to_children, parent_idx):
    """
    Smooth interpolation along unary chains to create gradual transitions.
    For each branching node, interpolate along upstream unary chains.
    """
    n_nodes = len(tree.branchpoints)
    
    # For each branching node, interpolate along upstream unary chains
    for b in range(n_nodes):
        children = node_to_children.get(b, [])
        if len(children) < 2:
            continue
        
        # Find upstream unary chain ending at the parent of this branch node
        end_node = parent_idx[b]
        if end_node is None:
            continue
        chain_nodes: list[int] = [b]
        node = end_node
        while node is not None and len(node_to_children.get(node, [])) == 1:
            chain_nodes.append(node)
            node = parent_idx[node]

        if not chain_nodes:
            continue

        chain_nodes = list(reversed(chain_nodes))  # from start -> end

        # Compute distances along the chain to weight interpolation
        dists: list[float] = [0.0] * len(chain_nodes)
        cum = 0.0
        for i in range(1, len(chain_nodes)):
            a = tree.branchpoints[chain_nodes[i-1]].v
            c = tree.branchpoints[chain_nodes[i]].v
            if a is None or c is None:
                seglen = 0.0
            else:
                va = a if isinstance(a, Vector) else Vector(a)
                vc = c if isinstance(c, Vector) else Vector(c)
                seglen = (vc - va).length
            cum += float(seglen)
            dists[i] = cum
        total = dists[-1] if dists else 0.0

        if total <= 0.0:
            # fallback: uniform steps
            total = float(max(1, len(chain_nodes)-1))
            dists = [float(i) for i in range(len(chain_nodes))]

        start_radius = radii_abs[chain_nodes[0]]
        # End radius should be the maximum radius of children at the branching node that starts the chain
        end_radius = max(radii_abs[c] for c in children)

        # Interpolate from start_radius to end_radius along the chain
        for i, node in enumerate(chain_nodes):
            t = (dists[i] / total) if total > 0 else 1.0
            # cosine-eased interpolation for smoother falloff
            t_ease = 0.5 - 0.5 * cos(np.pi * t)
            desired = (1.0 - t_ease) * start_radius + t_ease * end_radius
            # Only reduce radius towards desired to avoid conflicts if chains overlap
            radii_abs[node] = min(radii_abs[node], desired)
    
    return radii_abs


def calculate_leaf_area_from_allometry(trunk_radius, tree_height, a=0.2, b=2.2, c=0.5):
    """
    Calculate leaf area using allometric formula: a * (DBH^b) * (H^c)
    
    Parameters:
    - trunk_radius: radius of the trunk (in meters)
    - tree_height: height of the tree (in meters)
    - a: allometric coefficient (default: 0.2)
    - b: DBH exponent (default: 2.2)
    - c: height exponent (default: 0.5)
    
    Returns:
    - leaf_area: calculated leaf area
    """
    # Convert radius to DBH (Diameter at Breast Height)
    dbh = trunk_radius * 2.0
    
    # Apply allometric formula
    leaf_area = a * (dbh ** b) * (tree_height ** c) * 100
    
    return leaf_area


def get_trunk_nodes(branchpoints):
    """
    Get all trunk nodes by finding the top of the trunk and tracing back through the parent chain.
    Returns a list of trunk nodes from top to root.
    """
    if not branchpoints:
        return []
    
    # Find trunk nodes using the same logic as segmentIntoTrunkAndBranch
    top = find_top_of_trunk(branchpoints)
    trunk_nodes = [top]
    
    # Trace back through parent chain to get all trunk nodes
    while trunk_nodes[-1].parent is not None:
        trunk_nodes.append(branchpoints[trunk_nodes[-1].parent])
    
    return trunk_nodes


def calculate_actual_tree_height(tree):
    """
    Calculate the actual height of the tree from the trunk nodes only.
    Returns the height as the difference between the highest and lowest z-coordinates of trunk nodes.
    """
    if not tree.branchpoints:
        return 0.0
    
    # Get trunk nodes
    trunk_nodes = get_trunk_nodes(tree.branchpoints)
    
    # Extract z-coordinates from trunk nodes only
    trunk_z_coords = []
    for trunk_node in trunk_nodes:
        if trunk_node.v is not None:
            v_bp = trunk_node.v if isinstance(trunk_node.v, Vector) else Vector(trunk_node.v)
            trunk_z_coords.append(v_bp.z)
    
    if not trunk_z_coords:
        return 0.0
    
    return max(trunk_z_coords) - min(trunk_z_coords)


def compute_tree_radii(tree, trunk_radius, radius_exponent, index2position, expected_height=None):
    """
    Main function to compute tree radii using Da Vinci's law and smoothing.
    Applies allometric scaling based on actual vs expected tree height.
    Returns the computed radii array and the scaled trunk radius.
    """
    n_nodes = len(tree.branchpoints)
    node_to_children = create_inverse_graph(tree.branchpoints)

    # Fallback: ensure every node key exists
    for i in range(n_nodes):
        if i not in node_to_children:
            node_to_children[i] = []

    # Find root index in current, pruned list
    root_idx = next((bp.index for bp in tree.branchpoints if bp.parent is None), 0)

    # Apply allometric scaling if expected height is provided
    scaled_trunk_radius = trunk_radius
    if expected_height is not None and expected_height > 0:
        actual_height = calculate_actual_tree_height(tree)
        if actual_height > 0:
            # Allometric scaling: new_radius = trunk_radius * (height / expected_height)^0.75
            height_ratio = actual_height / expected_height
            scaled_trunk_radius = trunk_radius * (height_ratio ** 0.75)
            print(f"Tree height scaling: expected={expected_height:.2f}, actual={actual_height:.2f}, "
                  f"original_radius={trunk_radius:.4f}, scaled_radius={scaled_trunk_radius:.4f}")

    # Compute radii using Da Vinci's law
    radii_abs = compute_radii_da_vinci(tree, scaled_trunk_radius, radius_exponent, node_to_children, root_idx)

    # Build parent index map in current pruned indexing
    parent_idx: list[int | None] = [None] * n_nodes
    for n, bp in enumerate(tree.branchpoints):
        parent_idx[n] = None if (bp.parent is None) else index2position.get(bp.parent, None)

    # Apply smoothing along unary chains
    radii_abs = smooth_radii_along_unary_chains(tree, radii_abs, node_to_children, parent_idx)
    
    return radii_abs, scaled_trunk_radius


def createGeometry(tree,
    nomodifiers=True, skinmethod='NATIVE', subsurface=False,
    bleaf=4.0,
    leafParticles='None',
    particlesettings=None,
    objectParticles='None',
    emitterscale=0.1,
    timeperf=True,
    addLeaves=False,
    prune=0,
    class_id=0,
    radius_exponent=2.0,
    trunk_radius=0.25,
    taper_per_meter=0.02,
    stem_height=0.0,
    crown_height=0.0,
    crown_offset=0.0):

    if particlesettings is None and leafParticles != 'None':
        raise ValueError("No particlesettings available, cannot create leaf particles")

    timings = Timer()

    tree_position=bpy.context.scene.cursor.location.copy()
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
        if bp.v is None:
            raise RuntimeError(f"Branchpoint {n} has no position (v=None)")
        v_bp = bp.v if isinstance(bp.v, Vector) else Vector(bp.v)
        verts.append(v_bp + tree_position)
        # placeholder; will be computed from children using the radius rule
        radii.append(0.0)
        if not (bp.parent is None) :
            parent_mapped = index2position.get(bp.parent, None)
            if parent_mapped is not None:
                edges.append((len(verts)-1, parent_mapped))
        else :
            nv=len(verts)
            roots.add(bp)
        bp.index=n

    timings.add('skeleton')

    # Calculate expected height: stem_height + crown_height - crown_offset
    expected_height = stem_height + crown_height - crown_offset

    # Compute tree radii using Da Vinci's law and smoothing with allometric scaling
    radii_abs, scaled_trunk_radius = compute_tree_radii(tree, trunk_radius, radius_exponent, index2position, expected_height)

    # native skinning method
    if nomodifiers == False and skinmethod == 'NATIVE':
        # add a quad edge loop to all roots
        for r in roots:
            simpleskin(r, verts, faces, radii_abs, tree_position)

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

            bpy.ops.object.modifier_add(type='SKIN')
            bpy.context.active_object.modifiers[-1].use_smooth_shade=True
            bpy.context.active_object.modifiers[-1].use_x_symmetry=True
            bpy.context.active_object.modifiers[-1].use_y_symmetry=True
            bpy.context.active_object.modifiers[-1].use_z_symmetry=True

            skinverts = bpy.context.active_object.data.skin_vertices[0].data

            for i,v in enumerate(skinverts):
                vert_radius = radii_abs[i]
                v.radius = [vert_radius, vert_radius]
                if i in roots:
                    v.use_root = True

            bpy.ops.object.modifier_add(type='SUBSURF')
            bpy.context.active_object.modifiers[-1].levels = 1
            bpy.context.active_object.modifiers[-1].render_levels = 2

    timings.add('modifiers')
    # create a particles based leaf emitter (if we have leaves and/or objects)
    # bpy.context.scene.objects.active = obj_new
    obj_processed = segmentIntoTrunkAndBranch(tree, obj_new, np.array(radii_abs))
    bpy.ops.object.shade_smooth()

    obj_processed["class_id"] = class_id
    obj_processed.name = f"Tree_{obj_processed['class_id']}"

    converted_qsm = convert_sca_skeleton_to_qsm(tree, np.array(radii_abs))
    qsm_path = os.path.join(os.path.dirname(__file__), 'leafgen', 'src', 'example-data', 'generated_tree.mat')
    
    # Read leaf parameters from the tree configuration json (defaults provided upstream)
    leaf_params = getattr(tree, 'leaf_params', None)
    if leaf_params is None:
        leaf_params = {}
    else:
        leaf_params = dict(leaf_params)  # Make a copy to avoid modifying the original
    
    # Calculate leaf area using allometric formula based on trunk radius and tree height
    # Use the scaled trunk radius from the allometric scaling and the expected height
    calculated_leaf_area = calculate_leaf_area_from_allometry(
        trunk_radius=scaled_trunk_radius,
        tree_height=expected_height
    )
    
    # Override the totalLeafArea parameter with the calculated value
    leaf_params['totalLeafArea'] = calculated_leaf_area
    print(f"Calculated leaf area: {calculated_leaf_area:.2f} for tree with trunk_radius={trunk_radius:.4f}, scaled_radius={scaled_trunk_radius:.4f}, height={expected_height:.2f}")
    
    generate_foliage(converted_qsm, qsm_path, execute_matlab=True, leaf_params=leaf_params)
    timings.add('leaves')

    if timeperf:
        print(timings)

    # bpy.data.objects.remove(obj_new, do_unlink=True)
    return obj_processed

# This method is currently not being used.
def add_leaves_to_tree(tree, leave_nodes, obj_new):
    # Create a new mesh for the leaves
    leaf_mesh = bpy.data.meshes.new("Leaves")
    leaf_verts = []
    leaf_faces = []

    # uv_layer = leaf_mesh.loops.layers.uv.new()

    for leave_node in leave_nodes:
        pos = leave_node.v
        direction = (pos - tree.branchpoints[leave_node.parent].v).normalized() if leave_node.parent is not None else Vector((0, 0, 1))

        # First quad
        v1 = Vector((-0.1,-0.1,0))
        v2 = Vector((0.1,-0.1,0))
        v3 = Vector((0.1,0.1,0))
        v4 = Vector((-0.1,0.1,0))

        # Rotate the second quad vertices 90° from the direction vector
        # current_direction = Vector((0, 0, 1))  # Assuming the initial direction is along the Z-axis
        # rotation = current_direction.rotation_difference(direction)

        axis = Vector((0, 0, 1))  # Z-axis
        angle = radians(90)  # Convert degrees to radians

        # Create a rotation matrix
        rotation_matrix = Vector((0, 0, 1)).rotation_difference(direction).to_matrix().to_4x4()
        rotation_matrix = Matrix.Rotation(angle, 4, axis) @ rotation_matrix

        v1 = rotation_matrix @ v1
        v2 = rotation_matrix @ v2
        v3 = rotation_matrix @ v3
        v4 = rotation_matrix @ v4

        v1 += pos
        v2 += pos
        v3 += pos
        v4 += pos

        leaf_verts.extend([v1, v2, v3, v4])
        start_index = len(leaf_verts) - 4
        # Faces for the two quads
        leaf_faces.append((start_index + 0, start_index + 1, start_index + 2, start_index + 3))

    # Assign vertices and faces to the leaf mesh
    leaf_mesh.from_pydata(leaf_verts, [], leaf_faces)
    leaf_mesh.update()
    uv_layer = leaf_mesh.uv_layers.new(name="UVMap")
    uv_data = uv_layer.data
    for face in leaf_mesh.polygons:
        for loop_index, uv in zip(range(face.loop_start, face.loop_start + face.loop_total), [(0, 0), (1, 0), (1, 1), (0, 1)]):
            uv_data[loop_index].uv = uv

    # Create a new material
    mat = bpy.data.materials.new(name="LeafMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")

    # Load image
    image_path = "C:/Users/anton/Documents/Uni/Spatial Data Analysis/Procedual_Blender_Forest_Simulator/textures/chestnut_summer_color.png"  # Replace with your image path
    image = bpy.data.images.load(image_path)

    # Create texture node
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = image

    # Connect the texture to the base color
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # Create a new object for the leaves
    leaf_obj = bpy.data.objects.new("Leaves", leaf_mesh)

    if leaf_obj.data.materials:
        leaf_obj.data.materials[0] = mat
    else:
        leaf_obj.data.materials.append(mat)

    # Link the leaf object to the same collection as obj_new
    bpy.context.view_layer.active_layer_collection.collection.objects.link(leaf_obj)

    # Parent the leaves to the tree object
    leaf_obj.parent = obj_new

def segmentIntoTrunkAndBranch(tree, obj_new, radii):
    # Get trunk nodes using the shared function
    trunk_nodes = get_trunk_nodes(tree.branchpoints)
    trunk_indices = [trunk_node.index for trunk_node in trunk_nodes]

    trunk_node_positions = [trunk_node.v for trunk_node in trunk_nodes]
    branch_node_positions = [bp.v for bp in tree.branchpoints if bp not in trunk_nodes and bp.apex is not None]
    leave_nodes = [bp for bp in tree.branchpoints if bp not in trunk_nodes and bp.apex is None]
    branch_node_indices = [i for i in range(len(tree.branchpoints)) if i not in trunk_indices]

    trunk_material = create_material("TrunkMaterial", (0.77, 0.64, 0.52, 1), 2) # light brown
    branch_material = create_material("BranchMaterial", (0.36, 0.25, 0.20, 1), 3) # dark brown
    assign_material(obj_new, trunk_material)
    assign_material(obj_new, branch_material)
    trunk_vertex_indices = []
    branch_vertex_indices = []

    # maybe unnecessary
    bpy.context.view_layer.objects.active = obj_new
    obj_new.select_set(True)

    bpy.ops.object.modifier_apply(modifier="Subdivision")

    # obj_new.data = final_mesh

    trunk_node_kd_tree = KDTree(trunk_node_positions)
    branch_node_kd_tree = KDTree(branch_node_positions)
    for poly in obj_new.data.polygons:
        position = poly.center
        trunk_node_distance, trunk_node_index = trunk_node_kd_tree.query(position, 1)
        branch_node_distance, branch_node_index = branch_node_kd_tree.query(position, 1)

        if trunk_node_distance - radii[trunk_indices[trunk_node_index]] < branch_node_distance - radii[branch_node_indices[branch_node_index]]:
            poly.material_index = 0
        else:
            poly.material_index = 1

    assign_vertices_to_group(obj_new, "TrunkGroup", trunk_vertex_indices)
    assign_vertices_to_group(obj_new, "BranchGroup", branch_vertex_indices)

    obj_new.data.update()

    return obj_new

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

def create_material(name, color, pass_index):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.diffuse_color = color
        mat.pass_index = pass_index
    else:
        mat.diffuse_color = color
        mat.pass_index = pass_index

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
                class_id=0,
                interNodeLength=0.25,
                killDistance=0.1,
                influenceRange=15.,
                tropism=0.,
                useGroups=False,
                crownGroup='None',
                shadowGroup='None',
                crown_type='ellipsoid',
                crown_height=1.0,
                crown_width=1.0,
                crown_offset=0.0,
                stem_height=0.0,
                stem_diameter=0.0,
                shadowDensity=0.5,
                exclusionGroup='None',
                useTrunkGroup=False,
                trunkGroup=None,
                surface_bias=1.,
                top_bias=1.,
                randomSeed=0,
                maxIterations=40,
                pruningGen=0,
                numberOfEndpoints=100,
                newEndPointsPer1000=0,
                maxTime=0.0,
                bLeaf=4.0,
                addLeaves=False,
                emitterScale=0.01,
                noModifiers=True,
                subSurface=False,
                showMarkers=False,
                markerScale=0.05,
                timePerformance=False,
                apicalcontrol=0.0,
                apicalcontrolfalloff=1.0,
                apicalcontroltiming=10,
                context=None,
                trunk_radius=0.25,
                ):
        self.class_id = class_id
        self.internodeLength = interNodeLength
        self.killDistance = killDistance
        self.influenceRange = influenceRange
        self.tropism = tropism
        self.trunk_radius = float(trunk_radius)
        self.useGroups = useGroups
        self.crownGroup = crownGroup
        self.shadowGroup = shadowGroup
        self.shadowDensity = shadowDensity
        self.exclusionGroup = exclusionGroup
        self.useTrunkGroup = useTrunkGroup
        self.trunkGroup = trunkGroup
        self.crown_height = crown_height
        self.crown_width = crown_width
        self.crown_offset = crown_offset
        self.surface_bias = surface_bias
        self.top_bias = top_bias
        self.randomSeed = randomSeed
        self.maxIterations = maxIterations
        self.pruningGen = pruningGen
        self.numberOfEndpoints = numberOfEndpoints
        self.crown_type = crown_type
        self.stem_height = stem_height
        self.stem_diameter = stem_diameter
        self.newEndPointsPer1000 = newEndPointsPer1000
        self.maxTime = maxTime
        self.bLeaf = bLeaf
        self.addLeaves = addLeaves
        self.addLeaves = True

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

    def _prepare_common(self, context, edge_index: EdgeIndex):
        global barkmaterials
        barkmaterials = load_materials_from_bundled_lib('Procedual_Blender_Forest_Simulator', 'material_lib.blend', 'Bark')

        particlesettings = load_particlesettings_from_bundled_lib('Procedual_Blender_Forest_Simulator', 'material_lib.blend', 'LeafEmitter')
        bpy.types.MESH_OT_forest_generator.particlesettings = particlesettings

        self.leafParticles = availableParticleSettings(self, context, particlesettings)[9]

        timings = Timer()

        try:
            bpy.ops.object.mode_set(mode='EDIT', toggle=False)
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        except RuntimeError:
            pass
        except TypeError:
            pass

        if self.crown_type == 'ellipsoid':
            volumefie = partial(ellipsoid2, self.crown_width, self.crown_height, Vector((0, 0, self.stem_height - self.crown_offset)), self.surface_bias, self.top_bias)
        elif self.crown_type == 'columnar':
            volumefie = partial(cylinder_points, self.crown_width, self.crown_height, Vector((0, 0, self.stem_height - self.crown_offset)), self.surface_bias, self.top_bias)
        elif self.crown_type == 'spreading':
            volumefie = partial(hemisphere_points, self.crown_width, Vector((0, 0, self.stem_height - self.crown_offset)), self.surface_bias, self.top_bias)
        else:
            raise ValueError(f"Invalid crown type: {self.crown_type}")

        startingpoints = []
        if self.useTrunkGroup:
            if self.trunkGroup in bpy.data.collections:
                for ob in bpy.data.collection[self.trunkGroup].objects:
                    p = ob.location - context.scene.cursor.location
                    startingpoints.append(Branchpoint(p, None, 0))

        # register per-tree min edge distance
        edge_index.set_tree_min_distance(self.class_id, 0.8)

        origin_loc = bpy.context.scene.cursor.location
        sca = SCA(NBP=self.maxIterations,
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
            apicaltiming=self.apicalcontroltiming,
            tree_id=self.class_id,
            edge_index=edge_index,
            origin=(float(origin_loc.x), float(origin_loc.y), float(origin_loc.z))
        )

        return sca, particlesettings, timings

    def prepare_growth(self, context, edge_index):
        timings = Timer()
        timings.add('scastart')
        sca, particlesettings, base_timings = self._prepare_common(context, edge_index)
        timings.add('sca')

        # initialize step-wise growth
        sca.begin_growth(newendpointsper1000=self.newEndPointsPer1000, maxtime=self.maxTime)

        self._prepared = {
            'sca': sca,
            'particlesettings': particlesettings,
            'timings': timings,
        }
        return sca

    def finalize_tree(self, context):
        if not hasattr(self, '_prepared'):
            return None

        sca = self._prepared['sca']
        particlesettings = self._prepared['particlesettings']
        timings = self._prepared['timings']

        sca.finalize_after_growth()

        if self.showMarkers:
            mesh = createMarkers(sca, self.markerScale)
            obj_markers = bpy.data.objects.new(mesh.name, mesh)
            base = bpy.context.collection.objects.link(obj_markers)
        timings.add('showmarkers')

        self.leafParticles = next((k for k in particlesettings.keys() if k.startswith('LeavesAbstractSummer')), 'None')

        obj_new = createGeometry(
            sca,
            self.noModifiers, self.skinMethod, self.subSurface,
            self.bLeaf,
            self.leafParticles,
            particlesettings if self.addLeaves else 'None',
            'None',
            self.emitterScale,
            self.timePerformance,
            self.pruningGen,
            class_id=self.class_id,
            radius_exponent=2.0,
            trunk_radius=getattr(self, 'trunk_radius', 0.25),
            stem_height=getattr(self, 'stem_height', 0.0),
            crown_height=getattr(self, 'crown_height', 0.0),
            crown_offset=getattr(self, 'crown_offset', 0.0)
        )

        if obj_new is None:
            return None

        if self.showMarkers:
            obj_markers.parent = obj_new

        self.updateTree = False

        if self.timePerformance:
            timings.add('Total')
            print(timings)

        self.timings = timings
        return obj_new

    # removed legacy single-tree generation
