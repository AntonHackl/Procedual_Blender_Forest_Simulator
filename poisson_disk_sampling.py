from typing import Tuple, List, Union, Dict, Any
import numpy as np
import random
import bpy
from mathutils import Vector

def poisson_disk_sampling_on_surface(
  surface: bpy.types.Object,
  configuration_weights: List[float],
  base_configurations: List[Dict[str, Any]],
  k: int = 30
) -> List[Tuple[Tuple[float, float, float], Dict[str, Any]]]:
  """
  Generate Poisson-disc distributed tree positions on a terrain mesh.

  Approach: run 2D Poisson-disc in XY over the mesh's world-space bounding box,
  then ray-cast from above to determine Z on the mesh. Distance constraints use
  a configuration-dependent radius derived from crown widths.

  - surface: Blender mesh object (ANT Landscape terrain)
  - configuration_weights: sampling weights for choosing a tree configuration per point
  - base_configurations: list of tree configuration dicts (values may be [mean, std])
  - k: attempts per active point
  Returns: list of ((x, y, z), sampled_config_dict)
  """
  


  if not surface or surface.type != 'MESH':
    return []

  obj: bpy.types.Object = surface
  mw = obj.matrix_world
  verts_world: List[Vector] = [mw @ v.co for v in obj.data.vertices]
  if not verts_world:
    return []

  min_x = min(v.x for v in verts_world)
  max_x = max(v.x for v in verts_world)
  min_y = min(v.y for v in verts_world)
  max_y = max(v.y for v in verts_world)
  max_z = max(v.z for v in verts_world)

  # Sampling helpers mirror logic from caller
  _int_keys = {"numberOfEndpoints", "maxIterations"}
  _positive_float_keys = {
    "interNodeLength", "influenceRange", "stem_height",
    "stem_diameter", "crown_width", "crown_height", "surface_bias",
    "top_bias", "trunk_radius"
  }

  def _is_mean_std_list(value: Any) -> bool:
    try:
      return isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value)
    except Exception:
      return False

  def _sample_value(key: str, value: float | int | str | dict) -> Any:
    if isinstance(value, dict):
      return value
    if isinstance(value, str):
      return value
    if _is_mean_std_list(value):
      mean, std = float(value[0]), max(float(value[1]), 0.0)
      sampled = random.gauss(mean, std) if std > 0 else mean
    else:
      sampled = value
    if key in _int_keys:
      try:
        sampled = int(round(float(sampled)))
      except Exception:
        sampled = int(0)
      if key == "numberOfEndpoints":
        sampled = max(1, sampled)
      if key == "maxIterations":
        sampled = max(1, sampled)
      return sampled
    if key in _positive_float_keys:
      try:
        sampled = float(sampled)
      except Exception:
        sampled = 0.0
      sampled = max(1e-8, sampled)
      return sampled
    return sampled

  def _sample_configuration(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sampled: Dict[str, Any] = {}
    for k, v in cfg.items():
      if k == 'leaf_params':
        sampled[k] = v
      else:
        sampled[k] = _sample_value(k, v)
    return sampled

  def choose_config() -> Dict[str, Any]:
    idx = random.choices(range(len(configuration_weights)), weights=configuration_weights, k=1)[0]
    return _sample_configuration(base_configurations[idx])

  def min_distance(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any]) -> float:
    cw_a = float(cfg_a.get('crown_width', 1.0))
    cw_b = float(cfg_b.get('crown_width', 1.0))
    return max(cw_a, cw_b) / 2.0 + min(cw_a, cw_b) * 0.2

  def generate_candidate_xy(center_xy: Tuple[float, float], base_r: float) -> Tuple[float, float]:
    r = base_r * (1.0 + random.random())  # between R and 2R
    theta = 2.0 * np.pi * random.random()
    return (center_xy[0] + r * np.cos(theta), center_xy[1] + r * np.sin(theta))

  def raycast_to_surface(x: float, y: float) -> Union[Tuple[float, float, float], None]:
    origin = Vector((x, y, max_z + 100.0))
    direction = Vector((0.0, 0.0, -1.0))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    hit, location, normal, index = eval_obj.ray_cast(origin, direction)
    if hit:
      return (float(location.x), float(location.y), float(location.z))
    return None

  # storage with type annotations
  active_list: List[Tuple[Tuple[float, float, float], Dict[str, Any]]] = []
  points: List[Tuple[Tuple[float, float, float], Dict[str, Any]]] = []

  # seed with an initial valid point
  for _ in range(1000):
    sx = random.uniform(min_x, max_x)
    sy = random.uniform(min_y, max_y)
    hit = raycast_to_surface(sx, sy)
    if hit is None:
      continue
    ci = choose_config()
    initial = (hit, ci)
    points.append(initial)
    active_list.append(initial)
    break
  if not active_list:
    return []

  def too_near(new_point: Tuple[Tuple[float, float, float], Dict[str, Any]], others: List[Tuple[Tuple[float, float, float], Dict[str, Any]]]) -> bool:
    if not others:
      return False
    p = np.asarray(new_point[0])
    q = np.asarray([np.array([pp[0][0], pp[0][1], pp[0][2]]) for pp in others])
    dists = np.linalg.norm(q - p, axis=1)
    cfg_new = new_point[1]
    radii = [min_distance(cfg_new, pp[1]) for pp in others]
    return np.any(dists <= np.array(radii))

  while active_list:
    idx = random.randint(0, len(active_list) - 1)
    point = active_list[idx]
    found = False
    center_xy = (point[0][0], point[0][1])
    for _ in range(k):
      new_cfg = choose_config()
      base_r = min_distance(point[1], new_cfg)
      cx, cy = generate_candidate_xy(center_xy, base_r)
      if cx < min_x or cx > max_x or cy < min_y or cy > max_y:
        continue
      hit = raycast_to_surface(cx, cy)
      if hit is None:
        continue
      new_point = (hit, new_cfg)
      if not too_near(new_point, points):
        points.append(new_point)
        active_list.append(new_point)
        found = True
        break
    if not found:
      active_list.pop(idx)

  return points[:7]


def poisson_disk_sampling_low_vegetation(
    surface: bpy.types.Object,
    density: float = 1.0,
    k: int = 30
) -> List[Tuple[float, float, float]]:
    """
    Generate Poisson-disc distributed low vegetation positions on a terrain mesh.
    
    - surface: Blender mesh object (terrain)
    - density: density factor for vegetation placement (higher = more dense)
    - k: attempts per active point
    Returns: list of (x, y, z) positions
    """
    
    if not surface or surface.type != 'MESH':
        return []

    obj: bpy.types.Object = surface
    mw = obj.matrix_world
    verts_world: List[Vector] = [mw @ v.co for v in obj.data.vertices]
    if not verts_world:
        return []

    min_x = min(v.x for v in verts_world)
    max_x = max(v.x for v in verts_world)
    min_y = min(v.y for v in verts_world)
    max_y = max(v.y for v in verts_world)
    max_z = max(v.z for v in verts_world)

    # Base radius for low vegetation (adjust based on typical vegetation size)
    base_radius = 0.5 / density  # Smaller radius for higher density

    def generate_candidate_xy(center_xy: Tuple[float, float], r: float) -> Tuple[float, float]:
        radius = r * (1.0 + random.random())  # between R and 2R
        theta = 2.0 * np.pi * random.random()
        return (center_xy[0] + radius * np.cos(theta), center_xy[1] + radius * np.sin(theta))

    def raycast_to_surface(x: float, y: float) -> Union[Tuple[float, float, float], None]:
        origin = Vector((x, y, max_z + 100.0))
        direction = Vector((0.0, 0.0, -1.0))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        hit, location, normal, index = eval_obj.ray_cast(origin, direction)
        if hit:
            return (float(location.x), float(location.y), float(location.z))
        return None

    # Storage
    active_list: List[Tuple[float, float, float]] = []
    points: List[Tuple[float, float, float]] = []

    # Seed with an initial valid point
    for _ in range(1000):
        sx = random.uniform(min_x, max_x)
        sy = random.uniform(min_y, max_y)
        hit = raycast_to_surface(sx, sy)
        if hit is None:
            continue
        points.append(hit)
        active_list.append(hit)
        break
    
    if not active_list:
        return []

    def too_near(new_point: Tuple[float, float, float], others: List[Tuple[float, float, float]]) -> bool:
        if not others:
            return False
        p = np.asarray(new_point)
        q = np.asarray(others)
        dists = np.linalg.norm(q - p, axis=1)
        return np.any(dists <= base_radius)

    while active_list:
        idx = random.randint(0, len(active_list) - 1)
        point = active_list[idx]
        found = False
        center_xy = (point[0], point[1])
        
        for _ in range(k):
            cx, cy = generate_candidate_xy(center_xy, base_radius)
            if cx < min_x or cx > max_x or cy < min_y or cy > max_y:
                continue
            hit = raycast_to_surface(cx, cy)
            if hit is None:
                continue
            if not too_near(hit, points):
                points.append(hit)
                active_list.append(hit)
                found = True
                break
        if not found:
            active_list.pop(idx)

    return points