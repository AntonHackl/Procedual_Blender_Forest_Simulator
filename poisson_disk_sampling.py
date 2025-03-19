from typing import Tuple, List
import numpy as np
import random
import triangle
from shapely.geometry import Polygon, Point

def poisson_disk_sampling_on_surface(surface: List[Tuple[int, int]], configuration_weights, crown_widths, k=30):
  def random_point_in_triangle(v1, v2, v3):
    r1, r2 = random.random(), random.random()
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2
    x = u * v1[0] + v * v2[0] + w * v3[0]
    y = u * v1[1] + v * v2[1] + w * v3[1]
    return (x, y)
  
  def random_point_in_polygon(surface: Polygon):
    for_triangulate = {
      'vertices': surface.exterior.coords[:-1],
      'segments': [[i, (i+1)%(len(surface.exterior.coords)-1)] for i in range(len(surface.exterior.coords)-1)]
    }
    triangulated = triangle.triangulate(for_triangulate, 'p')
    triangles = [[surface.exterior.coords[int(i)] for i in tri] for tri in triangulated['triangles']]
    areas = []
    tri_list = []
    for tri in triangles:
        v1, v2, v3 = tri[0], tri[1], tri[2]
        area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))
        areas.append(area)
        tri_list.append((v1, v2, v3))
    
    chosen_triangle = random.choices(tri_list, weights=areas, k=1)[0]
    
    return random_point_in_triangle(*chosen_triangle)

  def too_near_to_sample(point, points):
    return any(
      np.linalg.norm(np.asarray([point[0][0], point[0][1]]) 
        - np.asarray([[neighbor_point[0][0], neighbor_point[0][1]] for neighbor_point in points]), axis=1) 
      <= np.array([max(crown_widths[neighbor_point[1]], crown_widths[point[1]]) / 2 + 
                   min(crown_widths[neighbor_point[1]], crown_widths[point[1]]) * 0.2 for neighbor_point in points])
    )

  def generate_random_point_around(point, new_configuration_index):
    r1 = random.random()
    r2 = random.random()
    radius = (max(
      crown_widths[point[1]],
      crown_widths[new_configuration_index]
    ) / 2
    + min(
      crown_widths[point[1]],
      crown_widths[new_configuration_index]
    ) * 0.2) * (r1 + 1)
    angle = 2 * np.pi * r2
    position = point[0]
    new_x = position[0] + radius * np.cos(angle)
    new_y = position[1] + radius * np.sin(angle)
    return (new_x, new_y)
  
  def chooseRandomConfiguration():
    return random.choices(range(len(configuration_weights)), weights=configuration_weights, k=1)[0]
  
  active_list: Tuple[Tuple[float, float], int] = []
  points: Tuple[Tuple[float, float], int] = []

  if surface == []:
    return []
  
  polygon = Polygon(surface)
  initial_position = random_point_in_polygon(polygon)
  configuration_index = chooseRandomConfiguration()
  initial_point = (initial_position, configuration_index)
  points.append(initial_point)
  active_list.append(initial_point)

  while active_list:
    idx = random.randint(0, len(active_list) - 1)
    point = active_list[idx]
    found = False
    for _ in range(k):
      new_configuration = chooseRandomConfiguration()
      new_position = generate_random_point_around(point, new_configuration)
      new_point = (new_position, new_configuration)
      if polygon.contains(Point(new_position)) and not too_near_to_sample(new_point, points):
        points.append(new_point)
        active_list.append(new_point)
        found = True
        break
    if not found:
      active_list.pop(idx)

  return points