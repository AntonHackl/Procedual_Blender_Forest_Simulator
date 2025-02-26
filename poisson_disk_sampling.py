from typing import Tuple, List
from functools import partial
from math import sin,cos
import numpy as np
import random
import triangle
from shapely.geometry import Polygon, Point
from shapely.ops import triangulate

def poisson_disk_sampling_on_surface(surface: List[Tuple[int, int]], radius: float, k=30):
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
  
  active_list = []
  points = []

  if surface == []:
    return []
  polygon = Polygon(surface)
  initial_point = random_point_in_polygon(polygon)
  points.append(initial_point)
  active_list.append(initial_point)

  while active_list:
    idx = random.randint(0, len(active_list) - 1)
    point = active_list[idx]
    found = False
    for _ in range(k):
      new_point = generate_random_point_around(point, radius)
      if polygon.contains(Point(new_point)) and not in_circle(new_point, radius, points):
        points.append(new_point)
        active_list.append(new_point)
        found = True
        break
    if not found:
      active_list.pop(idx)

  return points