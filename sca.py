from random import random, seed, expovariate
from functools import partial
from math import sqrt
from time import time
from array import array
from typing import TYPE_CHECKING, Callable, Tuple, Union, Optional, List
from dataclasses import dataclass
from mathutils import Vector
from .edge_index import EdgeIndex
if TYPE_CHECKING:
    from .types_definitions import VolumeGenerator, ExcludeFunction

try:
    from .utilc import closest
except:
    print('utilc.closest() not available, using pure python implementation instead')
    def closest(pos: array, count: array, n: int, x: float, y: float, z: float) -> Tuple[float, int, Tuple[float, float, float]]:
        d2 = 1e30
        ci = 0
        v = (0.0, 0.0, 0.0)
        for i in range(n):
            if count[i] > 1:
                continue
            dx, dy, dz = x - pos[i*3], y - pos[i*3+1], z - pos[i*3+2]
            d = dx*dx + dy*dy + dz*dz
            if d < d2:
                d2 = d
                ci = i
                v = (dx, dy, dz)
        return d2, ci, v

try:
    from .utilc import direction
except:
    print('utilc.direction() not available, using pure python implementation instead')
    def direction(v: array) -> Tuple[Tuple[float, float, float], float]:
        n = len(v) // 3
        x = 0.0
        y = 0.0
        z = 0.0
        for i in range(n):
            x += v[i*3]
            y += v[i*3+1]
            z += v[i*3+2]
        return (x, y, z), x*x + y*y + z*z

class Branchpoint:
    count: int = 0
    
    def __init__(self, p: Union["Vector", Tuple[float, float, float]], parent: Optional[int], generation: int) -> None:
        self.v: "Vector" = Vector(p)
        self.parent: Optional[int] = parent
        self.connections: int = 1
        self.generation: int = generation
        self.apex: Optional["Branchpoint"] = None
        self.shoot: Optional["Branchpoint"] = None
        Branchpoint.count += 1
        self.index: int = Branchpoint.count

    def __str__(self) -> str:
        return str(self.v) + " " + str(self.parent)
        
@dataclass
class GrowthState:
  starttime: float
  niterations: float
  rate: float
  t: float
  maxtime: float
  finished: bool

def sphere(r: float, p: "Vector") -> "VolumeGenerator":
    r2 = r * r
    while True:
        x = (random() * 2 - 1) * r
        y = (random() * 2 - 1) * r
        z = (random() * 2 - 1) * r
        if x*x + y*y + z*z <= r2:
            yield p + Vector((x, y, z))
            
class SCA:

  def __init__(self,
        NENDPOINTS: int = 100,
        d: float = 0.3,
        NBP: int = 2000,
        KILLDIST: float = 5,
        INFLUENCE: float = 15,
        SEED: int = 42,
        volume: Optional["VolumeGenerator"] = None,
        TROPISM: float = 0.0,
        exclude: "ExcludeFunction" = lambda p: False,
        startingpoints: List[Branchpoint] = [],
        apicalcontrol: float = 0,
        apicalcontrolfalloff: float = 1,
        apicaltiming: int = 0,
        tree_id: int = 0,
        edge_index: Optional["EdgeIndex"] = None,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
      ) -> None:
    if volume is None:
       raise ValueError("Volume function is required")
    
    self.killdistance: float = KILLDIST
    self.branchlength: float = d
    self.maxiterations: int = NBP
    self.tropism: float = TROPISM
    self.influence: float = INFLUENCE if INFLUENCE > 0 else 1e16
    self.apicalcontrol: float = apicalcontrol
    self.apicalcontrolfalloff: float = apicalcontrolfalloff
    self.apicaltiming: int = apicaltiming
    self.apicalstep: float = apicalcontrol / apicaltiming if apicaltiming > 0 else 0.0
    
    seed(SEED)
    
    self.bp: array = array('d')
    self.bp.extend((0, 0, 0))
    self.bpg: List[int] = [0]
    self.bpp: List[Optional[int]] = [None]
    self.bpc: array = array('i')
    self.bpc.append(0)
    self.bpa: List[int] = [0]
    self.ep: List[Tuple[float, float, float]] = []
    self.epb: List[int] = []
    self.epv: List[Tuple[float, float, float]] = []
    self.epd: List[float] = []
    
    self.volumepoint: "VolumeGenerator" = volume
    self.exclude: "ExcludeFunction" = exclude
    self.tree_id: int = int(tree_id)
    self.edge_index: Optional["EdgeIndex"] = edge_index
    self.origin: Tuple[float, float, float] = (float(origin[0]), float(origin[1]), float(origin[2]))

    self.branchpoints: List[Branchpoint] = []
    self.endpoints: List["Vector"] = []

    endpoints: List["Vector"] = []
    for _ in range(NENDPOINTS):
      endpoints.append(next(self.volumepoint()))
    for ep in endpoints:
        self.addEndPoint(ep)

    if len(startingpoints) > 0:
        self.bp = array('d')
        self.bpp = []
        self.bpc = array('i')
        for bp in startingpoints:
            self.addBranchPoint(bp.v, -1, 0)

    self._growth_state: Optional[GrowthState] = None

  def addBranchPoint(self, bp: Union["Vector", Tuple[float, float, float]], pi: int, generation: int) -> None:
    self.bp.extend(tuple(bp))
    self.bpg.append(generation)
    ppi: Optional[int] = pi
    while ppi is not None:
        self.bpg[ppi] = generation
        ppi = self.bpp[ppi]
    self.bpp.append(pi)
    self.bpc.append(0)
    self.bpa.append(0)
    self.bpc[pi] += 1
    bi = len(self.bp) // 3 - 1

    for epi, (ep, epd, epb) in enumerate(zip(self.ep, self.epd, self.epb)):
      if epb != -1:
        v = (ep[0] - bp[0], ep[1] - bp[1], ep[2] - bp[2])
        d2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
        d = sqrt(d2)
        if d < epd:
          if d > self.killdistance:
            self.epv[epi] = (v[0] / d, v[1] / d, v[2] / d)
            self.epd[epi] = d
            if d < self.influence:
                self.epb[epi] = bi
            else:
                self.epb[epi] = -2
          else:
            self.epb[epi] = -1
    if self.bpc[pi] > 1:
      for epi, epb in enumerate(self.epb):
        if epb == pi:
          bi, v, d = self.closestBranchPoint(self.ep[epi])
          self.epb[epi] = bi
          self.epv[epi] = v
          self.epd[epi] = d
    self.bpa[pi] += 1
    
  def addEndPoint(self, ep: Union["Vector", Tuple[float, float, float]]) -> None:
    self.ep.append(tuple(ep))
    bi, v, d = self.closestBranchPoint(ep)
    self.epb.append(bi)
    self.epv.append(v)
    self.epd.append(d)

  def closestBranchPoint(self, p: Union["Vector", Tuple[float, float, float]]) -> Tuple[int, Tuple[float, float, float], float]:
    d2, bbi, bv = closest(self.bp, self.bpc, len(self.bp) // 3, p[0], p[1], p[2])
    d = sqrt(d2)
    return bbi if d < self.influence else -2, (bv[0] / d, bv[1] / d, bv[2] / d), d

  def shootSupressed(self, apicalcontrolfactor: int) -> bool:
    """Returns true if a growing shoot should be suppressed."""
    if self.apicalcontrol <= 0:
        return False
    p = 1 - apicalcontrolfactor * self.apicalcontrol
    if p <= 0:
        return True
    p = p ** self.apicalcontrolfalloff
    return random() > p    
    
    
  def growBranches(self, generation: int) -> None:
    bis = set(self.epb)
    bis.discard(-1)
    bis.discard(-2)
    newbps: List[Tuple[float, float, float]] = []
    newbpps: List[int] = []
    for bpi in bis:
      if self.shootSupressed(self.bpa[bpi]):
          continue
      
      epvs = array('d', [c for epi, v in enumerate(self.epv) if self.epb[epi] == bpi for c in v])

      v, d2 = direction(epvs)
      d = sqrt(d2) / self.branchlength
      vd = (v[0] / d, v[1] / d, v[2] / d)

      newbps.append((self.bp[bpi*3] + vd[0], self.bp[bpi*3+1] + vd[1], self.bp[bpi*3+2] + vd[2] + self.tropism))
      newbpps.append(bpi)
    for newbp, newbpp in zip(newbps, newbpps):
      if not self.exclude(Vector(newbp)):
        if self.edge_index is not None and newbpp is not None:
          parent_pt = (self.bp[newbpp*3], self.bp[newbpp*3+1], self.bp[newbpp*3+2])
          gp0 = (parent_pt[0] + self.origin[0], parent_pt[1] + self.origin[1], parent_pt[2] + self.origin[2])
          gp1 = (newbp[0] + self.origin[0], newbp[1] + self.origin[1], newbp[2] + self.origin[2])
          if not self.edge_index.validate_edge(gp0, gp1, self.tree_id):
            continue
        self.addBranchPoint(newbp, newbpp, generation)
        if self.edge_index is not None and newbpp is not None:
          parent_pt = (self.bp[newbpp*3], self.bp[newbpp*3+1], self.bp[newbpp*3+2])
          gp0 = (parent_pt[0] + self.origin[0], parent_pt[1] + self.origin[1], parent_pt[2] + self.origin[2])
          gp1 = (newbp[0] + self.origin[0], newbp[1] + self.origin[1], newbp[2] + self.origin[2])
          self.edge_index.add_edge(gp0, gp1, self.tree_id)

  def nodeRelocation(self) -> None:
    """Move the branchpoints halfway to their parent."""
    relocated_branchpoints: List[float] = []
    for i in range(len(self.bpp)):
      bp_parent_index = self.bpp[i]
      if bp_parent_index is not None:
        relocated_branchpoints.append((self.bp[i*3] + self.bp[bp_parent_index*3]) / 2.0)
        relocated_branchpoints.append((self.bp[i*3+1] + self.bp[bp_parent_index*3+1]) / 2.0)
        relocated_branchpoints.append((self.bp[i*3+2] + self.bp[bp_parent_index*3+2]) / 2.0)
      else:
        relocated_branchpoints.extend(self.bp[i*3:i*3+3])
    self.bp = array('d', relocated_branchpoints)
   
  def iterate(self, newendpointsper1000: float = 0, maxtime: float = 0.0) -> None:
    starttime = time()
    endpointsadded = 0.0
    niterations = 0.0
    newendpointsper1000 /= 1000.0
    t = expovariate(newendpointsper1000) if newendpointsper1000 > 0.0 else 1.0

    for i in range(self.maxiterations):
        self.growBranches(i)
        if maxtime > 0 and time() - starttime > maxtime:
            break
        if newendpointsper1000 > 0.0:
            niterations += 1
            while t < niterations:
                new_point = next(self.volumepoint())
                self.addEndPoint(new_point)
                endpointsadded += 1
                t += expovariate(newendpointsper1000)
        if self.apicaltiming > 0:
            self.apicaltiming -= 1
            self.apicalcontrol -= self.apicalstep
            if self.apicalcontrol < 0:
                self.apicalcontrol = 0.0

    self.finalize_after_growth()

  def finalize_after_growth(self) -> None:
    """Build derived data (branchpoints, connections, endpoints) after growth steps."""
    self.branchpoints = []
    for bi in range(len(self.bp) // 3):
        bp = (self.bp[bi*3], self.bp[bi*3+1], self.bp[bi*3+2])
        bpp = self.bpp[bi]
        gen = self.bpg[bi]
        self.branchpoints.append(Branchpoint(bp, bpp, gen))
        if bpp is not None:
            parent = self.branchpoints[bpp]
            if parent.apex is None:
                parent.apex = self.branchpoints[-1]
            else:
                parent.shoot = self.branchpoints[-1]

    for bp in self.branchpoints:
        bpp = bp
        while bpp.parent is not None:
            bpp = self.branchpoints[bpp.parent]
            bpp.connections += 1

    self.endpoints = []
    for ep in self.ep:
        self.endpoints.append(Vector(ep))

  def begin_growth(self, newendpointsper1000: float = 0, maxtime: float = 0.0) -> None:
    """Initialize state for step-wise growth across generations."""
    rate = (newendpointsper1000 / 1000.0) if newendpointsper1000 > 0.0 else 0.0
    t = expovariate(rate) if rate > 0.0 else 1.0
    self._growth_state = GrowthState(
      starttime=time(),
      niterations=0.0,
      rate=rate,
      t=t,
      maxtime=maxtime,
      finished=False,
    )

  def step_growth(self, generation: int) -> None:
    """Perform a single growth generation. Call begin_growth() first."""
    if self._growth_state is None:
      self.begin_growth(0, 0.0)
    gs = self._growth_state
    if gs is None or gs.finished:
      return
    self.growBranches(generation)
    if gs.maxtime > 0.0 and (time() - gs.starttime) > gs.maxtime:
      gs.finished = True
    if gs.rate > 0.0 and not gs.finished:
      gs.niterations += 1.0
      while gs.t < gs.niterations:
        new_point = next(self.volumepoint())
        self.addEndPoint(new_point)
        gs.t += expovariate(gs.rate)
    if self.apicaltiming > 0:
      self.apicaltiming -= 1
      self.apicalcontrol -= self.apicalstep
      if self.apicalcontrol < 0:
        self.apicalcontrol = 0.0
    if generation >= self.maxiterations - 1:
      gs.finished = True

  def is_finished(self) -> bool:
    return self._growth_state is not None and self._growth_state.finished