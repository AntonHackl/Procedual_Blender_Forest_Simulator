"""
Type definitions for the Procedural Blender Forest Simulator.

This module contains TypedDict classes, Protocol classes, and type aliases
to provide complete type coverage without using Any types.
"""

from typing import TYPE_CHECKING, Protocol, TypedDict, Union, List, Tuple, Callable, Optional, Any as AnyType
from typing_extensions import NotRequired

if TYPE_CHECKING:
    import bpy.types
    import mathutils
    from mathutils import Vector
    import matlab.engine


class LeafParamsDict(TypedDict, total=False):
    """Leaf generation parameters for MATLAB foliage generation."""
    pLADDh: List[float]
    pLADDd: List[float]
    fun_pLSD: List[float]
    totalLeafArea: float


class TreeConfigDict(TypedDict, total=False):
    """Complete tree configuration dictionary loaded from JSON."""
    class_id: int
    interNodeLength: Union[float, List[float]]
    killDistance: Union[float, List[float]]
    influenceRange: Union[float, List[float]]
    tropism: Union[float, List[float]]
    useGroups: bool
    crownGroup: str
    shadowGroup: str
    crown_type: str
    crown_height: Union[float, List[float]]
    crown_width: Union[float, List[float]]
    crown_offset: Union[float, List[float]]
    stem_height: Union[float, List[float]]
    stem_diameter: Union[float, List[float]]
    shadowDensity: Union[float, List[float]]
    exclusionGroup: str
    useTrunkGroup: bool
    trunkGroup: Optional[str]
    surface_bias: Union[float, List[float]]
    top_bias: Union[float, List[float]]
    randomSeed: int
    maxIterations: Union[int, List[int]]
    pruningGen: int
    numberOfEndpoints: Union[int, List[int]]
    newEndPointsPer1000: Union[float, List[float]]
    maxTime: float
    bLeaf: float
    addLeaves: bool
    emitterScale: float
    noModifiers: bool
    subSurface: bool
    showMarkers: bool
    markerScale: float
    timePerformance: bool
    apicalcontrol: Union[float, List[float]]
    apicalcontrolfalloff: Union[float, List[float]]
    apicalcontroltiming: Union[int, List[int]]
    trunk_radius: Union[float, List[float]]
    trunk_radius_scaling: Union[float, List[float]]
    leaf_area_scaling: Union[float, List[float]]
    leaf_area_dbh_scaling: Union[float, List[float]]
    leaf_area_height_scaling: Union[float, List[float]]
    leaf_params: NotRequired[LeafParamsDict]


class SampledConfigDict(TypedDict, total=False):
    """Tree configuration with all values sampled (no lists, only concrete values)."""
    class_id: int
    interNodeLength: float
    killDistance: float
    influenceRange: float
    tropism: float
    useGroups: bool
    crownGroup: str
    shadowGroup: str
    crown_type: str
    crown_height: float
    crown_width: float
    crown_offset: float
    stem_height: float
    stem_diameter: float
    shadowDensity: float
    exclusionGroup: str
    useTrunkGroup: bool
    trunkGroup: Optional[str]
    surface_bias: float
    top_bias: float
    randomSeed: int
    maxIterations: int
    pruningGen: int
    numberOfEndpoints: int
    newEndPointsPer1000: float
    maxTime: float
    bLeaf: float
    addLeaves: bool
    emitterScale: float
    noModifiers: bool
    subSurface: bool
    showMarkers: bool
    markerScale: float
    timePerformance: bool
    apicalcontrol: float
    apicalcontrolfalloff: float
    apicalcontroltiming: int
    trunk_radius: float
    trunk_radius_scaling: float
    leaf_area_scaling: float
    leaf_area_dbh_scaling: float
    leaf_area_height_scaling: float
    leaf_params: NotRequired[LeafParamsDict]


class MatlabEngineProtocol(Protocol):
    """Protocol for MATLAB engine interface."""
    
    def start_matlab(self) -> "MatlabEngineProtocol":
        """Start a MATLAB engine."""
        ...
    
    def quit(self) -> None:
        """Quit the MATLAB engine."""
        ...
    
    def addpath(self, path: str, nargout: int = 0) -> None:
        """Add a path to the MATLAB search path."""
        ...
    
    def feval(self, func: str, *args: AnyType, nargout: int = 1) -> AnyType:
        """Evaluate a MATLAB function."""
        ...
    
    def eval(self, expr: str, nargout: int = 0) -> AnyType:
        """Evaluate a MATLAB expression."""
        ...
    
    def run_leaf_generation_with_params(self, params: AnyType, nargout: int = 0) -> None:
        """Run leaf generation with parameters."""
        ...
    
    def run_leaf_generation_parallel(
        self, 
        qsm_struct: AnyType, 
        leaf_params_struct: AnyType, 
        nargout: int = 1, 
        background: bool = False
    ) -> Union[str, "MatlabFutureProtocol"]:
        """Run parallel leaf generation."""
        ...


class MatlabFutureProtocol(Protocol):
    """Protocol for MATLAB Future result objects."""
    
    def result(self) -> AnyType:
        """Block until result is ready and return it."""
        ...
    
    def done(self) -> bool:
        """Check if the computation is complete."""
        ...
    
    def cancel(self) -> bool:
        """Attempt to cancel the computation."""
        ...


class MatlabDoubleProtocol(Protocol):
    """Protocol for MATLAB double arrays."""
    
    def tolist(self) -> List[List[float]]:
        """Convert to Python list."""
        ...


class BVHTreeProtocol(Protocol):
    """Protocol for Blender BVHTree."""
    
    @staticmethod
    def FromObject(
        obj: "bpy.types.Object", 
        depsgraph: "bpy.types.Depsgraph",
        deform: bool = True,
        render: bool = False,
        cage: bool = False,
        epsilon: float = 0.0
    ) -> "BVHTreeProtocol":
        """Create BVH tree from Blender object."""
        ...
    
    def ray_cast(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        distance: float = 1.84467e+19
    ) -> Tuple[Optional["mathutils.Vector"], Optional["mathutils.Vector"], Optional[int], Optional[float]]:
        """Cast a ray against the BVH tree."""
        ...


Position3D = Tuple[float, float, float]
Vector3D = Tuple[float, float, float]
ConfigValue = Union[float, int, str, List[float], List[int]]
TreePositionTuple = Tuple[Position3D, SampledConfigDict]
VegetationBBoxDict = TypedDict('VegetationBBoxDict', {
    'min': "mathutils.Vector",
    'max': "mathutils.Vector", 
    'height': float
})
TreeBBoxDict = TypedDict('TreeBBoxDict', {
    'min': "mathutils.Vector",
    'max': "mathutils.Vector"
})


VolumeGenerator = Callable[[], "mathutils.Vector"]
ExcludeFunction = Callable[["mathutils.Vector"], bool]

