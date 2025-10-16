from collections import OrderedDict
from time import time
from typing import List

class Timer:
	"""An ordered collection of timestamps."""

	def __init__(self) -> None:
		self.od: OrderedDict[str, float] = OrderedDict()
		self.od['Start'] = time()

	def add(self, label: str) -> None:
		"""Add a new labeled timestamp."""
		self.od[label] = time()

	def __str__(self) -> str:
		"""Print a list of timings in order of addition. Second column is time since start."""
		keys: List[str] = list(self.od.keys())
		if len(keys) < 2:
			return ""
		return "\n".join("%-20s: %6.1fs %6.1f" % (keys[i],
			self.od[keys[i]] - self.od[keys[i-1]],
			self.od[keys[i]] - self.od[keys[0]]) for i in range(1, len(keys)))
