from abc import ABC, abstractmethod
import numpy as np


class Obstacle(ABC):
	"""
	Abstract class for obstacles.
	"""

	def __init__(self, otype: str) -> None:
		"""
		Initializes the obstacle object.

		Parameters
		----------
		otype: str
			The type of obstacle.
		"""

		super(Obstacle, self).__init__()
		self._otype = otype

class SphereObstacle(Obstacle):
	"""
	Spherical obstacles.
	"""

	def __init__(self, c: np.ndarray, r: float) -> None:
		"""
		Initializes the spherical obstacle.

		Parameters
		----------
		c: np.ndarray, shape=(3,)
			Center of obstacle in NWU inertial coordinates.
		r: float
			Radius of sphere.
		"""

		super(SphereObstacle, self).__init__("sphere")

		assert c.shape == (3,)

		self._c = c
		self._r = r
