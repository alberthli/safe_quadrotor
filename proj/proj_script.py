import numpy as np

from mpccbfs.quadrotor import Quadrotor
from mpccbfs.simulator import SimulationEnvironment

"""
TODO
----
[1] Define an obstacle class, add attribute to simulator to store a list of
    obstacles and draw them.
[2] Define controller interface so controller design can begin.
[3] Set up a way to do multi-rate stuff. Add events for fast and slow control
    updates. This will probably be non-trivial.
"""

if __name__ == "__main__":
    # simple example of simulator usage
	quad = Quadrotor(1, np.array([1., 1., 1.]), 1., 1., 0.1)
	simulator = SimulationEnvironment(
        quad,
        (-1, 1),
        (-1, 1),
        (-1, 1))
	ctrl = lambda t, s: np.array(
        [quad._m * 9.81 + 0.1 * np.sin(t), 0, 0, 0.1 * np.sin(t)])
	s0 = np.zeros(12)
	tsim = np.linspace(0, 10, 101)
	simulator.simulate(s0, tsim, ctrl, animate=True, anim_name='test')
