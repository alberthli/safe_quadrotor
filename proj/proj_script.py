import numpy as np

from mpccbfs.quadrotor import Quadrotor
from mpccbfs.simulator import SimulationEnvironment
from mpccbfs.controllers import MultirateQuadController

"""
TODO
----
[1] Design the MPC/CBF controllers
[2] Write obstacle interface for simulator. maybe use some sort of set-based
    math? not sure what the best way is to do this, especially for something
    like collision detection.
"""

if __name__ == "__main__":
    # simple example of simulator usage. the current controller is NOT complete.
    # it simply outputs the time the controllers update internally to prove
    # that we can do multirate control.
    quad = Quadrotor(1, np.array([1., 1., 1.]), 1., 1., 0.1)
    mrc = MultirateQuadController(quad, 10, 100)
    simulator = SimulationEnvironment(
        quad,
        mrc,
        (-1, 1),
        (-1, 1),
        (-1, 1))
    s0 = np.zeros(12)
    tsim = np.linspace(0, 10, 101)
    simulator.simulate(s0, tsim, animate=True, anim_name='test')
