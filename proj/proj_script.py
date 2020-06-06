import numpy as np

from mpccbfs.quadrotor import Quadrotor
from mpccbfs.simulator import SimulationEnvironment
from mpccbfs.controllers import MultirateQuadController, PDQuadController
from mpccbfs.obstacles import Obstacle, SphereObstacle

"""
TODO
----
[1] Design the MPC/CBF controllers
"""

if __name__ == "__main__":
    quad = Quadrotor(
        1,                      # mass
        np.array([1., 1., 1.]), # principal moments of inertia
        1.,                     # thrust factor
        1.,                     # drag factor
        0.1,                    # rotor arm length
        0.05,                   # safe distance from obstacles
        0.025                   # safe rotation angle (rad)
    )
    # mrc = MultirateQuadController(quad, 10, 100) # INCOMPLETE: multi-rate ctrl
    pdc = PDQuadController(
        quad,
        0.01, # dt for simulation
        0.01, # kp_xyz, position
        0.04, # kd_xyz
        10,   # kp_a, attitude
        5,    # kd_a
        lambda t: np.array([0.3 * np.cos(t), 0.3 * np.sin(t), 0, 0]) # ref
    ) # PD Controller
    obs1 = SphereObstacle(np.array([0.2,0,0]), 0.15)
    simulator = SimulationEnvironment(
        quad,    # quadcopter
        pdc,     # controller
        None,    # obstacle list
        (-1, 1), # xyz limits
        (-1, 1),
        (-1, 1))
    s0 = np.zeros(12) # initial state
    tsim = np.linspace(0, 30, 301) # query times
    _ = simulator.simulate(s0, tsim, animate=True)
