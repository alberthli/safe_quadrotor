import numpy as np

import fannypack

from mpccbfs.quadrotor import Quadrotor
from mpccbfs.simulator import SimulationEnvironment
from mpccbfs.controllers import MultirateQuadController, PDQuadController
from mpccbfs.obstacles import Obstacle, SphereObstacle

"""
CURRENT USAGE
-------------
[1] To see obstacle CBF, just run with an obstacle list
[2] To see speed constraint, just make obstacle list None

The only reason [2] works is because right now the slow controller is hard-coded
to just always input a vertical thrust no matter what. There's no additional
complex controller being composed with the low-level safe controller.


TODO
----
- Finish designing the slow update in the multirate controller
- Test disturbances
- Make a safe PD controller as a comparative baseline for the safe MPC

- Maybe: look for niche 3d aspect ratio workaround so spheres look proper
"""

fannypack.utils.pdb_safety_net()

# QUADROTOR #
m = 1.                     # mass
I = np.array([1., 1., 1.]) # principal moments of inertia
kf = 1.                    # thrust factor
km = 1.                    # drag factor
l = 0.1                    # rotor arm length
Jtp = 0.1                  # Optional: total rot moment about prop axes (gyro)

quad = Quadrotor(m, I, kf, km, l, Jtp)


# CONTROLLERS #

# multirate controller - TODO: write the slow update
slow_rate = 10.       # slow controller rate
fast_rate = 100.      # fast controller rate
lv_func = lambda x: x # class-K function for relative degree 1 constraints
c1 = 1.               # limits for ECBF pole placement
c2 = 2.
safe_dist = 0.05      # safe distance from obstacles
safe_rot = 0.025      # safe rotation angle (rad)
safe_vel = 0.1        # safe linear velocity

mrc = MultirateQuadController(
    quad,
    slow_rate,
    fast_rate,
    lv_func,
    c1,
    c2,
    safe_dist,  
    safe_rot, 
    safe_vel    
)

# pd controller
sim_dt = 0.01 # dt for simulation
kp_xyz = 0.01 # gains for Cartesian position control
kd_xyz = 0.04
kp_a = 10     # gains for attitude control
kd_a = 5
ref = lambda t: np.array([0.3 * np.cos(t), 0.3 * np.sin(t), 0, 0]) # reference

pdc = PDQuadController(
    quad,
    sim_dt,
    kp_xyz,
    kd_xyz,
    kp_a,
    kd_a,
    ref
)


# OBSTACLES #
obs_list = []
obs1 = SphereObstacle(
    np.array([0.2, 0.0, 0.3]), # position
    0.1                      # radius
)
obs_list = [obs1]


# SIMULATOR #
simulator = SimulationEnvironment(
    quad,     # quadcopter
    mrc,      # controller
    obs_list, # obstacle list
    (-1, 1),  # xyz limits
    (-1, 1),
    (-1, 1)
)


if __name__ == "__main__":
    s0 = np.zeros(12) # initial state
    s0[5] = 1 # initial yaw
    tsim = np.linspace(0, 5, 101) # query times
    sim_data = simulator.simulate(
        s0,
        tsim,
        dfunc=None,    # disturbance function
        animate=True,
        anim_name="diverge" # 'NAME.mp4' to save the run
    )
