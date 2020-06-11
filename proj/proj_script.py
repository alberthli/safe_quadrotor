import numpy as np

from mpccbfs.quadrotor import Quadrotor
from mpccbfs.simulator import SimulationEnvironment
from mpccbfs.controllers import PDQuadController, MultirateQuadController
from mpccbfs.multiratePD import MultiratePD
from mpccbfs.obstacles import Obstacle, SphereObstacle

"""
CURRENT USAGE
-------------

TODO
----
- Finish designing the slow update in the multirate controller
- Test disturbances
- Make a safe PD controller as a comparative baseline for the safe MPC

- Maybe: look for niche 3d aspect ratio workaround so spheres look proper
"""


# QUADROTOR #
m = 1.                     # mass
I = np.array([1., 1., 1.]) # principal moments of inertia
kf = 1.                    # thrust factor
km = 1.                    # drag factor
l = 0.1                    # rotor arm length
Jtp = None #0.1                  # Optional: total rot moment about prop axes (gyro)

quad = Quadrotor(m, I, kf, km, l, Jtp)

###############
# CONTROLLERS #
###############

# PD # 
sim_dt = 0.01 # dt for simulation
kp_xyz = 0.01 # gains for Cartesian position control
kd_xyz = 0.04
kp_a = 10     # gains for attitude control
kd_a = 5
# ref = lambda t: np.array([0.3 * np.cos(0.2*t), 0.3 * np.sin(0.2*t), 0.05*t, 0]) # reference
ref = lambda t: np.array([1, 0, 0, 0]) # reference
# ref = lambda t: np.array([1, 0, 1, 0]) # reference

pdc = PDQuadController(
    quad,
    sim_dt,
    kp_xyz,
    kd_xyz,
    kp_a,
    kd_a,
    ref
)

# Multirate PD #
slow_rate = 10.       # slow controller rate
fast_rate = 100.      # fast controller rate
lv_func = lambda x: x # class-K function for relative degree 1 constraints
c1 = 1.               # limits for ECBF pole placement
c2 = 2.
safe_dist = 0.05      # safe distance from obstacles
safe_rot = 0.025      # safe rotation angle (rad)
safe_vel = 0.1        # safe linear velocity

mrpdc = MultiratePD(
    quad,
    kp_xyz,
    kd_xyz,
    kp_a,
    kd_a,
    ref,
    slow_rate,
    fast_rate,
    lv_func,
    c1,
    c2,
    safe_dist,
    safe_rot,
    safe_vel
)

# Multirate MPC # 
mrmpc = MultirateQuadController(
    quad,
    slow_rate,
    fast_rate,
    ref,
    lv_func,
    c1,
    c2,
    safe_dist,
    safe_rot,
    safe_vel,
)

# OBSTACLES #
obs_list = []
obs1 = SphereObstacle(
    np.array([0.5, 0.0, 0.0]), # position
    0.1                      # radius
)
obs2 = SphereObstacle(
    np.array([0.5, 0.0, 0.1]), # position
    0.1                      # radius
)
obs3 = SphereObstacle(
    np.array([0.7, 0.0, -0.3]), # position
    0.1                      # radius
)
bigObs = SphereObstacle(np.array([0.5, 0., 0.]), 0.2)
# obs_list = [bigObs]
# obs_list = [obs1, obs2, obs3]
obs_list = []


# SIMULATOR #
simulator = SimulationEnvironment(
    quad,     # quadcopter
    mrmpc,    # mcontroller
    obs_list, # obstacle list
    (-1, 1),  # xyz limits
    (-1, 1),
    (-1, 1)
)

if __name__ == "__main__":
    s0 = np.zeros(12) # initial state
    s0[5] = 0 #1 # initial yaw
    tsim = np.linspace(0, 5, 101) # query times
    sim_data = simulator.simulate(
        s0,
        tsim,
        dfunc=None,    # disturbance function
        animate=True,
        anim_name="mpc_testing.mp4" # 'NAME.mp4' to save the run
    )
