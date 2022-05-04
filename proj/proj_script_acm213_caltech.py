import numpy as np
from mpccbfs.controllers import MultirateQuadController
from mpccbfs.obstacles import SphereObstacle
from mpccbfs.quadrotor import Quadrotor
from mpccbfs.simulator import SimulationEnvironment

# QUADROTOR #
m = 1.0  # mass
I = np.array([1.0, 1.0, 1.0])  # principal moments of inertia
kf = 1.0  # thrust factor
km = 1.0  # drag factor
l = 0.1  # rotor arm length
Jtp = 0.1  # [optional] total rot moment about prop axes (gyro)

quad = Quadrotor(m, I, kf, km, l, Jtp)


# CONTROLLERS #

# multirate controller
slow_rate = 10.0  # slow controller rate
fast_rate = 100.0  # [unused] fast controller rate
lv_func = lambda x: x  # [unused] class-K function for relative degree 1 constraints
c1 = 5.0  # [unused] limits for ECBF pole placement
c2 = 10.0
safe_dist = 0.05  # safe distance from obstacles
safe_rot = 0.02  # safe rotation angle (rad)
safe_vel = 5.0  # safe linear velocity
mpc_T = 5  # MPC planning steps
mpc_P = np.eye(12)  # terminal cost - None means DARE solution
mpc_P[0:3, 0:3] *= 12
mpc_P[6:9, 6:9] *= 2
mpc_Q = np.eye(12)  # state cost
mpc_Q[0:3, 0:3] *= 12
mpc_Q[6:9, 6:9] *= 2
mpc_R = 0.01 * np.eye(4)  # control cost


def ref_func(t, quad):  # [world frame] reference function
    _ref = np.zeros(12)
    _ref[0:3] = np.array([np.cos(0.2 * t), np.sin(0.2 * t), 0.0])
    _ref[6:9] = quad._Rwb(np.zeros(3)).T @ np.array(
        [-0.2 * np.sin(0.2 * t), 0.2 * np.cos(0.2 * t), 0.0]
    )
    return _ref


ref = lambda t: ref_func(t, quad)

# toggled flags for MPC only with all constraints
mpc_only = True
mpc_vel = True
mpc_obs = True

mrc = MultirateQuadController(
    quad,
    slow_rate,
    fast_rate,
    lv_func,
    c1,
    c2,
    safe_dist,
    safe_rot,
    safe_vel,
    mpc_T,
    mpc_P,
    mpc_Q,
    mpc_R,
    ref,
    mpc_only=mpc_only,
    mpc_vel=mpc_vel,
    mpc_obs=mpc_obs,
)

# OBSTACLES #
obs1 = SphereObstacle(np.array([0.0, 1.0, 0.0]), 0.2)
obs2 = SphereObstacle(np.array([-1.0, 0.0, 0.0]), 0.2)
obs3 = SphereObstacle(np.array([0.0, -1.0, 0.0]), 0.2)
obs_list = [obs1, obs2, obs3]


# SIMULATORS #
simulator1 = SimulationEnvironment(
    quad,  # quadcopter
    mrc,  # controller
    obs_list,  # obstacle list
    (-2, 2),  # xyz limits
    (-2, 2),
    (-2, 2),
)
# simulator2 = SimulationEnvironment(
#     quad,  # quadcopter
#     mrc,  # controller
#     None,  # obstacle list
#     (-2, 2),  # xyz limits
#     (-2, 2),
#     (-2, 2),
# )


if __name__ == "__main__":
    # simulation 1: no disturbance
    s0 = np.zeros(12)  # initial state
    sim_length = 30.0  # simulation time
    frames = int(10 * sim_length + 1)  # number of frames
    fps = 20.0  # animation fps
    tsim = np.linspace(0, sim_length, frames)  # query times
    name = "mpc_obs_3obs"  # [TOGGLE]
    # name = None
    sim_data = simulator1.simulate(
        s0,
        tsim,
        dfunc=None,  # disturbance function
        animate=True,  # flag for animation
        anim_name=name,  # 'NAME' to save the run as 'NAME.mp4'
        fps=fps,  # frames per second
    )

    # simulation 2: yes disturbance
    # s0 = np.zeros(12)  # initial state
    # sim_length = 30.0  # simulation time
    # frames = int(10 * sim_length + 1)  # number of frames
    # fps = 20.0  # animation fps
    # tsim = np.linspace(0, sim_length, frames)  # query times

    # def dfunc(t, s):  # disturbance function
    #     if t >= 5 and t < 15:
    #         return np.array([0, 0, -3, 0, 0, 0])
    #     elif t >= 15 and t < 25:
    #         return np.array([0, 0, 3, 0, 0, 0])
    #     else:
    #         return np.zeros(6)

    # name = "mpc_obs_dist_big"  # [TOGGLE]
    # # name = None
    # sim_data = simulator2.simulate(
    #     s0,
    #     tsim,
    #     dfunc=dfunc,  # disturbance function
    #     animate=True,  # flag for animation
    #     anim_name=name,  # 'NAME' to save the run as 'NAME.mp4'
    #     fps=fps,  # frames per second
    # )
