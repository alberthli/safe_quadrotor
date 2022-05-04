import matplotlib.pyplot as plt
import numpy as np
from mpccbfs.controllers import MultirateQuadController
from mpccbfs.obstacles import SphereObstacle
from mpccbfs.quadrotor import Quadrotor
from mpccbfs.simulator import SimulationEnvironment

"""
Example of when the multi-rate controller works but MPC alone fails:
MPC running at 8Hz, CBF running at 100Hz, following circle trajectory.
"""

# QUADROTOR #
m = 1.0  # mass
I = np.array([1.0, 1.0, 1.0])  # principal moments of inertia
kf = 1.0  # thrust factor
km = 1.0  # drag factor
l = 0.1  # rotor arm length
Jtp = 0.1  # Optional: total rot moment about prop axes (gyro)

quad = Quadrotor(m, I, kf, km, l, Jtp)


# CONTROLLERS #

# multirate controller
slow_rate = 8.0  # slow controller rate
fast_rate = 100.0  # fast controller rate
lv_func = lambda x: x  # class-K function for relative degree 1 constraints
c1 = 5.0  # limits for ECBF pole placement
c2 = 10.0
safe_dist = 0.05  # safe distance from obstacles
safe_rot = 0.02  # safe rotation angle (rad)
safe_vel = 5.0  # safe linear velocity
mpc_T = 4  # MPC planning steps
mpc_P = np.eye(12)  # terminal cost - None means DARE solution
mpc_P[0:6, 0:6] *= 15
mpc_Q = np.eye(12)  # state cost
mpc_Q[0:6, 0:6] *= 15
mpc_R = 0.01 * np.eye(4)  # control cost


def ref_func(t, quad):  # reference function
    _ref = np.zeros(12)
    _ref[0:3] = np.array([np.cos(0.2 * t), np.sin(0.2 * t), 0.0])
    _ref[6:9] = quad._Rwb(np.zeros(3)).T @ np.array(
        [-0.2 * np.sin(0.2 * t), 0.2 * np.cos(0.2 * t), 0.0]
    )
    return _ref


ref = lambda t: ref_func(t, quad)

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
)

# PD controller
# sim_dt = 0.01 # dt for simulation
# kp_xyz = 0.01 # gains for Cartesian position control
# kd_xyz = 0.04
# kp_a = 10     # gains for attitude control
# kd_a = 5
# ref = lambda t: np.array([0.3 * np.cos(t), 0.3 * np.sin(t), 0, 0]) # reference

# pdc = PDQuadController(
#     quad,
#     sim_dt,
#     kp_xyz,
#     kd_xyz,
#     kp_a,
#     kd_a,
#     ref
# )


# OBSTACLES #
obs1 = SphereObstacle(np.array([0.0, 1.0, 0.0]), 0.2)  # position  # radius
obs2 = SphereObstacle(np.array([-1.0, 0.0, 0.0]), 0.2)  # position  # radius
obs_list = [obs1, obs2]


# SIMULATOR #
simulator = SimulationEnvironment(
    quad,  # quadcopter
    mrc,  # controller
    obs_list,  # obstacle list
    (-2, 2),  # xyz limits
    (-2, 2),
    (-2, 2),
)


if __name__ == "__main__":
    # running simulation
    s0 = np.zeros(12)  # initial state
    sim_length = 30.0  # simulation time
    frames = int(10 * sim_length + 1)  # number of frames
    fps = 20.0  # animation fps
    tsim = np.linspace(0, sim_length, frames)  # query times
    sim_data = simulator.simulate(
        s0,
        tsim,
        dfunc=None,  # disturbance function
        animate=True,  # flag for animation
        anim_name=None,  # 'NAME' to save the run as 'NAME.mp4'
        fps=fps,  # frames per second
    )

    # getting reference data
    ref_traj = np.zeros((12, len(tsim)))
    for i in range(len(tsim)):
        ref_traj[:, i] = ref(tsim[i])

    # plotting positional results
    o = sim_data[0:3, :]
    alpha = sim_data[3:6, :]

    fig, axs = plt.subplots(2, 1)
    fig.tight_layout(pad=3.0)

    # positions
    axs[0].plot(tsim, o[0, :].T, color="orange")
    axs[0].plot(tsim, o[1, :].T, color="green")
    axs[0].plot(tsim, o[2, :].T, color="blue")
    axs[0].legend(["x", "y", "z"], loc="lower left", ncol=3)
    axs[0].plot(tsim, ref_traj[0, :].T, color="orange", linestyle="dashed")
    axs[0].plot(tsim, ref_traj[1, :].T, color="green", linestyle="dashed")
    axs[0].plot(tsim, ref_traj[2, :].T, color="blue", linestyle="dashed")
    axs[0].set_title("Quadrotor Position and Reference Trajectory")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("position")
    axs[0].set_ylim([-2, 2])
    axs[0].set_xlim([tsim[0], tsim[-1]])

    # angles
    axs[1].plot(tsim, alpha[0, :].T, color="orange")
    axs[1].plot(tsim, alpha[1, :].T, color="green")
    axs[1].legend(["roll", "pitch"], loc="lower left", ncol=2)
    axs[1].plot([tsim[0], tsim[-1]], [safe_rot, safe_rot], "k--")
    axs[1].plot([tsim[0], tsim[-1]], [-safe_rot, -safe_rot], "k--")
    axs[1].set_title("Quadrotor Roll/Pitch and Safety Limits")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("angle")
    axs[1].set_ylim([-2.0 * safe_rot, 2.0 * safe_rot])
    axs[1].set_xlim([tsim[0], tsim[-1]])

    plt.show()  # TODO: fix this also showing the simulator when unwanted
