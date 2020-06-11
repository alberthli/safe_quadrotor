import numpy as np
from typing import Callable, Tuple, List
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import mpl_toolkits
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from mpccbfs.quadrotor import Quadrotor
from mpccbfs.controllers import Controller, MultirateQuadController
from mpccbfs.obstacles import Obstacle, SphereObstacle

from mpccbfs.multiratePD import MultiratePD

class SimulationEnvironment:
    """
    Simulation environment for the quadrotor. Note: for some reason matplotlib
    has not implemented equal aspect ratios for 3d plots for almost a decade.
    This means spheres will be ellipsoids. Ridiculous...

    The green rotor denotes the front of the quadrotor.
    """

    def __init__(
        self,
        quad: Quadrotor,
        ctrler: Controller,
        obs_list: List[Obstacle],
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zlim: Tuple[float, float]
    ) -> None:
        """
        Initializes the simulator.

        Parameters
        ----------
        quad: Quadrotor
            A quadrotor object.
        ctrler: Controller
            A controller object.
        obs_list: List[Obstacle]
            A list of obstacles in the environment.
        xlim/ylim/zlim: Tuple[float, float]
            The x, y, and z limits for plotting environment
        """

        self._quad = quad
        self._ctrler = ctrler
        self._obs_list = obs_list

        self._fig = plt.figure()
        self._ax = p3.Axes3D(self._fig)
        self._ax.set_proj_type('ortho')
        self._ax.grid(False)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_zticks([])
        self._ax.set_xlim3d(xlim)
        self._ax.set_ylim3d(ylim)
        self._ax.set_zlim3d(zlim)

    def _clear_frame(self, clear_obs: bool = False) -> None:
        """
        Clears the environment frame.

        Parameters
        ----------
        clear_obs: bool
            Flag for whether to clear obstacles.
        """

        for artist in (self._ax.lines + self._ax.collections):

            # obstacles are line collections
            if not type(artist) == mpl_toolkits.mplot3d.art3d.Line3DCollection:
                artist.remove()
            else:
                if clear_obs:
                    artist.remove()

    def _draw_quad(self, s) -> None:
        """
        Function for drawing the quadcopter in a given state.

        Parameters
        ----------
        s: np.ndarray, shape=(12,)
        """

        assert s.shape == (12,)

        # quadrotor plotting params
        quad = self._quad
        l = quad._l
        o = s[0:3] # x, y, z
        alpha = s[3:6] # phi, theta, psi
        Rwb = quad._Rwb(alpha)

        # rotor base locations on frame in inertial frame
        r1 = o + Rwb @ np.array([l, 0., 0.])
        r2 = o + Rwb @ np.array([0., -l, 0.])
        r3 = o + Rwb @ np.array([-l, 0., 0.])
        r4 = o + Rwb @ np.array([0., l, 0.])

        # rotor vertical offsets
        ro = Rwb @ np.array([0., 0., l / 10.])
        r1o = r1 + ro
        r2o = r2 + ro
        r3o = r3 + ro
        r4o = r4 + ro

        # drawing quadrotor body and rotors
        ax = self._ax
        ax.plot([r2[0], r4[0]], [r2[1], r4[1]], [r2[2], r4[2]], 'k-')
        ax.plot([r1[0], r3[0]], [r1[1], r3[1]], [r1[2], r3[2]], 'k-')
        ax.plot([r1[0], r1o[0]], [r1[1], r1o[1]], [r1[2], r1o[2]], 'k-')
        ax.plot([r2[0], r2o[0]], [r2[1], r2o[1]], [r2[2], r2o[2]], 'k-')
        ax.plot([r3[0], r3o[0]], [r3[1], r3o[1]], [r3[2], r3o[2]], 'k-')
        ax.plot([r4[0], r4o[0]], [r4[1], r4o[1]], [r4[2], r4o[2]], 'k-')
        self._draw_circle(r1o, l / 2., ro, color='green')
        self._draw_circle(r2o, l / 2., ro)
        self._draw_circle(r3o, l / 2., ro)
        self._draw_circle(r4o, l / 2., ro)

    def _draw_traj(self, s_sol, ref_traj, i) -> None:
        """
        Function for drawing the reference traj and the quadcopter traj. 

        Parameters
        ----------
        s_sol: np.ndarray, shape=(12, T)

        ref_traj: np.ndarray, shape=(4, T)

        i: int
        """

        assert s_sol[:,0].shape == (12,)

        ax = self._ax

        # Plot quadcopter traj
        ax.plot(s_sol[0, :i+1], s_sol[1, :i+1], s_sol[2, :i+1], 'c--')

        # Plot reference traj
        ax.plot(ref_traj[0, :i+1], ref_traj[1, :i+1], ref_traj[2, :i+1], 'm--')
        ax.plot(ref_traj[0, i:i+1], ref_traj[1, i:i+1], ref_traj[2, i:i+1], '.m')

    def _draw_obs(self) -> None:
        """
        Draws obstacles in the environment.
        """

        ax = self._ax

        if self._obs_list is None:
            return

        for obs in self._obs_list:
            if obs._otype == 'sphere':
                u = np.linspace(0, 2 * np.pi, 181)
                v = np.linspace(0, np.pi, 91)
                x = obs._r * np.outer(np.cos(u), np.sin(v)) + obs._c[0]
                y = obs._r * np.outer(np.sin(u), np.sin(v)) + obs._c[1]
                z = obs._r * np.outer(np.ones(len(u)), np.cos(v)) + obs._c[2]
                ax.plot_wireframe(x, y, z, color="r")

            else:
                raise NotImplementedError

    def _draw_circle(
        self,
        c: np.ndarray,
        r: float,
        n: np.ndarray,
        color: str = 'black'
    ) -> None:
        """
        Draws a circle in the simulation environment.

        Parameters
        ----------
        c: np.ndarray, shape=(3,)
            Center of circle.
        r: float
            Radius of circle.
        n: np.ndarray, shape=(3,)
            Normal vector of plane in which circle lies.
        color: str
            Color of the circle.
        """

        assert c.shape == (3,)
        assert n.shape == (3,)

        # parameterize circle by planar spanning vectors
        a = np.random.rand(3)
        a = np.cross(n, a)
        b = np.cross(n, a)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        # draw the circle
        thetas = np.linspace(0, 2 * np.pi, 361)
        xs = c[0] + r * np.cos(thetas) * a[0] + r * np.sin(thetas) * b[0]
        ys = c[1] + r * np.cos(thetas) * a[1] + r * np.sin(thetas) * b[1]
        zs = c[2] + r * np.cos(thetas) * a[2] + r * np.sin(thetas) * b[2]

        ax = self._ax
        ax.plot(xs, ys, zs, color=color)

    def simulate(
        self,
        s0: np.ndarray,
        tsim: np.ndarray,
        dfunc: Callable[[float, np.ndarray], np.ndarray] = None,
        animate: bool = False,
        anim_name: str = None
    ) -> np.ndarray:
        """
        Simulates a quadrotor run.

        Parameters
        ----------
        s0: np.ndarray, shape=(12,)
            Initial state.
        tsim: np.ndarray, shape=(T,)
            Simulation query points.
        dfunc: Callable[np.ndarray, np.ndarray]
            Disturbance function. Takes in state and time and returns a
            simulated disturbance.
        animate: bool
            Flag for whether an animation of the run should play.

        Returns
        -------
        s_sol: np.ndarray, shape=(12, T)
            Solution trajectories at the query times.
        """

        assert s0.shape == (12,)
        assert tsim.ndim == 1

        quad = self._quad

        # controller
        if type(self._ctrler) == MultirateQuadController:
            ctrl = lambda t, s: self._ctrler.ctrl(t, s, self._obs_list)
        elif type(self._ctrler) == MultiratePD:
            ctrl = lambda t, s: self._ctrler.ctrl(t, s, self._obs_list)
        else:
            ctrl = lambda t, s: self._ctrler.ctrl(t, s)

        # disturbance function
        if dfunc is not None:
            dyn = lambda t, s: quad._dyn(s, ctrl(t, s), dfunc(t, s))
        else:
            dyn = lambda t, s: quad._dyn(s, ctrl(t, s))

        # simulating dynamics
        sol = solve_ivp(
            dyn,
            (tsim[0], tsim[-1]),
            s0,
            t_eval=tsim,
            max_step=self._ctrler._sim_dt) # cap framerate of reality
        s_sol = sol.y

        # Get ref traj for plotting
        if type(self._ctrler) == MultiratePD:
            ctrler = self._ctrler
            ref = ctrler._ref
            ref_traj = np.zeros((4, s_sol.shape[1]))
            for i in range(s_sol.shape[1]):
                ref_traj[:,i] = ref(tsim[i])

        self._ctrler.reset()

        # animation
        if animate:
            self._draw_obs()

            def _anim_quad(i):
                self._clear_frame()
                self._draw_quad(s_sol[:, i])

                if type(self._ctrler) == MultiratePD:
                    self._draw_traj(s_sol, ref_traj, i)

            anim = animation.FuncAnimation(
                self._fig, _anim_quad, interval=(50 / 3.), frames=len(tsim))

            if anim_name is not None:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=60, bitrate=1800)
                anim.save('{}.mp4'.format(anim_name), writer=writer)

            plt.show()

            self._clear_frame(clear_obs=True)

        return s_sol
