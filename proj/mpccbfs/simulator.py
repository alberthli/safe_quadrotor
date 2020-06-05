import numpy as np
from typing import Callable, Tuple
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from mpccbfs.quadrotor import Quadrotor


class SimulationEnvironment:
    """
    Simulation environment for the quadrotor. Though the dynamics are computed
    with the inertial frame in NED coordinates, plotting will be done in NWU
    coordinates, since this is more natural. 

    The green rotor denotes the front of the quadrotor.
    """

    def __init__(
        self,
        quad: Quadrotor,
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
        xlim/ylim/zlim: Tuple[float, float]
            The x, y, and z limits for plotting environment
        """

        self._quad = quad

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

    def _clear_frame(self) -> None:
        """
        Clears the environment frame.
        """

        for artist in self._ax.lines + self._ax.collections:
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
        Rnwu = np.array([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.]]) # rotates 180 degrees about north (x)

        # rotor locations (NED -> NWU)
        r1 = o + Rwb @ np.array([0., l, 0.])
        r2 = o + Rwb @ np.array([l, 0., 0.])
        r3 = o + Rwb @ np.array([0., -l, 0.])
        r4 = o + Rwb @ np.array([-l, 0., 0.])

        r1 = Rnwu @ r1
        r2 = Rnwu @ r2
        r3 = Rnwu @ r3
        r4 = Rnwu @ r4

        ro = Rnwu @ Rwb @ np.array([0., 0., -l / 10.]) # rotor offset
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
        self._draw_circle(r1o, l / 2., ro)
        self._draw_circle(r2o, l / 2., ro, color='green')
        self._draw_circle(r3o, l / 2., ro)
        self._draw_circle(r4o, l / 2., ro)

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
        ifunc: Callable[[float, np.ndarray], np.ndarray],
        dfunc: Callable[[float, np.ndarray], np.ndarray] = None,
        animate: bool = False,
        animation_name: str = None
    ) -> np.ndarray:
        """
        Simulates a quadrotor run.

        Parameters
        ----------
        s0: np.ndarray, shape=(12,)
            Initial state.
        tsim: np.ndarray, shape=(T,)
            Simulation query points.
        ifunc: Callable[np.ndarray, np.ndarray]
            Controller. Function that takes in the time and state and returns
            control input.
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

        # simulating dynamics
        quad = self._quad

        if dfunc is not None:
            dyn = lambda t, s: quad._dyn(s, ifunc(t, s), dfunc(t, s))
        else:
            dyn = lambda t, s: quad._dyn(s, ifunc(t, s))

        sol = solve_ivp(dyn, (tsim[0], tsim[-1]), s0, t_eval=tsim)
        s_sol = sol.y

        # animation
        if animate:
            def _anim_quad(i):
                self._clear_frame()
                self._draw_quad(s_sol[:, i])

            anim = animation.FuncAnimation(
                self._fig, _anim_quad, interval=(50 / 3.), frames=len(tsim))
            plt.show()

            if animation_name is not None:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=60, bitrate=1800)
                anim.save('sims/{}.mp4'.format(animation_name), writer=writer)

        return s_sol
