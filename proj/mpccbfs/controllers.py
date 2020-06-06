import numpy as np
from abc import ABC, abstractmethod
from typing import Callable

import mpccbfs.utils
from mpccbfs.quadrotor import Quadrotor


# constants
g = 9.80665 # gravitational acceleration

class Controller(ABC):
    """
    Abstract class for controllers.
    """

    def __init__(self, n: float, m: float) -> None:
        super(Controller, self).__init__()

        self._n = n
        self._m = m
        self._sim_dt = None

    @abstractmethod
    def ctrl(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Control law.

        Parameters
        ----------
        t: float
            Time.
        s: np.ndarray, shape=(n,)
            State.

        Returns
        -------
        i: np.ndarray, shape=(m,)
            Control input.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets controller between runs.
        """

class PDQuadController(Controller):
    """
    A simple PD controller for position control of a quadrotor.

    NOTE: it's pretty hard to tune this controller, very finicky.
    """

    def __init__(
        self,
        quad: Quadrotor,
        sim_dt: float,
        kp_xyz: float,
        kd_xyz: float,
        kp_a: float,
        kd_a: float,
        ref: Callable[[float], np.ndarray]
    ) -> None:
        """
        Initialization for the quadrotor PD controller.

        Parameters
        ----------
        quad: Quadrotor
            Quadrotor object to be controlled.
        sim_dt: float
            Maximum simulation timestep.
        kp: float
            Proportional gain.
        kd: float
            Derivative gain.
        ref: Callable[[float], np.ndarray]
            Reference function. Takes in time, outputs desired (x, y, z, psi).
            Assumes the desired pitch and roll are zero.
        """

        assert kp_xyz >= 0.
        assert kd_xyz >= 0.
        assert kp_a >= 0.
        assert kd_a >= 0.

        super(PDQuadController, self).__init__(12, 4)

        self._quad = quad
        self._sim_dt = sim_dt
        self._kp_xyz = kp_xyz # xy pd gains
        self._kd_xyz = kd_xyz # attitude pd gains
        self._kp_a = kp_a
        self._kd_a = kd_a
        self._ref = ref

    def ctrl(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        PD control law. This is an inner-outer loop controller, where the inner
        loop controls the states (z, phi, theta, psi) with PID and the outer
        loop sends desired values by converting errors in (x, y, z).

        Parameters
        ----------
        t: float
            Time.
        s: np.ndarray, shape=(n,)
            State.

        Returns
        -------
        i: np.ndarray, shape=(m,)
            Control input.
        """

        assert s.shape == (self._n,)

        # gains
        kp_xyz = self._kp_xyz
        kd_xyz = self._kd_xyz
        kp_a = self._kp_a
        kd_a = self._kd_a

        # reference
        ref = self._ref(t)
        x_d, y_d, z_d, psi_d = ref

        # state extraction
        x, y, z = s[0:3]
        phi, theta, psi = s[3:6]
        u, v, w = s[6:9]
        p, q, r = s[9:12]

        # outer loop: position control
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        e_x = x - x_d
        e_y = y - y_d
        e_xb = e_x * cpsi + e_y * spsi
        e_yb = -e_x * spsi + e_y * cpsi
        de_xb = u
        de_yb = v

        phi_d = -(-kp_xyz * e_yb - kd_xyz * de_yb)
        theta_d = -kp_xyz * e_xb - kd_xyz * de_xb

        # inner loop: attitude control
        e_phi = phi - phi_d
        e_theta = theta - theta_d
        e_psi = psi - psi_d
        e_z = z - z_d

        phi_cor = -kp_a * e_phi - kd_a * p
        theta_cor = -kp_a * e_theta - kd_a * q
        psi_cor = -kp_a * e_psi - kd_a * r
        z_cor = -kp_xyz * e_z - kd_xyz * w

        # rotor speed mixing law -> real inputs
        wsq = np.zeros(4)
        wsq[0] = z_cor - theta_cor - psi_cor
        wsq[1] = z_cor - phi_cor + psi_cor
        wsq[2] = z_cor + theta_cor - psi_cor
        wsq[3] = z_cor + phi_cor + psi_cor

        # conversion to virtual inputs for simulation
        U = self._quad._U
        i = U @ wsq
        i[0] += self._quad._m * g

        return i

    def reset(self):
        """
        Reset controller. No functionality for this controller.
        """
        pass

class MultirateQuadController(Controller):
    """
    A multirate controller for a quadrotor. Design from

    'Multi-Rate Control Design Leveraging Control Barrier Functions and
    Model Predictive Control Policies'.

    A slow MPC-based controller runs high-level planning using a linearized
    model while a fast CBF-based controller runs low-level safety enforcement
    on the full nonlinear model.
    """

    def __init__(
        self,
        quad: Quadrotor,
        slow_rate: float,
        fast_rate: float,
    ) -> None:
        """
        Initialization for the quadrotor multirate controller.

        Parameters
        ----------
        quad: Quadrotor
            Quadrotor object to be controlled.
        slow_rate: float
            Rate of operation of the slow controller in Hz.
        fast_rate: float
            Rate of operation of the fast controller in Hz.
        """

        assert slow_rate > 0.
        assert fast_rate > 0.

        super(MultirateQuadController, self).__init__(12, 4)

        self._quad = quad

        self._slow_dt = 1. / slow_rate
        self._fast_dt = 1. / fast_rate
        self._sim_dt = self._fast_dt / 10.

        # memory variables for control scheduling
        self._slow_T_mem = None
        self._fast_T_mem = None
        self._iv = None    # ZOH slow control
        self._iu = None    # ZOH fast control
        self._s_bar = None # planned state

        # control design variables
        self._A = quad._A
        self._B = quad._B
        self._fdyn = lambda s: quad._fdyn(s)
        self._gdyn = lambda s: quad._gdyn(s)

    def _slow_ctrl(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Slow control law. Updates _s_bar internally using MPC. Trajectory
        generation strategy taken from

        'Minimum Snap Trajectory Generation and Control for Quadrotors'.

        Uses differential flatness of the COM position and yaw to yield fast
        trajectory planning for high-level control.

        Parameters
        ----------
        t: float
            Time.
        s: np.ndarray, shape=(n,)
            State.

        Returns
        -------
        i: np.ndarray, shape=(m,)
            Control input.
        """
        
        assert s.shape == (self._n,)

        # TODO: replace this with actual code. strategy: use linear hover
        # dynamics to plan stuff on a high level.
        print("slow: {}".format(t))
        return np.zeros(self._m)

    def _fast_ctrl(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Fast control law. Outputs deviation from _iv using CBFs.

        Parameters
        ----------
        t: float
            Time.
        s: np.ndarray, shape=(n,)
            State.

        Returns
        -------
        i: np.ndarray, shape=(m,)
            Control input.
        """

        assert s.shape == (self._n,)

        v = self._iv

        # TODO: replace this with actual code.
        print("fast: {}".format(t))
        return np.zeros(self._m)

    def reset(self) -> None:
        """
        Resets the controller internals.
        """

        self._slow_T_mem = None
        self._fast_T_mem = None
        self._iv = None
        self._iu = None
        self._s_bar = None

    def ctrl(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Multirate control law.

        Parameters
        ----------
        t: float
            Time.
        s: np.ndarray, shape=(n,)
            State.

        Returns
        -------
        i: np.ndarray, shape=(m,)
            Control input.
        """
        
        assert s.shape == (self._n,)

        # initializing memory
        if self._slow_T_mem is None and self._fast_T_mem is None:
            self._slow_T_mem = t
            self._fast_T_mem = t

            self._iv = self._slow_ctrl(t, s)
            self._iu = self._fast_ctrl(t, s)

            assert self._iv.shape == (self._m,)
            assert self._iu.shape == (self._m,)

            return self._iv + self._iu

        # slow control update
        if (t - self._slow_T_mem) > self._slow_dt:
            self._slow_T_mem = self._slow_T_mem + self._slow_dt
            self._iv = self._slow_ctrl(t, s)
            assert self._iv.shape == (self._m,)

        # fast control update
        if (t - self._fast_T_mem) > self._fast_dt:
            self._fast_T_mem = self._fast_T_mem + self._fast_dt
            self._iu = self._fast_ctrl(t, s)
            assert self._iu.shape == (self._m,)

        return self._iv + self._iu
