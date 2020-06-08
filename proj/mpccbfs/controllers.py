import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.signal import place_poles
from abc import ABC, abstractmethod
from typing import Callable, List

from mpccbfs.quadrotor import Quadrotor
from mpccbfs.obstacles import Obstacle, SphereObstacle


# constants
g = 9.80665 # gravitational acceleration

class Controller(ABC):
    """
    Abstract class for controllers.
    """

    def __init__(self, n: float, m: float) -> None:
        super(Controller, self).__init__()

        self._n = n
        self._control_dim = m
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

    def _rebalance(self, wsq_cand: np.ndarray) -> np.ndarray:
        """
        Rebalances a vector of candidate squared rotor speeds to ensure that
        they remain non-negative.

        Parameters
        ----------
        wsq_cand: np.ndarray, shape=(4,)
            Candidate squared rotor speeds.

        Returns
        -------
        wsq: np.ndarray, shape=(4,)
            Rebalanced squared rotor speeds.
        """

        assert wsq_cand.shape == (4,)

        if not any(wsq_cand < 0.):
            return wsq_cand

        else:
            # recovering commanded correction values
            D = np.array([         # cors -> wsq
                [0., -1., -1, 1.],
                [-1., 0., 1., 1.],
                [0., 1., -1., 1.],
                [1., 0., 1., 1.]])
            invD = np.array([      # wsq -> cors
                [0., -2., 0., 2.],
                [-2., 0., 2., 0.],
                [-1., 1., -1., 1],
                [1., 1., 1., 1.]]) / 4.
            cors = invD @ wsq_cand # (phi, theta, psi, z)

            z_off = ( # gravity offset
                (self._quad._invU @
                    np.array([self._quad._m * g, 0., 0., 0.]))[0]
            )
            z_cor = cors[0] # z correction
            max_vio = np.max( # maximum non-negative violation occurs from here
                (np.abs(cors[0]) + np.abs(cors[1])),
                (np.abs(cors[0]) + np.abs(cors[2])),
                (np.abs(cors[1]) + np.abs(cors[2]))
            )

            vio_ratio = max_vio / z_cor
            cors /= vio_ratio
            cors[0] = z_cor
            wsq = D @ cors

            assert all(wsq >= 0.)
            return wsq

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

        z_off = ( # gravity offset
            (self._quad._invU @ np.array([self._quad._m * g, 0., 0., 0.]))[0]
        )

        phi_cor = -kp_a * e_phi - kd_a * p
        theta_cor = -kp_a * e_theta - kd_a * q
        psi_cor = -kp_xyz * e_psi - kd_xyz * r # not too aggressive
        z_cor = -kp_xyz * e_z - kd_xyz * w + z_off # gravity offset
        z_cor = np.maximum(z_cor, 0.1) # minimum correction to avoid freefall

        # rotor speed mixing law -> real inputs
        wsq = np.zeros(4)
        wsq[0] = z_cor - theta_cor - psi_cor
        wsq[1] = z_cor - phi_cor + psi_cor
        wsq[2] = z_cor + theta_cor - psi_cor
        wsq[3] = z_cor + phi_cor + psi_cor
        wsq = self._rebalance(wsq)

        # conversion to virtual inputs for simulation
        U = self._quad._U
        i = U @ wsq

        return i

    def reset(self) -> None:
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
        lv_func: Callable[[float], float] = None,
        c1: float = 1.,
        c2: float = 1.,
        safe_dist: float = None,
        safe_rot: float = None,
        safe_vel: float = None
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
        lv_func: Callable[[float], float]
            Linear velocity class-K function for CBF controller.
        c1, c2: float
            Upper limits for ECBF pole placement. Strictly positive.
        safe_dist: float
            Safe distance kept from obstacles.
        safe_rot: float
            Safe amount of rotation in radians.
        safe_vel: float
            Safe linear velocity.
        """

        assert slow_rate > 0.
        assert fast_rate > 0.
        assert safe_dist > 0.
        assert safe_rot > 0.
        assert safe_vel > 0.
        assert c1 > 0.
        assert c2 > 0.

        super(MultirateQuadController, self).__init__(12, 4)

        self._quad = quad

        # time-related variables
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

        # safety-related variables
        self._lv_func = lv_func
        self._c1 = c1
        self._c2 = c2
        self._K_vals = None
        self._safe_dist = quad._l + safe_dist
        self._safe_rot = safe_rot
        self._safe_vel = safe_vel

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
        # print("slow: {}".format(t))
        iv = np.zeros(self._control_dim)
        iv[0] = self._quad._m * g# + 1
        return iv

    def _fast_ctrl(
        self,
        t: float,
        s: np.ndarray,
        obs_list: List[Obstacle]
    ) -> np.ndarray:
        """
        Fast control law. Outputs deviation from _iv using CBFs.

        Parameters
        ----------
        t: float
            Time.
        s: np.ndarray, shape=(n,)
            State.
        obs_list: List[Obstacle]
            List of obstacles.

        Returns
        -------
        iu: np.ndarray, shape=(m,)
            Control input.
        """

        assert s.shape == (self._n,)

        iv = self._iv
        safety_cons = self._get_quad_cons(self._quad, s, obs_list)
        obj = lambda iu: np.linalg.norm(iu - iv) ** 2 # objective
        sol = minimize(obj, np.zeros(4), constraints=safety_cons)

        iu = sol.x - iv
        return iu

    def _get_quad_cons(
        self,
        quad: Quadrotor,
        s: np.ndarray,
        obs_list: List[Obstacle]
    ) -> LinearConstraint:
        """
        Computes the safety constraints for a quadrotor input in the current
        state. Lie derivatives computed using the symbolic helper function. For
        obstacles, we assume the system can omnisciently observe on the
        obstacle parameters. This simplifies the ECBF computations greatly. Also
        has a non-negativity constraints on the rotor speeds.

        Parameters
        ----------
        quad: Quadrotor
            Quadrotor object.
        s: np.ndarray
            Quadrotor state.
        obs_list: List[Obstacle]
            List of obstacles.

        Returns
        -------
        cons: LinearConstraint
            LinearConstraint object for quadratic program.
        """

        assert s.shape == (12,)

        # initializing constraint matrices
        if obs_list is not None:
            num_constr = 7 + len(obs_list)
        else:
            num_constr = 7

        A = np.zeros((num_constr, 4))
        lb = -np.inf * np.ones(num_constr)
        ub = np.zeros(num_constr)

        # unpacking state
        x, y, z = s[0:3]
        phi, theta, psi = s[3:6]
        u, v, w = s[6:9]
        p, q, r = s[9:12]

        cphi = np.cos(phi)
        cth = np.cos(theta)
        cpsi = np.cos(psi)
        sphi = np.sin(phi)
        sth = np.sin(theta)
        spsi = np.sin(psi)
        tth = np.tan(theta)

        # safety params
        v_s = self._safe_vel
        ang_s = self._safe_rot
        d_s = self._safe_dist

        if self._K_vals is None:
            # uninitialized ECBF gains
            self._K_vals = np.NaN * np.zeros((num_constr, 2))

            F = np.array([[0., 1.], [0., 0.]])
            G = np.array([0., 1.]).reshape((2, 1))

        # dynamics
        fdyn = quad._fdyn(s)
        gdyn = quad._gdyn(s)

        # quad params
        m = quad._m
        Ix, Iy, Iz = quad._I


        # linear velocity
        h_v = v_s ** 2. - (u ** 2. + v ** 2. + w ** 2.)
        lfh_v = 2. * g * (w * cphi * cth - u * sth + v * cth * sphi)
        lgh_v = np.array([[-(2. * w) / m, 0., 0., 0.]])

        alpha_v = self._lv_func(h_v)
        A[0, :] = -lgh_v
        ub[0] = lfh_v + alpha_v


        # roll limits
        h_phi = ang_s ** 2. - phi ** 2.
        lfh_phi = -2. * phi * (p + r * cphi * tth + q * sphi * tth)
        lf2h_phi = (
            (2. * p * phi * r * sphi * tth * (Ix - Iz)) / Iy -
            (2. * phi * (r * cphi + q * sphi) *
                (q * cphi - r * sphi)) / cth ** 2. -
            (2. * phi * q * r * (Iy - Iz)) / Ix -
            (2. * p * phi * q * cphi * tth * (Ix - Iy)) / Iz -
            (p + r * cphi * tth + q * sphi * tth) *
            (2. * p + 2. * phi * (q * cphi * tth - r * sphi * tth) +
                2. * r * cphi * tth + 2. * q * sphi * tth)
        )
        lglfh_phi = np.array([
            0.,
            -(2. * phi) / Ix,
            -(2. * phi * sphi * tth) / Iy,
            -(2. * phi * cphi * tth) / Iz])

        if any(np.isnan(self._K_vals[0, :])):
            ddh_nom = lf2h_phi + lglfh_phi @ self._iv
            l1 = np.minimum(lfh_phi / h_phi, -self._c1)
            l2 = np.minimum(
                (ddh_nom + l1 * lfh_phi) / (lfh_phi + l1 * h_phi), -self._c2
            )
            K_phi = place_poles(F, G, np.array([l1, l2])).gain_matrix
            self._K_vals[0, :] = K_phi

        K_phi = self._K_vals[0, :]
        A[1, :] = -lglfh_phi
        ub[1] = K_phi @ np.array([h_phi, lfh_phi]) + lf2h_phi


        # pitch limits
        h_th = ang_s ** 2. - theta ** 2.
        lfh_th = -2. * theta * (q * cphi - r * sphi)
        lf2h_th = (2. * theta * (r * cphi + q * sphi) *
            (p + r * cphi * tth + q * sphi * tth) - (q * cphi - r * sphi) *
            (2. * q * cphi - 2. * r * sphi) + (2. * p * r * theta * cphi *
                (Ix - Iz)) / Iy + (2. * p * q * theta * sphi * (Ix - Iy)) / Iz
        )
        lglfh_th = np.array([
            0.,
            0.,
            -(2. * theta * cphi) / Iy,
            (2. * theta * sphi) / Iz])

        if any(np.isnan(self._K_vals[1, :])):
            ddh_nom = lf2h_th + lglfh_th @ self._iv
            l1 = np.minimum(lfh_th / h_th, -self._c1)
            l2 = np.minimum(
                (ddh_nom + l1 * lfh_th) / (lfh_th + l1 * h_th), -self._c2
            )
            K_th = place_poles(F, G, np.array([l1, l2])).gain_matrix
            self._K_vals[1, :] = K_th

        K_th = self._K_vals[1, :]
        A[2, :] = -lglfh_th
        ub[2] = K_th @ np.array([h_th, lfh_th]) + lf2h_th


        # obstacles
        if obs_list is None:
            obs_list = []

        for k_obs in range(len(obs_list)):
            obs = obs_list[k_obs]

            if obs._otype == 'sphere':
                c_o = obs._c
                x_o, y_o, z_o = c_o # obs center
                r_o = obs._r        # obs radius
                d_so = r_o + d_s    # obs safe distance

                h_o = np.linalg.norm(np.array([x, y, z]) - c_o) ** 2 - d_so ** 2
                lfh_o = (2. * (z - z_o) *
                    (w * cphi * cth - u * sth + v * cth * sphi) +
                    2. * (x - x_o) *
                    (w * (sphi * spsi + cphi * cpsi * sth) -
                        v * (cphi * spsi - cpsi * sphi * sth) +
                        u * cpsi * cth) + 2. * (y - y_o) *
                    (v * (cphi * cpsi + sphi * spsi * sth) -
                        w * (cpsi * sphi - cphi * spsi * sth) + u * cth * spsi)
                )
                lf2h_o = 2. * (u ** 2 + v ** 2 + w ** 2 - g * (z - z_o))
                lglfh_o = np.array([
                    (2. * (x - x_o) * (sphi * spsi + cphi * cpsi * sth) -
                        2. * (y - y_o) * (cpsi * sphi - cphi * spsi * sth) +
                        cphi * cth * 2. * (z - z_o)) / m,
                    0, 0, 0])

                if any(np.isnan(self._K_vals[(k_obs + 2), :])):
                    ddh_nom = lf2h_o + lglfh_o @ self._iv
                    l1 = np.minimum(lfh_o / h_o, -self._c1)
                    l2 = np.minimum(
                        (ddh_nom + l1 * lfh_o) / (lfh_o + l1 * h_o), -self._c2
                    )
                    K_o = place_poles(F, G, np.array([l1, l2])).gain_matrix
                    self._K_vals[(k_obs + 2), :] = K_o

                K_o = self._K_vals[(k_obs + 2), :]
                A[(k_obs + 3), :] = -lglfh_o
                ub[(k_obs + 3)] = K_o @ np.array([h_o, lfh_o]) + lf2h_o

            else:
                raise NotImplementedError

        # non-negative rotor speed constraints
        A[-4:, :] = -self._quad._invU
        ub[-4:] = 0.

        # constraint object
        safety_cons = LinearConstraint(A, lb, ub)
        return safety_cons

    def reset(self) -> None:
        """
        Resets the controller internals.
        """

        self._slow_T_mem = None
        self._fast_T_mem = None
        self._iv = None
        self._iu = None
        self._s_bar = None
        self._K_vals = None

    def ctrl(
        self,
        t: float,
        s: np.ndarray,
        obs_list: List[Obstacle]
    ) -> np.ndarray:
        """
        Multirate control law.

        Parameters
        ----------
        t: float
            Time.
        s: np.ndarray, shape=(n,)
            State.
        obs_list: List[Obstacle]
            List of obstacles.

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
            self._iu = self._fast_ctrl(t, s, obs_list)

            assert self._iv.shape == (self._control_dim,)
            assert self._iu.shape == (self._control_dim,)

            return self._iv + self._iu

        # slow control update
        if (t - self._slow_T_mem) > self._slow_dt:
            self._slow_T_mem = self._slow_T_mem + self._slow_dt
            self._iv = self._slow_ctrl(t, s)
            assert self._iv.shape == (self._control_dim,)

        # fast control update
        if (t - self._fast_T_mem) > self._fast_dt:
            self._fast_T_mem = self._fast_T_mem + self._fast_dt
            self._iu = self._fast_ctrl(t, s, obs_list)
            assert self._iu.shape == (self._control_dim,)

        return self._iv + self._iu
