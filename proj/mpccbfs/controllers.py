from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import numpy as np
import scipy
from mpccbfs.obstacles import Obstacle
from mpccbfs.quadrotor import Quadrotor
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize
from scipy.signal import place_poles
from scipy.sparse import bsr_matrix

# constants
g = 9.80665  # gravitational acceleration


class Controller(ABC):
    """Abstract class for controllers."""

    def __init__(self, n: float, m: float) -> None:
        """Initialize a controller."""
        super(Controller, self).__init__()

        self._n = n
        self._control_dim = m
        self._sim_dt = None

    @abstractmethod
    def ctrl(self, t: float, s: np.ndarray) -> np.ndarray:
        """Control law.

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
        """Resets controller between runs."""


class PDQuadController(Controller):
    """A simple PD controller for position control of a quadrotor.

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
        ref: Callable[[float], np.ndarray],
    ) -> None:
        """Initialize a quadrotor PD controller.

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
        assert kp_xyz >= 0.0
        assert kd_xyz >= 0.0
        assert kp_a >= 0.0
        assert kd_a >= 0.0

        super(PDQuadController, self).__init__(12, 4)

        self._quad = quad
        self._sim_dt = sim_dt
        self._kp_xyz = kp_xyz  # xy pd gains
        self._kd_xyz = kd_xyz  # attitude pd gains
        self._kp_a = kp_a
        self._kd_a = kd_a
        self._ref = ref

    def _rebalance(self, wsq_cand: np.ndarray) -> np.ndarray:
        """Rebalance true quadrotor inputs.

        Ensures candidate squared rotor speeds remain non-negative.

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

        if not any(wsq_cand < 0.0):
            return wsq_cand

        else:
            # recovering commanded correction values
            D = np.array(
                [  # cors -> wsq
                    [0.0, -1.0, -1, 1.0],
                    [-1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, -1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                ]
            )
            invD = (
                np.array(
                    [  # wsq -> cors
                        [0.0, -2.0, 0.0, 2.0],
                        [-2.0, 0.0, 2.0, 0.0],
                        [-1.0, 1.0, -1.0, 1],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
                / 4.0
            )
            cors = invD @ wsq_cand  # (phi, theta, psi, z)

            # z_off = (  # gravity offset
            #     self._quad._invU @ np.array([self._quad._m * g, 0.0, 0.0, 0.0])
            # )[0]
            z_cor = cors[0]  # z correction
            max_vio = np.max(  # maximum non-negative violation occurs from here
                (np.abs(cors[0]) + np.abs(cors[1])),
                (np.abs(cors[0]) + np.abs(cors[2])),
                (np.abs(cors[1]) + np.abs(cors[2])),
            )

            # rebalance
            vio_ratio = max_vio / z_cor
            cors /= vio_ratio
            cors[0] = z_cor
            wsq = D @ cors

            assert all(wsq >= 0.0)
            return wsq

    def ctrl(self, t: float, s: np.ndarray) -> np.ndarray:
        """PD control law.

        This is an inner-outer loop controller, where the inner
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

        z_off = (  # gravity offset
            self._quad._invU @ np.array([self._quad._m * g, 0.0, 0.0, 0.0])
        )[0]

        phi_cor = -kp_a * e_phi - kd_a * p
        theta_cor = -kp_a * e_theta - kd_a * q
        psi_cor = -kp_xyz * e_psi - kd_xyz * r  # not too aggressive
        z_cor = -kp_xyz * e_z - kd_xyz * w + z_off  # gravity offset
        z_cor = np.maximum(z_cor, 0.1)  # minimum correction to avoid freefall

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
        """Reset controller. Does nothing."""


class MultirateQuadController(Controller):
    """A multirate controller for a quadrotor.

    Design from

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
        lv_func: Callable[[float], float],
        c1: float,
        c2: float,
        safe_dist: float,
        safe_rot: float,
        safe_vel: float,
        mpc_T: int,
        mpc_P: np.ndarray,
        mpc_Q: np.ndarray,
        mpc_R: np.ndarray,
        ref: Callable[[float], np.ndarray],
        mpc_only: Optional[bool] = False,
        mpc_vel: Optional[bool] = False,
        mpc_obs: Optional[bool] = False,
    ) -> None:
        """Initialize a quadrotor multirate controller.

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
        mpc_T: int
            MPC future planning horizon in discrete steps.
        mpc_P: np.ndarray, shape=(n, n)
            Quadratic terminal cost matrix. If none, internally computed as the
            solution to the discrete algebraic Riccati equation.
        mpc_Q: np.ndarray, shape=(n, n)
            Quadratic stage cost matrix.
        mpc_R: np.ndarray, shape=(m, m)
            Quadratic input cost matrix.
        ref: Callable[[float], np.ndarray]
            Reference function. Takes in time, outputs desired full state
            trajectory. Tip: it's easy to just solve for x, y, z using something
            like cubic spline interpolation between waypoints, which also
            gives u, v, w. Then, the angles/angular velocity references can be
            set identically to 0.
        mpc_only: Optional[bool], default=False
            Optional kwarg indicating whether to only use MPC or not.
        mpc_obs: Optional[bool], default=False
            Optional kwarg indicating whether the MPC loop accounts for velocity.
        mpc_obs: Optional[bool], default=False
            Optional kwarg indicating whether the MPC loop accounts for obstacles.
        """
        super(MultirateQuadController, self).__init__(12, 4)

        assert slow_rate > 0.0
        assert fast_rate > 0.0
        assert safe_dist > 0.0
        assert safe_rot > 0.0
        assert safe_vel > 0.0
        assert c1 > 0.0
        assert c2 > 0.0
        assert mpc_T >= 1
        assert isinstance(mpc_T, int)
        assert mpc_Q.shape == (12, 12)
        assert mpc_R.shape == (4, 4)
        assert np.array_equal(mpc_Q, mpc_Q.T)
        assert np.array_equal(mpc_R, mpc_R.T)
        assert np.all(np.linalg.eigvals(mpc_Q) >= 0.0)
        np.linalg.cholesky(mpc_R)  # PD check

        if mpc_P is not None:
            assert mpc_P.shape == (12, 12)  # shape
            assert np.array_equal(mpc_P, mpc_P.T)  # symmetry
            assert np.all(np.linalg.eigvals(mpc_P) >= 0.0)  # PSD

        self._quad = quad

        # time-related variables
        self._slow_dt = 1.0 / slow_rate
        self._fast_dt = 1.0 / fast_rate
        self._sim_dt = self._fast_dt / 10.0

        # memory variables for control scheduling
        self._slow_T_mem = None
        self._fast_T_mem = None
        self._iv = None  # ZOH slow control
        self._iu = None  # ZOH fast control
        self._s_bar = None  # planned state

        # control design variables
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

        # MPC variables
        self._mpc_T = mpc_T
        self._mpc_P = mpc_P
        self._mpc_Q = mpc_Q
        self._mpc_R = mpc_R
        self._ref = ref
        self._mpc_only = mpc_only
        self._mpc_vel = mpc_vel
        self._mpc_obs = mpc_obs

    def _slow_ctrl(
        self,
        t: float,
        s: np.ndarray,
        obs_list: List[Obstacle],
    ) -> np.ndarray:
        """Slow control law. Computes the next control action by running MPC.

        Parameters
        ----------
        t: float
            Time.
        s: np.ndarray, shape=(n,)
            State.

        Returns
        -------
        iv: np.ndarray, shape=(m,)
            Control input.
        """
        assert s.shape == (self._n,)

        # dimensions
        n = self._n
        m = self._control_dim

        # local dynamics: ds = A@(s_next-s) + B@(iv-u_bar)
        A = self._quad._A(s)
        B = self._quad._B(s)

        # mpc variables
        T = self._mpc_T
        P = self._mpc_P
        Q = self._mpc_Q
        R = self._mpc_R

        if P is None:
            P = scipy.linalg.solve_discrete_are(A, B, Q, R)

        # optimization
        obj = lambda z: self._get_slow_cost(t, z, P, Q, R)
        safety_cons = self._get_slow_quad_cons(self._quad, s, obs_list, A, B)
        if (self._mpc_obs and obs_list is not None) or self._mpc_vel:
            # computing state reference values over horizon
            r = np.zeros(n * (T + 1) + m * T)
            for i in range(T + 1):
                r[(n * i) : n * (i + 1)] = self._ref(t + i * self._slow_dt)

            # static Hessian
            H = block_diag(
                *[
                    np.kron(np.eye(T), 2 * Q),
                    2 * P,
                    np.kron(np.eye(T), 4 * R)
                    + np.kron(np.diag(np.ones(T - 1), 1), -2 * R)
                    + np.kron(np.diag(np.ones(T - 1), -1), -2 * R),
                ]
            )
            H[-1, -1] /= 2
            H = bsr_matrix(H)

            u_p = np.zeros(m)
            u_p[0] = self._quad._m * g
            u_eq_subtract = np.zeros(n * (T + 1) + m * T)
            u_eq_subtract[(n * (T + 1)) : (n * (T + 1) + m)] = 2 * R @ u_p
            jac = lambda x: H @ (x - r) - u_eq_subtract
            hess = lambda x: H

            sol = minimize(
                obj,
                np.zeros((n * (T + 1) + m * T)),
                jac=jac,
                hess=hess,
                constraints=safety_cons,
                method="trust-constr",
                options={"sparse_jacobian": True},
            )
        else:
            sol = minimize(
                obj,
                np.zeros((n * (T + 1) + m * T)),
                constraints=safety_cons,
                method="SLSQP",
            )
        iv = sol.x[(n * (T + 1)) : (n * (T + 1)) + m]

        return iv

    def _get_slow_cost(
        self,
        t: float,
        z: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> float:
        """Computes cost.

        Parameters
        ----------
        t: float
            Time.
        z: np.ndarray, shape((n * (T + 1) + m * T),)
            Concatenation of x and i over the MPC planning horizon.
        P: np.ndarray, shape=(n, n)
            Quadratic terminal cost matrix. If none, internally computed as the
            solution to the discrete algebraic Riccati equation.
        Q: np.ndarray, shape=(n, n)
            Quadratic stage cost matrix.
        R: np.ndarray, shape=(m, m)
            Quadratic input cost matrix.

        Returns
        -------
        cost: float
            Cost of the trajectory.
        """
        n = self._n
        m = self._control_dim
        T = self._mpc_T

        assert z.shape == ((n * (T + 1) + m * T),)

        # unpacking states
        xs = z[: (n * (T + 1))]  # (x_0, x_1, ..., x_T)
        us = z[(n * (T + 1)) :]  # (u_0, ..., u_{T-1})

        cost = 0

        # state costs - minimize deviation from reference state
        for i in range(T):
            x = xs[(n * i) : (n * (i + 1))]
            r = self._ref(t + i * self._slow_dt)
            cost += (x - r) @ Q @ (x - r)

        xf = xs[(n * T) : (n * (T + 1))]
        rf = self._ref(t + T * self._slow_dt)
        cost += (xf - rf) @ P @ (xf - rf)

        # input costs - minimize deviation from previous control
        u_prev = self._iv
        if u_prev is None:
            u_prev = np.zeros(m)
            u_prev[0] = self._quad._m * g  # equilibrium thrust

        for i in range(T):
            u = us[(m * i) : (m * (i + 1))]
            cost += (u - u_prev) @ R @ (u - u_prev)
            u_prev = u

        return cost

    def _get_slow_quad_cons(
        self,
        quad: Quadrotor,
        s: np.ndarray,
        obs_list: List[Obstacle],
        A: np.ndarray,
        B: np.ndarray,
    ) -> Union[LinearConstraint, NonlinearConstraint]:
        """Compute slow control constraints for MPC.

        Constraints include:
        - rotor speed non-negativity constraints
        - (linearized) dynamics constraints
        - maximum angular tilt constraints

        If the optional flag mpc_vel in the controller is set to
        False, this controller does not include velocity
        constraints. Else, it adds the constraints.

        If the optional flag mpc_obs in the controller is set to
        False, this controller will completely ignore obstacles,
        leaving that for the CBF-based low-level controller. Else,
        it adds the constraints here.

        TODO: add bounds for state and input

        Parameters
        ----------
        quad: Quadrotor
            Quadrotor object.
        s: np.ndarray
            Quadrotor state.
        obs_list: List[Obstacle]
            List of obstacles.
        A: np.ndarray, shape=(n, n)
            Linear continuous passive dynamics.
        B: np.ndarray, shape=(n, m)
            Linear continuous control dynamics.

        Returns
        -------
        cons: Union[LinearConstraint, NonlinearConstraint]
            LinearConstraint object for quadratic program.
        """
        assert s.shape == (self._n,)

        # unpacking variables
        n = self._n
        m = self._control_dim
        T = self._mpc_T
        l = n * (T + 1) + m * T  # length of decision variable
        dt = self._slow_dt
        u_bar = np.array([self._quad._m * g, 0.0, 0.0, 0.0])

        # initializing constraints
        Cineqs = []
        lbineqs = []
        ubineqs = []
        Ceqs = []
        lbeqs = []
        ubeqs = []
        u_off = n * (T + 1)  # input offset for indexing

        # rotor speed non-negativity constraints
        invU = self._quad._invU
        for i in range(T):
            C = np.zeros((m, l))
            C[:, (u_off + m * i) : (u_off + m * (i + 1))] = invU
            lb = 1e-6 * np.ones(m)
            ub = np.inf * np.ones(m)

            Cineqs.append(C)
            lbineqs.append(lb)
            ubineqs.append(ub)

        # dynamics constraints
        for i in range(T + 1):
            C = np.zeros((n, l))

            if i == 0:
                C[:, :n] = np.eye(n)
                lb = s
                ub = s
            else:
                C[:, (n * (i - 1)) : (n * i)] = -(np.eye(n) + dt * A)
                C[:, (n * i) : (n * (i + 1))] = np.eye(n)
                C[:, (u_off + m * (i - 1)) : (u_off + m * i)] = -dt * B
                lb = -dt * (A @ s + B @ u_bar)
                ub = -dt * (A @ s + B @ u_bar)

            Ceqs.append(C)
            lbeqs.append(lb)
            ubeqs.append(ub)

        # angle constraints
        for i in range(T + 1):
            C = np.zeros((2, l))
            C[0, 3 + n * i] = 1
            C[1, 4 + n * i] = 1
            lb = -self._safe_rot * np.ones(2)
            ub = self._safe_rot * np.ones(2)

            Cineqs.append(C)
            lbineqs.append(lb)
            ubineqs.append(ub)

        # [optional] velocity constraints. These constraints are NOT linear.
        nl_cons = []
        if self._mpc_vel:
            sv = self._safe_vel

            for i in range(T + 1):
                f_cons = (
                    lambda x: x[n * i + 6] ** 2 + x[n * i + 7] ** 2 + x[n * i + 8] ** 2
                )

                def J_cons(x):
                    J = np.zeros(l)
                    J[n * i + 6] = 2 * x[n * i + 6]
                    J[n * i + 7] = 2 * x[n * i + 7]
                    J[n * i + 8] = 2 * x[n * i + 8]
                    return J

                def H_cons(x, v):
                    H = np.zeros((l, l))
                    H[n * i + 6, n * i + 6] = 2
                    H[n * i + 7, n * i + 7] = 2
                    H[n * i + 8, n * i + 8] = 2
                    return H

                lb = -sv
                ub = sv

                cons = NonlinearConstraint(f_cons, lb, ub, jac=J_cons, hess=H_cons)
                nl_cons.append(cons)

        # [optional] obstacle constraints. These constraints are NOT linear.
        if self._mpc_obs and obs_list is not None:
            d_s = self._safe_dist

            for k_obs in range(len(obs_list)):
                obs = obs_list[k_obs]

                if obs._otype == "sphere":
                    c_o = obs._c
                    x_o, y_o, z_o = c_o  # obs center
                    r_o = obs._r  # obs radius
                    d_so = r_o + d_s  # obs safe distance

                    for i in range(T + 1):
                        f_cons = (
                            lambda x: (x[n * i] - x_o) ** 2
                            + (x[1 + n * i] - y_o) ** 2
                            + (x[2 + n * i] - z_o) ** 2
                        )

                        def J_cons(x):
                            J = np.zeros(l)
                            J[n * i] = 2 * (x[n * i] - x_o)
                            J[1 + n * i] = 2 * (x[1 + n * i] - y_o)
                            J[2 + n * i] = 2 * (x[2 + n * i] - z_o)
                            return J

                        def H_cons(x, v):
                            H = np.zeros((l, l))
                            H[n * i, n * i] = 2
                            H[1 + n * i, 1 + n * i] = 2
                            H[2 + n * i, 2 + n * i] = 2
                            return H

                        lb = d_so**2
                        ub = np.inf

                        cons = NonlinearConstraint(
                            f_cons, lb, ub, jac=J_cons, hess=H_cons
                        )
                        nl_cons.append(cons)
                else:
                    raise NotImplementedError

        # consolidating constraints
        Cineq = np.vstack(Cineqs)
        lbineq = np.hstack(lbineqs)
        ubineq = np.hstack(ubineqs)

        Ceq = np.vstack(Ceqs)
        lbeq = np.hstack(lbeqs)
        ubeq = np.hstack(ubeqs)

        # constraint object
        lin_cons_eq = LinearConstraint(Ceq, lbeq, ubeq)
        lin_cons_ineq = LinearConstraint(Cineq, lbineq, ubineq)
        cons = [lin_cons_eq, lin_cons_ineq]
        if (self._mpc_obs and obs_list is not None) or self._mpc_vel:
            cons = cons + nl_cons
        return cons

    def _fast_ctrl(
        self,
        t: float,
        s: np.ndarray,
        obs_list: List[Obstacle],
    ) -> np.ndarray:
        """Fast control law. Outputs deviation from _iv using CBFs.

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
        safety_cons = self._get_fast_quad_cons(self._quad, s, obs_list)
        obj = lambda _u: np.linalg.norm(_u - iv) ** 2  # objective
        sol = minimize(
            obj, np.zeros(self._control_dim), constraints=safety_cons, method="SLSQP"
        )
        iu = sol.x - iv
        return iu

    def _get_fast_quad_cons(
        self,
        quad: Quadrotor,
        s: np.ndarray,
        obs_list: List[Obstacle],
    ) -> LinearConstraint:
        """Compute fast control constraints.

        Computes the safety constraints for a quadrotor input in the current
        state. Lie derivatives computed using the symbolic helper function. For
        obstacles, we assume the system can omnisciently observe on the
        obstacle parameters. This simplifies the ECBF computations greatly. Also
        has non-negativity constraints on the rotor speeds.

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
        # NOTE: if more constraints are added, the number "7" needs to be adjusted.
        if obs_list is not None:
            num_constr = 7 + len(obs_list)
        else:
            num_constr = 7

        Cs = []
        lbs = []
        ubs = []

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

            F = np.array([[0.0, 1.0], [0.0, 0.0]])
            G = np.array([0.0, 1.0]).reshape((2, 1))

        # quad params
        m = quad._m
        Ix, Iy, Iz = quad._I

        # linear velocity
        h_v = v_s**2.0 - (u**2.0 + v**2.0 + w**2.0)
        lfh_v = 2.0 * g * (w * cphi * cth - u * sth + v * cth * sphi)
        lgh_v = np.array([[-(2.0 * w) / m, 0.0, 0.0, 0.0]])

        alpha_v = self._lv_func(h_v)

        Cs.append(-lgh_v)
        lbs.append(-np.array([np.inf]))
        ubs.append(np.array([lfh_v + alpha_v]))

        # roll limits
        h_phi = ang_s**2.0 - phi**2.0
        lfh_phi = -2.0 * phi * (p + r * cphi * tth + q * sphi * tth)
        lf2h_phi = (
            (2.0 * p * phi * r * sphi * tth * (Ix - Iz)) / Iy
            - (2.0 * phi * (r * cphi + q * sphi) * (q * cphi - r * sphi)) / cth**2.0
            - (2.0 * phi * q * r * (Iy - Iz)) / Ix
            - (2.0 * p * phi * q * cphi * tth * (Ix - Iy)) / Iz
            - (p + r * cphi * tth + q * sphi * tth)
            * (
                2.0 * p
                + 2.0 * phi * (q * cphi * tth - r * sphi * tth)
                + 2.0 * r * cphi * tth
                + 2.0 * q * sphi * tth
            )
        )
        lglfh_phi = np.array(
            [
                0.0,
                -(2.0 * phi) / Ix,
                -(2.0 * phi * sphi * tth) / Iy,
                -(2.0 * phi * cphi * tth) / Iz,
            ]
        )

        if any(np.isnan(self._K_vals[0, :])):
            ddh_nom = lf2h_phi + lglfh_phi @ self._iv
            l1 = np.minimum(lfh_phi / h_phi, -self._c1)
            l2 = np.minimum(
                (ddh_nom + l1 * lfh_phi) / (lfh_phi + l1 * h_phi), -self._c2
            )
            K_phi = place_poles(F, G, np.array([l1, l2])).gain_matrix
            self._K_vals[0, :] = K_phi

        K_phi = self._K_vals[0, :]

        Cs.append(-lglfh_phi)
        lbs.append(-np.array([np.inf]))
        ubs.append(K_phi @ np.array([h_phi, lfh_phi]) + lf2h_phi)

        # pitch limits
        h_th = ang_s**2.0 - theta**2.0
        lfh_th = -2.0 * theta * (q * cphi - r * sphi)
        lf2h_th = (
            2.0 * theta * (r * cphi + q * sphi) * (p + r * cphi * tth + q * sphi * tth)
            - (q * cphi - r * sphi) * (2.0 * q * cphi - 2.0 * r * sphi)
            + (2.0 * p * r * theta * cphi * (Ix - Iz)) / Iy
            + (2.0 * p * q * theta * sphi * (Ix - Iy)) / Iz
        )
        lglfh_th = np.array(
            [0.0, 0.0, -(2.0 * theta * cphi) / Iy, (2.0 * theta * sphi) / Iz]
        )

        if any(np.isnan(self._K_vals[1, :])):
            ddh_nom = lf2h_th + lglfh_th @ self._iv
            l1 = np.minimum(lfh_th / h_th, -self._c1)
            l2 = np.minimum((ddh_nom + l1 * lfh_th) / (lfh_th + l1 * h_th), -self._c2)
            K_th = place_poles(F, G, np.array([l1, l2])).gain_matrix
            self._K_vals[1, :] = K_th

        K_th = self._K_vals[1, :]

        Cs.append(-lglfh_th)
        lbs.append(-np.array([np.inf]))
        ubs.append(K_th @ np.array([h_th, lfh_th]) + lf2h_th)

        # obstacles
        if obs_list is None:
            obs_list = []

        for k_obs in range(len(obs_list)):
            obs = obs_list[k_obs]

            if obs._otype == "sphere":
                c_o = obs._c
                x_o, y_o, z_o = c_o  # obs center
                r_o = obs._r  # obs radius
                d_so = r_o + d_s  # obs safe distance

                h_o = np.linalg.norm(np.array([x, y, z]) - c_o) ** 2 - d_so**2
                lfh_o = (
                    2.0 * (z - z_o) * (w * cphi * cth - u * sth + v * cth * sphi)
                    + 2.0
                    * (x - x_o)
                    * (
                        w * (sphi * spsi + cphi * cpsi * sth)
                        - v * (cphi * spsi - cpsi * sphi * sth)
                        + u * cpsi * cth
                    )
                    + 2.0
                    * (y - y_o)
                    * (
                        v * (cphi * cpsi + sphi * spsi * sth)
                        - w * (cpsi * sphi - cphi * spsi * sth)
                        + u * cth * spsi
                    )
                )
                lf2h_o = 2.0 * (u**2 + v**2 + w**2 - g * (z - z_o))
                lglfh_o = np.array(
                    [
                        (
                            2.0 * (x - x_o) * (sphi * spsi + cphi * cpsi * sth)
                            - 2.0 * (y - y_o) * (cpsi * sphi - cphi * spsi * sth)
                            + cphi * cth * 2.0 * (z - z_o)
                        )
                        / m,
                        0,
                        0,
                        0,
                    ]
                )

                if any(np.isnan(self._K_vals[(k_obs + 2), :])):
                    ddh_nom = lf2h_o + lglfh_o @ self._iv
                    l1 = np.minimum(lfh_o / h_o, -self._c1)
                    l2 = np.minimum(
                        (ddh_nom + l1 * lfh_o) / (lfh_o + l1 * h_o), -self._c2
                    )
                    K_o = place_poles(F, G, np.array([l1, l2])).gain_matrix
                    self._K_vals[(k_obs + 2), :] = K_o

                K_o = self._K_vals[(k_obs + 2), :]

                Cs.append(-lglfh_o)
                lbs.append(-np.array([np.inf]))
                ubs.append(K_o @ np.array([h_o, lfh_o]) + lf2h_o)

            else:
                raise NotImplementedError

        # non-negative rotor speed constraints
        Cs.append(self._quad._invU)
        lbs.append(1e-6 * np.ones(4))
        ubs.append(np.inf * np.ones(4))

        # consolidating constraints
        C = np.vstack(Cs)
        lb = np.hstack(lbs)
        ub = np.hstack(ubs)

        # constraint object
        safety_cons = LinearConstraint(C, lb, ub)
        return safety_cons

    def reset(self) -> None:
        """Reset the controller internals."""
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
        obs_list: List[Obstacle],
    ) -> np.ndarray:
        """Multirate control law.

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

            self._iv = self._slow_ctrl(t, s, obs_list)
            if self._mpc_only:
                self._iu = self._fast_ctrl(t, s, obs_list)
            else:
                self._iu = np.zeros(self._control_dim)

            assert self._iv.shape == (self._control_dim,)
            assert self._iu.shape == (self._control_dim,)

            return self._iv + self._iu

        # slow control update
        if (t - self._slow_T_mem) > self._slow_dt:
            print(t)  # [DEBUG]
            self._slow_T_mem = self._slow_T_mem + self._slow_dt
            self._iv = self._slow_ctrl(t, s, obs_list)
            assert self._iv.shape == (self._control_dim,)

        # fast control update
        if (t - self._fast_T_mem) > self._fast_dt:
            self._fast_T_mem = self._fast_T_mem + self._fast_dt
            if self._mpc_only:
                self._iu = self._fast_ctrl(t, s, obs_list)
            else:
                self._iu = np.zeros(self._control_dim)
            assert self._iu.shape == (self._control_dim,)

        # non-negativity check - for some reason the optimizer sometimes
        # doesn't respect the constraints
        i = self._iv + self._iu
        wsq = self._quad._invU @ i
        if not all(wsq >= 0.0):
            wsq[wsq < 0.0] = 1e-6
            i = self._quad._U @ wsq

        return i
