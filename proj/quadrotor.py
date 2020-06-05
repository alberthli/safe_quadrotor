import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable


# constants
g = 9.80665 # gravitational acceleration


class Quadrotor:
    """
    Quadrotor object. Conventions are from:

    'Quadrotor control: modeling, nonlinear control design, and simulation'

    Conventions:
    [1] The default inertial frame is fixed and NED (north, east, down).
    [2] Forward direction is x, left is y
    [3] Forward and rear rotors are numbered 2 and 4 respectively. Left and
        right rotors are numbered 3 and 1 respectively. The even rotors
        rotate CCW and the odd rotors rotate CW. When aligned with the inertial
        frame, this means x is aligned with N, y is aligned with E, and
        z is aligned with D.

                          2 ^ x
                            |
                            |      y
                     3 <----o----> 1
                            |
                            |
                            v 4

    [4] Let the quadrotor state be in R^12. The origin and axes of the body
        frame should coincide with the barycenter of the quadrotor and the
        principal axes.
        (x, y, z): Cartesian position in inertial frame
        (phi, theta, psi): angular position in inertial frame (RPY respectively)
        (u, v, w): linear velocity in body frame
        (p, q, r): angular velocities in body frame
    [5] Throughout the code, s is used to refer to the full state, i is used to
        refer to control inputs, and d is used to refer to disturbances, where
        the disturbances are specified as forces and torques in the body frame.
    """

    def __init__(
        self,
        m: float,
        I: np.ndarray,
        kf: float,
        km: float,
        l: float,
        coord_sys: str = 'NED'
    ) -> None:
        """
        Quadrotor initialization

        Parameters
        ----------
        m: float
            Mass of the quadrotor.
        I: np.ndarray, shape=(3,)
            Principal moments of inertia.
        kf: float
            Thrust factor. Internal parameter for rotor speed to body force and
            moment conversion.
        km: float
            Drag factor. Internal parameter for rotor speed to body force and
            moment conversion.
        l: float
            Distance from rotors to center of quadrotor.
        coord_sys: str
            Coordinate convention for the quadrotor. Default is NED.
        """

        self._m = m

        assert I.shape == (3,)
        self._I = I

        self._kf = kf
        self._km = km
        self._l = l

        if coord_sys == 'NWU':
            self._coord_sys = 'NWU' # TODO: implement support for NWU
        else:
            self._coord_sys = 'NED'

    @property
    def _U(self) -> np.ndarray:
        """
        Matrix converting squared rotor speeds to virtual forces/moments.

        i = U @ wsq
        """

        kf = self._kf
        km = self._km
        l = self._l

        U = np.array([
            [kf, kf, kf, kf],
            [kf * l, 0. -kf * l, 0.],
            [0., -kf * l, 0., kf * l],
            [-km, km, -km, km]])

        return U

    @property
    def _invU(self) -> np.ndarray:
        """
        Matrix converting virtual forces/moments to squared rotor speeds.

        wsq = invU @ i
        """

        kf = self._kf
        km = self._km
        l = self._l

        invU = np.array([
            [1. / kf, -2. / (kf * l), 0., -1. / km],
            [1. / kf, 0., -2. / (kf * l), -1 / km],
            [1. / kf, 2. / (kf * l), 0., -1. / km],
            [1. / kf, 0., 2. / (kf * l), 0., 1. / km]]) / 4.

        return invU

    @property
    def _A(self) -> np.ndarray:
        """Linearized autonomous dynamics about hover."""

        A = np.zeros((12, 12))

        A[0:3, 6:9] = np.eye(3)
        A[3:6, 9:12] = np.eye(3)
        A[6, 4] = -g
        A[7, 3] = g

        return A

    @property
    def _B(self) -> np.ndarray:
        """Linearized control dynamics about hover."""

        m = self._m
        Ix, Iy, Iz = self._I
        B = np.zeros((12, 4))

        B[8, 0] = 1 / m
        B[9, 1] = 1 / Ix
        B[10, 2] = 1 / Iy
        B[11, 3] = 1 / Iz

        return B

    @property
    def _D(self) -> np.ndarray:
        """Linearized disturbance dynamics about hover."""

        m = self._m
        Ix, Iy, Iz = self._I
        D = np.zeros((12, 6))

        D[6:9, 0:3] = np.eye(3) / m
        D[9, 3] = 1 / Ix
        D[10, 4] = 1 / Iy
        D[11, 5] = 1 / Iz

        return D

    def _Rwb(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """
        Rotation matrix from BODY to WORLD frame.
        
        Parameters
        ----------
        phi: float
            Roll.
        theta: float
            Pitch.
        psi: float
            Yaw.

        Returns
        -------
        R: np.ndarray, shape=(3,3)
            Rotation matrix from BODY to WORLD frame.
        """

        cphi = np.cos(phi)
        cth = np.cos(theta)
        cpsi = np.cos(psi)
        sphi = np.sin(phi)
        sth = np.sin(theta)
        spsi = np.sin(psi)

        R = np.array([
            [cth * cpsi,                          # row 1
                sphi * sth * cpsi - cphi * spsi,
                cphi * sth * cpsi + sphi * spsi],
            [cth * spsi,                          # row 2
                sphi * sth * spsi + cphi * cpsi,
                cphi * sth * spsi - sphi * cpsi,
            [-sth, sphi * cth, cphi * cth]]])     # row 3

        return R

    def _Twb(self, phi: float, theta: float) -> np.ndarray:
        """
        Angular velocity transformation matrix from BODY to WORLD frame.

        Parameters
        ----------
        phi: float
            Roll.
        theta: float
            Pitch.

        Returns
        -------
        T: np.ndarray, shape=(3,3)
            Angular velocity transformation matrix from BODY to WORLD frame.
        """

        cphi = np.cos(phi)
        cth = np.cos(theta)
        sphi = np.sin(phi)
        tth = np.tan(theta)

        T = np.array([
            [1., sphi * tth, cphi * tth],
            [0., cphi, -sphi],
            [0., sphi / cth, cphi / cth]])

        return T

    def _dyn(
        self,
        s: np.ndarray,
        i: np.ndarray,
        d: np.ndarray = np.zeros(6)
    ) -> np.ndarray:
        """
        Quadrotor dynamics function.

        Parameters
        ----------
        s: np.ndarray, shape=(12,)
            State of the quadrotor in order of (o, alpha, dob, dalphab), where
            o is the position and alpha is the vector of angular positions in
            the INERTIAL frame. dob and dalphab are the linear and angular
            velocities in the BODY frame.
        i: np.ndarray, shape=(4,)
            Virtual input in order of (ft, taux, tauy, tauz), where ft is the
            total force and tauxyz are the angular torques. These are all
            with respect to the BODY frame.
        d: np.ndarray, shape=(6,)
            Disturbances in the BODY frame in order of (fdb, taudb), where fdb
            are forces and taudb are torques. Ordered (x,y,z) each.

        Returns
        -------
        ds: np.ndarray, shape=(12,)
            Time derivative of the states.
        """

        assert s.shape == (12,)
        assert i.shape == (4,)
        assert d.shape == (6,)

        # states
        o = s[0:3]         # x, y, z
        alpha = s[3:6]     # phi, theta, psi
        do_b = s[6:9]      # u, v, w
        dalpha_b = s[9:12] # p, q, r

        # inputs and disturbances
        ft, taux, tauy, tauz = i
        fdx, fdy, fdz, taudx, taudy, taudz = d

        # mass and inertias
        m = self._m
        Ix, Iy, Iz = self._I

        # body -> world transformations
        Rwb = self._Rwb(alpha[0], alpha[1], alpha[2])
        Twb = self._Twb(alpha[0], alpha[1])

        # velocities
        do = Rwb @ do_b
        dalpha = Twb @ dalpha_b

        # accelerations
        u, v, w = do_b
        p, q, r = dalpha_b
        phi, th, _ = alpha

        ddo_b = np.array([
            r * v - q * w - g * np.sin(th) + fdx / m,
            p * w - r * u + g * np.sin(phi) * np.cos(th) + fdy / m,
            q * u - p * v + g * np.cos(th) * np.cos(phi) + (fdz - ft) / m])
        ddalpha_b = np.array([
            ((Iy - Iz) * q * r + taux + taudx) / Ix,
            ((Iz - Ix) * p * r + tauy + taudy) / Iy,
            ((Ix - Iy) * p * q + tauz + taudz) / Iz])
        
        ds = np.hstack((do, dalpha, ddo_b, ddalpha_b))
        return ds

    def _dyn_linear_hover(
        self,
        s: np.ndarray,
        i: np.ndarray,
        d: np.ndarray = np.zeros(6)
    ) -> np.ndarray:
        """
        Linearized quadrotor dynamics function. See docstring for _dyn(...).
        Specifically, we linearize about a hover with constant gravity. This
        function is NOT used for simulation, but for any linear controllers.
        """

        assert s.shape == (12,)
        assert i.shape == (4,)
        assert d.shape == (6,)

        A = self._A
        B = self._B
        D = self._D

        ds = A @ s + B @ i + D @ d
        return ds

    def _simulate(
        self,
        s0: np.ndarray,
        tsim: np.ndarray,
        ifunc: Callable[[np.ndarray], np.ndarray],
        dfunc: Callable[[float, np.ndarray], np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulator for the quadrotor.

        Parameters
        ----------
        s0: np.ndarray, shape=(12,)
            Initial state.
        tsim: np.ndarray, shape=(T,)
            Simulation query points.
        ifunc: Callable[np.ndarray, np.ndarray]
            Controller. Function that takes in the state and returns control.
            We assume the controller to be time-invariant, but this could
            easily be changed in the future.
        dfunc: Callable[np.ndarray, np.ndarray]
            Disturbance function. Takes in state and time and returns a
            simulated disturbance.

        Returns
        -------
        s_sol: np.ndarray, shape=(12, T)
            Solution trajectories at the query times.
        """

        assert s0.shape == (12,)
        assert tsim.ndim == 1

        if dfunc is not None:
            dyn = lambda t, s: self._dyn(s, ifunc(s), dfunc(t, s))
        else:
            dyn = lambda t, s: self._dyn(s, ifunc(s))

        sol = solve_ivp(dyn, (tsim[0], tsim[-1]), s0, t_eval=tsim)
        s_sol = sol.y

        return s_sol

