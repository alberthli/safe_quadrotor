import numpy as np


# constants
g = 9.80665 # gravitational acceleration

class Quadrotor:
    """
    Quadrotor object. Conventions are from:

    'Modelling, Identification and Control of a Quadrotor Helicopter'

    Conventions:
    [1] The default inertial frame is fixed and NWU (north, west, up).
    [2] Forward direction is x, left is y
    [3] Forward and rear rotors are numbered 1 and 3 respectively. Left and
        right rotors are numbered 4 and 2 respectively. The even rotors
        rotate CCW and the odd rotors rotate CW. When aligned with the inertial
        frame, this means x is aligned with N, y is aligned with E, and
        z is aligned with U.

                          1 ^ x
                            |
                     y      |      
                     4 <----o----> 2
                            |
                            |
                            v 3

    [4] Let the quadrotor state be in R^12. The origin and axes of the body
        frame should coincide with the barycenter of the quadrotor and the
        principal axes. Angle states follow ZYX Euler angle conventions.
        (x, y, z): Cartesian position in inertial frame
        (phi, theta, psi): angular position in inertial frame (RPY respectively)
        (u, v, w): linear velocity in body frame
        (p, q, r): angular velocities in body frame
    [5] Vector conventions
        s: State of the quadrotor in order of (o, alpha, dob, dalphab), where
           o is the position and alpha is the vector of angular positions in
           the INERTIAL frame. dob and dalphab are the linear and angular
           velocities in the BODY frame.
        i: Virtual input in order of (ft, taux, tauy, tauz), where ft is the
           total thrust in the -z direction and tauxyz are the angular torques.
           These are all with respect to the BODY frame.
        d: Disturbances in the BODY frame in order of (fdb, taudb), where fdb
           are forces and taudb are torques. Ordered (x,y,z) each.
    """

    def __init__(
        self,
        m: float,
        I: np.ndarray,
        kf: float,
        km: float,
        l: float,
        safe_dist: float,
        safe_rot: float
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
        safe_dist: float
            Safe distance kept from obstacles.
        safe_rot: float
            Safe amount of rotation in radians.
        """

        assert I.shape == (3,)
        assert np.all(I > 0.)

        self._m = m
        self._I = I
        self._kf = kf
        self._km = km
        self._l = l
        self._safe_dist = self._l + safe_dist

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
            [0., -kf * l, 0., kf * l],
            [-kf * l, 0., kf * l, 0.],
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
            [1. / kf, 0., -2. / (kf * l), -1. / km],
            [1. / kf, -2. / (kf * l), 0., 1 / km],
            [1. / kf, 0., 2. / (kf * l), -1. / km],
            [1. / kf, 2. / (kf * l), 0., 1. / km]]) / 4.

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

        B[8, 0] = 1. / m
        B[9, 1] = 1. / Ix
        B[10, 2] = 1. / Iy
        B[11, 3] = 1. / Iz

        return B

    @property
    def _D(self) -> np.ndarray:
        """Linearized disturbance dynamics about hover."""

        m = self._m
        Ix, Iy, Iz = self._I
        D = np.zeros((12, 6))

        D[6:9, 0:3] = np.eye(3) / m
        D[9, 3] = 1. / Ix
        D[10, 4] = 1. / Iy
        D[11, 5] = 1. / Iz

        return D

    def _Rwb(self, alpha: np.ndarray) -> np.ndarray:
        """
        Rotation matrix from BODY to WORLD frame.
        
        Parameters
        ----------
        alpha: np.ndarray, shape=(3,)
            Roll, pitch, yaw vector.

        Returns
        -------
        R: np.ndarray, shape=(3,3)
            Rotation matrix from BODY to WORLD frame.
        """

        assert alpha.shape == (3,)

        phi, theta, psi = alpha
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
                cphi * sth * spsi - sphi * cpsi],
            [-sth, sphi * cth, cphi * cth]])      # row 3

        return R

    def _Twb(self, alpha: np.ndarray) -> np.ndarray:
        """
        Angular velocity transformation matrix from BODY to WORLD frame.

        Parameters
        ----------
        alpha: np.ndarray, shape=(3,)
            Roll, pitch, yaw vector.

        Returns
        -------
        T: np.ndarray, shape=(3,3)
            Angular velocity transformation matrix from BODY to WORLD frame.
        """

        assert alpha.shape == (3,)

        phi, theta, _ = alpha
        cphi = np.cos(phi)
        cth = np.cos(theta)
        sphi = np.sin(phi)
        tth = np.tan(theta)

        T = np.array([
            [1., sphi * tth, cphi * tth],
            [0., cphi, -sphi],
            [0., sphi / cth, cphi / cth]])

        return T

    def _fdyn(self, s: np.ndarray) -> np.ndarray:
        """
        Quadrotor autonomous dynamics WITHOUT gyroscopic effects for 
        simplicity. They are included in the source. If gyroscopic effects are
        present, the system is no longer control affine.

        Parameters
        ----------
        s: np.ndarray, shape=(12,)
            State of quadrotor.

        Returns
        -------
        fdyn: np.ndarray, shape=(12,)
            Time derivatives of states from autonomous dynamics.
        """

        assert s.shape == (12,)

        # states
        alpha = s[3:6]     # phi, theta, psi
        do_b = s[6:9]      # u, v, w
        dalpha_b = s[9:12] # p, q, r

        # moments of inertia
        Ix, Iy, Iz = self._I

        # body -> world transformations
        Rwb = self._Rwb(alpha)
        Twb = self._Twb(alpha)

        # velocities
        do = Rwb @ do_b
        dalpha = Twb @ dalpha_b

        # accelerations
        u, v, w = do_b
        p, q, r = dalpha_b
        phi, th, _ = alpha

        ddo_b = np.array([
            r * v - q * w + g * np.sin(th),
            p * w - r * u - g * np.sin(phi) * np.cos(th),
            q * u - p * v - g * np.cos(th) * np.cos(phi)])
        ddalpha_b = np.array([
            ((Iy - Iz) * q * r) / Ix,
            ((Iz - Ix) * p * r) / Iy,
            ((Ix - Iy) * p * q) / Iz])
        
        fdyn = np.hstack((do, dalpha, ddo_b, ddalpha_b))
        return fdyn

    def _gdyn(self, s: np.ndarray) -> np.ndarray:
        """
        Quadrotor control dynamics.

        Parameters
        ----------
        s: np.ndarray, shape=(12,)
            State of quadrotor.

        Returns
        -------
        gdyn: np.ndarray, shape=(12, 4)
            Matrix representing affine control dynamics.
        """

        assert s.shape == (12,)

        # mass and inertias
        m = self._m
        Ix, Iy, Iz = self._I

        # accelerations
        ddo_b = np.zeros((3, 4))
        ddo_b[2, 0] = 1. / m

        ddalpha_b = np.zeros((3, 4))
        ddalpha_b[0, 1] = 1. / Ix
        ddalpha_b[1, 2] = 1. / Iy
        ddalpha_b[2, 3] = 1. / Iz

        gdyn = np.vstack((np.zeros((6, 4)), ddo_b, ddalpha_b))
        return gdyn

    def _wdyn(self, d: np.ndarray) -> np.ndarray:
        """
        Quadrotor disturbance dynamics.

        Parameters
        ----------
        d: np.ndarray, shape=(6,)
            Disturbances to the quadrotor.

        Returns
        -------
        w: np.ndarray, shape=(12,)
            Time derivatives of states from disturbance dynamics
        """

        assert d.shape == (6,)

        # mass and inertias
        m = self._m
        Ix, Iy, Iz = self._I

        # disturbances
        fdx, fdy, fdz, taudx, taudy, taudz = d

        # accelerations
        ddo_b = np.array([fdx / m, fdy / m, fdz / m])
        ddalpha_b = np.array([taudx / Ix, taudy / Iy, taudy / Iz])

        wdyn = np.hstack((np.zeros(6), ddo_b, ddalpha_b))
        return wdyn

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
            State of the quadrotor.
        i: np.ndarray, shape=(4,)
            Virtual input of the quadrotor.
        d: np.ndarray, shape=(6,)
            Disturbances to the quadrotor.

        Returns
        -------
        ds: np.ndarray, shape=(12,)
            Time derivative of the states.
        """

        assert s.shape == (12,)
        assert i.shape == (4,)
        assert d.shape == (6,)

        fdyn = self._fdyn(s)
        gdyn = self._gdyn(s)
        wdyn = self._wdyn(d)

        ds = fdyn + gdyn @ i + wdyn
        return ds
