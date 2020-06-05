from abc import ABC, abstractmethod
import numpy as np

from mpccbfs.quadrotor import Quadrotor


class Controller(ABC):
    def __init__(self, n: float, m: float) -> None:
        super(Controller, self).__init__()

        self._n = n
        self._m = m

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

        super(MultirateController, self).__init__(12, 4)

        self._slow_dt = 1. / slow_rate
        self._fast_dt = 1. / fast_rate

        # memory variables for control scheduling
        self._slow_T_mem = None
        self._fast_T_mem = None
        self._iv = None     # ZOH slow control
        self._iu = None     # ZOH fast control
        self._s_bar = None      # planned state

    def _slow_ctrl(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Slow control law. Updates _s_bar internally.

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
        raise NotImplementedError

    def _fast_ctrl(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Fast control law. Outputs deviation from _iv.

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
        raise NotImplementedError

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

            return self._iv + self._iu

        # slow control update
        if (t - self._slow_T_mem) > self._slow_dt:
            self._slow_T_mem = self._slow_T_mem + self._slow_dt
            self._iv = self._slow_ctrl(t, s)

        # fast control update
        if (t - self._fast_T_mem) > self._fast_dt:
            self._fast_T_mem = self._fast_T_mem + self._fast_dt
            self._iu = self._fast_ctrl(t, s)

        return self._iv + self._iu
        