import numpy as np
from ukfm import SO3, SE3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PENDULUM:
    """Pendulum example, where the state lives on the 2-sphere.
    See a text description of the spherical pendulum dynamics in
    :cite:`sjobergAn2019`, Section 7, and  :cite:`kotaruVariation2019`.

    :arg T: sequence time (s).
    :arg model_freq: model frequency (Hz).
    """

    g = 9.81
    "gravity constant (m/s^2) :math:`g`."

    m = 1.0
    "mass of payload (kg) :math:`m`."

    b = 0.0
    "damping :math:`b`."

    L = 1.3
    "wire length :math:`L`."

    e3 = -np.array([0, 0, 1])
    "third coordinate vector :math:`\mathbf{e}^b=-[0,0,1]^T`."

    H = np.zeros((2, 3))
    "observability matrix :math:`\mathbf{H}`."
    H[:, 1:3] = np.eye(2)

    class STATE:
        """State of the system.

        It represents the orientation of the wire and its angular velocity.

        .. math::

            \\boldsymbol{\\chi} \in \\mathcal{M} = \\left\\{ \\begin{matrix} 
           \\mathbf{C} \in SO(3),
            \\mathbf{u} \in \\mathbb R^3
           \\end{matrix} \\right\\}

        :ivar Rot: rotation matrix :math:`\mathbf{C}`.
        :ivar u: angular velocity vector :math:`\mathbf{u}`.
        """

        def __init__(self, Rot, u):
            self.Rot = Rot
            self.u = u

    class INPUT:
        """Input of the propagation model.

        The model does not require any input.
        """

        def __init__(self):
            pass

    def __init__(self, T, model_freq):
        # sequence time (s)
        self.T = T
        # model frequency (Hz)
        self.model_freq = model_freq
        # total number of timestamps
        self.N = T * model_freq
        # integration step (s)
        self.dt = 1 / model_freq

    @classmethod
    def f(cls, state, omega, w, dt):
        """ Propagation function.

        .. math::

            \\mathbf{C}_{n+1}  &= \\mathbf{C}_{n} \\exp\\left(\\left(\\mathbf{u}
            + \\mathbf{w}^{(0:3)} \\right) dt\\right),  \\\\
            \\mathbf{u}_{n+1}  &= \\mathbf{u}_{n} + \\dot{\\mathbf{u}}  dt,

        where

        .. math::

            \\dot{\\mathbf{u}}  = \\begin{bmatrix} 
            -\\omega_y  \\omega_x\\ \\\\ \\omega_x \\omega_z
            \\\\ 0 \end{bmatrix} +
            \\frac{g}{l}  \\left(\\mathbf{e}^b \\right)^\\wedge
            \\mathbf{C}^T \\mathbf{e}^b + \\mathbf{w}^{(3:6)} 

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var w: noise :math:`\\mathbf{w}`.
        :var dt: integration step :math:`dt` (s).

        """

        e3_i = state.Rot.T.dot(cls.e3)
        u = state.u
        d_u = np.array([-u[1]*u[2], u[0]*u[2], 0]) + \
            cls.g/cls.L*np.cross(cls.e3, e3_i)

        new_state = cls.STATE(
            Rot=state.Rot.dot(SO3.exp((u+w[:3])*dt)),
            u=state.u + (d_u+w[3:6])*dt
        )
        return new_state

    @classmethod
    def h(cls, state):
        """ Observation function.

        .. math::

            h\\left(\\boldsymbol{\\chi}\\right)  = 
            \\mathbf{H} \mathbf{x},

        where 

        .. math::

            \mathbf{H}&= \\begin{bmatrix} 0 & 1 & 0 \\\\  0
            & 0 & 1 \end{bmatrix} \\\\
            \mathbf{x} &= L \\mathbf{C} \mathbf{e}^b

        with :math:`\mathbf{x}` the position of the pendulum.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        """
        x = cls.L*state.Rot.dot(cls.e3)
        return cls.H.dot(x)

    @classmethod
    def phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) = 
          \\left( \\begin{matrix}
            \\exp\\left(\\boldsymbol{\\xi}^{(0:3)}\\right) \\mathbf{C}  \\\\
            \\mathbf{u} + \\boldsymbol{\\xi}^{(3:6)}
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3) 
        \\times \\mathbb R^3`.

        Its corresponding inverse operation is :meth:`~ukfm.PENDULUM.phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """

        new_state = cls.STATE(
            Rot=state.Rot.dot(SO3.exp(xi[:3])),
            u=state.u + xi[3:6],
        )
        return new_state

    @classmethod
    def phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}\\left(\\boldsymbol{\\chi}
          \\right) = \\left( \\begin{matrix}
            \\log\\left(\\mathbf{\\hat{C}}^T \\mathbf{C}  \\right)\\\\
            \\mathbf{u} - \\mathbf{\\hat{u}}
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3) 
        \\times \\mathbb R^3`.

        Its corresponding retraction is :meth:`~ukfm.PENDULUM.phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        xi = np.hstack([SO3.log(hat_state.Rot.T.dot(state.Rot)),
                        state.u - hat_state.u])
        return xi

    @classmethod
    def state2chi(cls, state):
        chi = np.eye(4)
        chi[:3, :3] = state.Rot
        chi[:3, 3] = state.u
        return chi

    def simu_f(self, model_std):
        # set noise to zero to compute true trajectory
        w = np.zeros(6)

        # init variables at zero and do for loop
        omegas = []
        Rot0 = SO3.from_rpy(57.3/180*np.pi, 40/180*np.pi, 0)
        states = [self.STATE(Rot0, np.array([-10/180*np.pi,
                                             30/180*np.pi, 0]))]

        for n in range(1, self.N):
            # true input
            omegas.append(self.INPUT())
            # propagate state
            w[:3] = model_std[0]*np.random.randn(3)
            w[3:] = model_std[1]*np.random.randn(3)
            states.append(self.f(states[n-1], omegas[n-1], w, self.dt))

        return states, omegas

    def simu_h(self, states, obs_freq, obs_std):
        # vector to know where measurement happen
        one_hot_ys = np.zeros(self.N)
        # imu_freq/obs_freq must be integer
        one_hot_ys[::int(self.model_freq / obs_freq)] = 1
        idxs = np.where(one_hot_ys == 1)[0]  # indexes where measurement happen
        # total number of measurements
        K = idxs.shape[0]

        # measurement iteration number
        ys = np.zeros((K, 2))
        for k in range(K):
            ys[k] = self.h(states[idxs[k]]) + obs_std * np.random.randn(2)
        return ys, one_hot_ys

    def plot_results(self, hat_states, hat_Ps, states):
        Rots, us = self.get_states(states, self.N)
        hat_Rots, hat_us = self.get_states(hat_states, self.N)

        t = np.linspace(0, self.T, self.N)
        ps = np.zeros((self.N, 3))
        ukf3sigma = np.zeros((self.N, 3))
        hat_ps = np.zeros_like(ps)
        A = np.eye(6)
        e3wedge = SO3.wedge(self.L*self.e3)
        for n in range(self.N):
            ps[n] = self.L*Rots[n].dot(self.e3)
            hat_ps[n] = self.L*hat_Rots[n].dot(self.e3)
            A[:3, :3] = hat_Rots[n].dot(e3wedge)
            P = A.dot(hat_Ps[n]).dot(A.T)
            ukf3sigma[n] = np.diag(P[:3, :3])
        errors = np.linalg.norm(ps - hat_ps, axis=1)
        ukf3sigma = 3*np.sqrt(np.sum(ukf3sigma, 1))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='position (m)', title='position (m)')
        plt.plot(t, ps, linewidth=2)
        plt.plot(t, hat_ps)
        ax.legend([r'$x$', r'$y$', r'$z$', r'$x$ UKF', r'$y$ UKF', r'$z$ UKF'])
        ax.set_xlim(0, t[-1])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)',
               title='position in $xs$ plan')
        plt.plot(ps[:, 0], ps[:, 1], linewidth=2, c='black')
        plt.plot(hat_ps[:, 0], hat_ps[:, 1], c='blue')
        ax.legend([r'true position', r'UKF'])
        ax.axis('equal')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$y$ (m)', ylabel='$z$ (m)',
               title='position in $yz$ plan')
        plt.plot(ps[:, 1], ps[:, 2], linewidth=2, c='black')
        plt.plot(hat_ps[:, 1], hat_ps[:, 2], c='blue')
        ax.legend([r'true position', r'UKF'])
        ax.axis('equal')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (m)',
               title=' position error (m)')
        plt.plot(t, errors, c='blue')
        plt.plot(t, ukf3sigma, c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1])
        ax.set_ylim(0, 0.5)

    @classmethod
    def get_states(cls, states, N):
        Rots = np.zeros((N, 3, 3))
        us = np.zeros((N, 3))
        for n in range(N):
            Rots[n] = states[n].Rot
            us[n] = states[n].u
        return Rots, us
