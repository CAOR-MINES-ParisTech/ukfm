import numpy as np
from ukfm import SO3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PENDULUM:
    """Pendulum example, where the state lives on the 2-sphere.
    You can have a text description of the spherical pendulum dynamics in
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
    "Wire length :math:`L`."

    e3 = np.array([0, 0, 1])

    H = np.zeros((2, 3))
    H[:2, :2] = np.eye(2)

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
            + \\mathbf{w}^{(0:3)} \\right) dt\\right)  \\\\
            \\mathbf{u}_{n+1}  &= \\mathbf{u}_{n} + \\dot{\\mathbf{u}}  dt,

        where

        .. math::

            \\dot{\\mathbf{u}}  = \\begin{bmatrix} 
            -\\omega_y  \\omega_x\\ \\\\ \\omega_x \\omega_z
            \\\\ 0 \end{bmatrix} +
            \\frac{g}{l}  \\left(\\mathbf{z}^b \\right)^\\wedge
            \\mathbf{C}^T \\mathbf{z}^b + \\mathbf{w}^{(3:6)} 

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
            \\mathbf{H} \mathbf{x}

        where 

        .. math::

            \mathbf{x} = L \\mathbf{C} \mathbf{e}_3

        :var state: state :math:`\\boldsymbol{\\chi}`.
        """
        x = cls.L*state.Rot.dot(cls.e3)
        return cls.H.dot(x)

    @classmethod
    def phi(cls, state, xi):
        """Retraction.

        .. math::
          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) = \\left( \\begin{matrix}
            \\mathbf{C} \\exp\\left(\\boldsymbol{\\xi}^{(0:3)}\\right) \\\\
            \\mathbf{v} + \\boldsymbol{\\xi}^{(3:6)} \\\\
            \\mathbf{p} + \\boldsymbol{\\xi}^{(6:9)}
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3) \\times \\mathbb R^6`.

        Its corresponding inverse operation is :meth:`~ukfm.PENDULUM.phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """

        new_state = cls.STATE(
            Rot=SO3.exp(xi[:3]).dot(state.Rot),
            u=state.u + xi[3:6],
        )
        return new_state

    @classmethod
    def phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::
          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}\\left(\\boldsymbol{\\chi}\\right) = \\left( \\begin{matrix}
            \\log\\left(\\mathbf{C} \\mathbf{\\hat{C}}^T \\right)\\\\
            \\mathbf{v} - \\mathbf{\\hat{v}} \\\\
            \\mathbf{p} - \\mathbf{\\hat{p}} 
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3) \\times \\mathbb R^6`.

        Its corresponding retraction is :meth:`~ukfm.PENDULUM.phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        xi = np.hstack([SO3.log(state.Rot.dot(hat_state.Rot.T)),
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
        Rot0 = SO3.from_rpy(0, 1, 0)
        states = [self.STATE(Rot0, np.array([0, 0, 0.2]))]

        for n in range(1, self.N):
            # true input
            omegas.append(self.INPUT())
            # propagate state
            w[:3] = model_std[0]*np.random.randn(3)
            w[3:] = model_std[1]*np.random.randn(3)
            states.append(self.f(states[n-1], omegas[n-1], w, self.dt))
            
        return states, omegas

    def simu_h(self, states, obs_std):
        # measurement iteration number
        ys = np.zeros((self.N, 2))
        for n in range(self.N):
            ys[n] = self.h(states[n]) + obs_std * np.random.randn(2)
        return ys

    def errors(self, Rots, us, hat_Rots, hat_us):
        errors = np.zeros((self.N, 2))
        for n in range(self.N):
            errors[n, 0] = np.linalg.norm(SO3.log(Rots[n].T.dot(hat_Rots[n])))
        errors[:, 1] = np.linalg.norm(us - hat_us, axis=1)
        return errors

    def plot_results(self, hat_states, hat_P, states):
        Rots, us = self.get_states(states, self.N)
        hat_Rots, hat_us = self.get_states(hat_states, self.N)

        errors = self.errors(Rots, us, hat_Rots, hat_us)

        t = np.linspace(0, self.T, self.N)

        #fig = plt.figure()
        #ax = plt.axes(projection='3d')

        # Reserve memory

        euls = np.zeros((self.N, 3))
        hat_euls = np.zeros_like(euls)

        for n in range(self.N):

            euls[n] = SO3.to_rpy(Rots[n])
            hat_euls[n] = SO3.to_rpy(hat_Rots[n])

            a = self.L*Rots[n].dot(np.array([0, 0, -1]))

            #if n % 10 == 1:
            #    ax.scatter3D(a[0], a[1], a[2], c='blue')

        #plt.show()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(t, euls)
        ax2.plot(t, hat_euls)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(t, euls-hat_euls)
        plt.show()

    @classmethod
    def get_states(cls, states, N):
        Rots = np.zeros((N, 3, 3))
        us = np.zeros((N, 3))
        for n in range(N):
            Rots[n] = states[n].Rot
            us[n] = states[n].u
        return Rots, us

