import numpy as np
from ukfm import SO2, SE2, SEK2
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


class SLAM2D:
    """
    2D SLAM based on robot odometry and unknown landmark position measurements.
    See a description of the model in the two references
    :cite:`huangObservability2010` , :cite:`HuangA2013`.

    :var T: sequence time (s). 

    :var odo_freq: odometry frequency (Hz).
    """

    J = np.array([[0, -1],
                  [1, 0]])

    max_range = 5
    """maximal range of observation (m)."""
    min_range = 1
    """minimal range of observation (m)."""
    N_ldk = 20
    """number of landmarks :math:`L`."""

    class STATE:
        """State of the system.

        It represents the orientation and the position of the robot along with
        yet observed landmarks.

        .. math::

            \\boldsymbol{\\chi} \in \\mathcal{M} = \\left\\{ \\begin{matrix} 
           \\mathbf{C} \in SO(2),
            \\mathbf{p} \in \\mathbb R^2,
            \\mathbf{p}^l_1 \in \\mathbb R^2, 
            \ldots,
            \\mathbf{p}^l_L \in \\mathbb R^2
           \\end{matrix} \\right\\}

        :ivar Rot: rotation matrix :math:`\mathbf{C}`.
        :ivar p: position of the robot :math:`\mathbf{p}`.
        :ivar p_l: position of the landmark :math:`\mathbf{p}^l_1, \ldots,
              \mathbf{p}^l_L`.
        """

        def __init__(self, Rot, p, p_l=np.zeros((2, 0))):
            self.Rot = Rot
            self.p = p
            self.p_l = p_l

    class INPUT:
        """Input of the propagation model.

        The input are the robot velocities that can be obtained from a
        differential wheel system.

        .. math:: 

            \\boldsymbol{\\omega} \in \\mathcal{U} = \\left\\{ \\begin{matrix}
            \\mathbf{v} \in \\mathbb R,
            \\omega \in \\mathbb R 
            \\end{matrix} \\right\\}

        :ivar v: robot forward  velocity :math:`v`.
        :ivar gyro: robot orientation velocity :math:`\\omega`.
        """

        def __init__(self, v, gyro):
            self.v = v
            self.gyro = gyro

    def __init__(self, T, odo_freq):
        # sequence time (s)
        self.T = T
        # odometry frequency (Hz)
        self.odo_freq = odo_freq
        # total number of timestamps
        self.N = T*odo_freq
        # integration step (s)
        self.dt = 1/odo_freq

    @classmethod
    def f(cls, state, omega, w, dt):
        """ Propagation function.

        .. math::

            \\mathbf{C}_{n+1}  &= \\mathbf{C}_{n} \\exp\\left(\\left(\\omega +
            \\mathbf{w}^{(1)} \\right) dt\\right)  \\\\
            \\mathbf{p}_{n+1}  &= \\mathbf{p}_{n} + \\left( \\mathbf{v}_{n} +
            \\mathbf{w}^{(0)} \\right) dt \\\\
            \\mathbf{p}_{1,n+1}^l  &= \\mathbf{p}_{1,n}^l \\\\
            \\vdots \\\\
            \\mathbf{p}_{L,n+1}^l  &= \\mathbf{p}_{L,n}^l

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var w: noise :math:`\\mathbf{w}`.
        :var dt: integration step :math:`dt` (s).
        """
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO2.exp((omega.gyro + w[1])*dt)),
            p=state.p + state.Rot.dot(np.hstack([omega.v + w[0], 0]))*dt,
            p_l=state.p_l
        )
        return new_state

    @classmethod
    def h(cls, state):
        """Observation function for 1 landmark.

        .. math::

            h\\left(\\boldsymbol{\\chi}\\right)  =
            \\mathbf{C}^T \\left( \\mathbf{p} - \\mathbf{p}^l\\right) 

        :var state: state :math:`\\boldsymbol{\\chi}`.
        """
        y = state.Rot.T.dot(state.p_l - state.p)
        return y

    @classmethod
    def z(cls, state, y):
        """Augmentation function. 

        Return a vector of the novel part of the state only.

        .. math::

            z\\left(\\boldsymbol{\\chi}, \mathbf{y}\\right)  =
            \\mathbf{C} \\mathbf{y} + \\mathbf{p}

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var y: measurement :math:`\\mathbf{y}`.
        """
        z = state.Rot.dot(y) + state.p
        return z

    @classmethod
    def aug_z(cls, state, y):
        """Augmentation function. Return the augmented state.

        .. math::

            \\boldsymbol{\\chi} \\leftarrow \\left(\\boldsymbol{\\chi},
            z\\left(\\boldsymbol{\\chi}, \mathbf{y}\\right) \\right) 

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var y: measurement :math:`\\mathbf{y}`.
        """
        new_state = cls.STATE(
            Rot=state.Rot,
            p=state.p,
            p_l=state.Rot.dot(y) + state.p
        )
        return new_state

    @classmethod
    def phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) =
          \\left( \\begin{matrix}
            \\mathbf{C} \\exp\\left(\\boldsymbol{\\xi}^{(0)}\\right) \\\\
            \\mathbf{p} + \\boldsymbol{\\xi}^{(1:3)} \\\\
            \\mathbf{p}_1^l + \\boldsymbol{\\xi}^{(3:5)} \\\\
            \\vdots \\\\
            \\mathbf{p}_L^l + \\boldsymbol{\\xi}^{(3+2L:5+2L)}
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(2)
        \\times \\mathbb R^{2(L+1)}`.

        Its corresponding inverse operation (for robot state only) is
        :meth:`~ukfm.SLAM2D.red_phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        k = int(xi[3:].shape[0] / 2)
        p_ls = state.p_l + np.reshape(xi[3:], (k, 2))
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO2.exp(xi[0])),
            p=state.p + xi[1:3],
            p_l=p_ls
        )
        return new_state

    @classmethod
    def red_phi(cls, state, xi):
        """Retraction (reduced).

        The retraction :meth:`~ukfm.SLAM2D.phi` applied on the robot state only.
        """
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO2.exp(xi[0])),
            p=state.p + xi[1:3],
            p_l=state.p_l
        )
        return new_state

    @classmethod
    def red_phi_inv(cls, state, hat_state):
        """Inverse retraction (reduced).

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}\\left(\\boldsymbol{\\chi}
          \\right) = \\left( \\begin{matrix} \\log\\left(\\mathbf{C}
          \\mathbf{\\hat{C}}^T\\right) \\\\
            \\mathbf{p} - \\mathbf{\\hat{p}} \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(2)
        \\times \\mathbb R^{2(L+1)}`.

        Its corresponding retraction is :meth:`~ukfm.SLAM2D.red_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`. 

        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        xi = np.hstack([SO2.log(hat_state.Rot.dot(state.Rot.T)),
                        hat_state.p - state.p])
        return xi

    @classmethod
    def aug_phi(cls, state, xi):
        """Retraction used for augmenting state.

        The retraction :meth:`~ukfm.SLAM2D.phi` applied on the robot state only.
        """
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO2.exp(xi[0])),
            p=state.p + xi[1:]
        )
        return new_state

    @classmethod
    def aug_phi_inv(cls, state, aug_state):
        """Retraction used for augmenting state.

        The inverse retraction :meth:`~ukfm.SLAM2D.phi` applied on the landmark
        only.
        """
        return aug_state.p_l - state.p_l

    @classmethod
    def up_phi(cls, state, xi):
        """Retraction used for updating state and infering Jacobian.

        The retraction :meth:`~ukfm.SLAM2D.phi` applied on the robot state and
        one landmark only.
        """
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO2.exp(xi[0])),
            p=state.p + xi[1:3],
            p_l=state.p_l + xi[3:5]
        )
        return new_state

    @classmethod
    def left_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) =
               \\left( \\begin{matrix}
               \\mathbf{C} \\mathbf{C}_\\mathbf{T} \\\\
               \\mathbf{p} + \\mathbf{C} \\mathbf{r}_1 \\\\
               \\mathbf{p}_1^l + \\mathbf{C} \\mathbf{r}_2 \\\\
               \\vdots \\\\
               \\mathbf{p}_L^l + \\mathbf{C} \\mathbf{r}_{1+L} \\\\
          \\end{matrix} \\right)

        where

        .. math::

                \\mathbf{T} = \\exp\\left(\\boldsymbol{\\xi}\\right) =
                \\begin{bmatrix}
                \\mathbf{C}_\\mathbf{T} & \\mathbf{r}_1 & \\cdots &
                \\mathbf{r}_{1+L}  \\\\
                    \\mathbf{0}^T & & \\mathbf{I}&
                \\end{bmatrix}

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in
        SE_{1+L}(2)` with left multiplication.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """

        chi = SEK2.exp(xi)
        new_state = cls.STATE(
            Rot=state.Rot.dot(chi[:2, :2]),
            p=state.p + state.Rot.dot(chi[:2, 2]),
            p_l=state.p_l + state.Rot.dot(chi[:2, 3:]).T
        )
        return new_state

    @classmethod
    def left_red_phi(cls, state, xi):
        """Retraction (reduced).

        The retraction :meth:`~ukfm.SLAM2D.left_phi` applied on the robot state
        only.
        """
        return cls.left_phi(state, xi)

    @classmethod
    def left_red_phi_inv(cls, state, hat_state):
        """Inverse retraction (reduced).

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}
          \\left(\\boldsymbol{\\chi}\\right) =
          \\log\\left(\\boldsymbol{\\chi}
          \\boldsymbol{\\hat{\\chi}}^{-1}\\right)

        The robot state is viewed as a element :math:`\\boldsymbol{\chi} \\in
        SE(2)`.

        Its corresponding retraction is :meth:`~ukfm.SLAM2D.left_red_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        chi = cls.state2chi(state)
        hat_chi = cls.state2chi(hat_state)
        xi = SEK2.log(SEK2.inv(chi).dot(hat_chi))
        return xi

    @classmethod
    def left_aug_phi(cls, state, xi):
        """Retraction used for augmenting state.

        The retraction :meth:`~ukfm.SLAM2D.left_phi` applied on the robot state
        only.
        """
        chi = SE2.exp(xi)
        new_state = cls.STATE(
            Rot=state.Rot.dot(chi[:2, :2]),
            p=state.p + state.Rot.dot(chi[:2, 2])
        )
        return new_state

    @classmethod
    def left_aug_phi_inv(cls, state, aug_state):
        """Retraction used for augmenting state.

        The inverse retraction :meth:`~ukfm.SLAM2D.left_phi` applied on the
        landmark only.
        """
        chi = cls.aug_state2chi(state)
        aug_chi = cls.aug_state2chi(aug_state)
        return SE2.log(SE2.inv(chi).dot(aug_chi))[1:3]

    @classmethod
    def left_up_phi(cls, state, xi):
        """Retraction used for updating state and infering Jacobian.

        The retraction :meth:`~ukfm.SLAM2D.left_phi` applied on the robot state
        and one landmark only.
        """
        chi = SEK2.exp(xi)
        new_state = cls.STATE(
            Rot=state.Rot.dot(chi[:2, :2]),
            p=state.p + state.Rot.dot(chi[:2, 2]),
            p_l=state.p_l + np.squeeze(state.Rot.dot(chi[:2, 3:]))
        )
        return new_state

    @classmethod
    def right_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) =
                \\left( \\begin{matrix}
                \\mathbf{C}_\\mathbf{T} \\mathbf{C}  \\\\
                 \\mathbf{C}_\\mathbf{T}\\mathbf{p} +  \\mathbf{r}_1 \\\\
               \\mathbf{C}_\\mathbf{T} \\mathbf{p}_1^l + \\mathbf{r}_2 \\\\
               \\vdots \\\\
               \\mathbf{C}_\\mathbf{T} \\mathbf{p}_L^l + \\mathbf{r}_{1+L} \\\\
          \\end{matrix} \\right)

        where

        .. math::

                \\mathbf{T} = \\exp\\left(\\boldsymbol{\\xi}\\right) =
                \\begin{bmatrix}
                \\mathbf{C}_\\mathbf{T} & \\mathbf{r}_1 & \\cdots &
                \\mathbf{r}_{1+L}  \\\\
                    \\mathbf{0}^T & & \\mathbf{I}&
                \\end{bmatrix}

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in
        SE_{1+L}(2)` with right multiplication.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        chi = SEK2.exp(xi)
        p_l = (chi[:2, 3:] + chi[:2, :2].dot(state.p_l.T)).T

        new_state = cls.STATE(
            Rot=chi[:2, :2].dot(state.Rot),
            p=chi[:2, 2] + chi[:2, :2].dot(state.p),
            p_l=p_l
        )
        return new_state

    @classmethod
    def right_red_phi(cls, state, xi):
        """Retraction (reduced).

        The retraction :meth:`~ukfm.SLAM2D.right_phi`.
        """
        return cls.right_phi(state, xi)

    @classmethod
    def right_red_phi_inv(cls, state, hat_state):
        """Inverse retraction (reduced).

        .. math::

            \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}
            \\left(\\boldsymbol{\\chi}\\right) = 
            \\log\\left(\\boldsymbol{\\hat{\\chi}}^{-1} 
            \\boldsymbol{\\chi}\\right)

        The robot state is viewed as a element :math:`\\boldsymbol{\chi} \\in
        SE_{L+1}(2)`.

        Its corresponding retraction is :meth:`~ukfm.SLAM2D.right_red_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        chi = cls.state2chi(state)
        hat_chi = cls.state2chi(hat_state)
        xi = SEK2.log(hat_chi.dot(SEK2.inv(chi)))
        return xi

    @classmethod
    def right_aug_phi(cls, state, xi):
        """Retraction used for augmenting state.

        The retraction :meth:`~ukfm.SLAM2D.right_phi` applied on the robot state
        only.
        """
        chi = SE2.exp(xi)
        new_state = cls.STATE(
            Rot=chi[:2, :2].dot(state.Rot),
            p=chi[:2, :2].dot(state.p) + chi[:2, 2]
        )
        return new_state

    @classmethod
    def right_aug_phi_inv(cls, state, aug_state):
        """Retraction used for augmenting state.

        The inverse retraction :meth:`~ukfm.SLAM2D.right_phi` applied on the
        landmark only.
        """
        chi = cls.aug_state2chi(state)
        aug_chi = cls.aug_state2chi(aug_state)
        return SE2.log(aug_chi.dot(SE2.inv(chi)))[1:3]

    @classmethod
    def right_up_phi(cls, state, xi):
        """Retraction used for updating state and infering Jacobian.

        The retraction :meth:`~ukfm.SLAM2D.right_phi` applied on the robot state
        and one landmark only.
        """
        chi = SEK2.exp(xi)
        new_state = cls.STATE(
            Rot=chi[:2, :2].dot(state.Rot),
            p=chi[:2, 2] + chi[:2, :2].dot(state.p),
            p_l=np.squeeze(chi[:2, 3]) + np.squeeze(chi[:2, :2].dot(state.p_l))
        )
        return new_state

    @classmethod
    def state2chi(cls, state):
        l = state.p_l.shape[0] + 1
        chi = np.eye(l + 2)
        chi[:2, :2] = state.Rot
        chi[:2, 2] = state.p
        chi[:2, 3:] = state.p_l.T
        return chi

    @classmethod
    def aug_state2chi(cls, state):
        chi = np.eye(3)
        chi[:2, :2] = state.Rot
        chi[:2, 2] = np.squeeze(state.p_l)
        return chi

    @classmethod
    def get_states(cls, states, N):
        Rots = np.zeros((N, 2, 2))
        ps = np.zeros((N, 2))
        for n in range(N):
            Rots[n] = states[n].Rot
            ps[n] = states[n].p
        return Rots, ps

    @classmethod
    def get_cov(cls, list_covs, N):
        covs = np.zeros((N, 3+2*cls.N_ldk, 3+2*cls.N_ldk))
        for n in range(N):
            P = list_covs[n]
            covs[n, :P.shape[0], :P.shape[0]] = P
        return covs

    def errors(self, Rots, hat_Rots, ps, hat_ps):
        errors = np.zeros((self.N, 3))
        for n in range(self.N):
            errors[n, 0] = SO2.log(Rots[n].T.dot(hat_Rots[n]))
        errors[:, 1:] = ps-hat_ps
        return errors

    def plot_traj(self, states, ldks):
        Rots, ps = self.get_states(states, self.N)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)', title="Robot position")
        plt.plot(ps[:, 0], ps[:, 1], linewidth=2, c='black')
        ax.scatter(ldks[:, 0], ldks[:, 1], c='red')
        ax.legend([r'true position',
                   r'landmarks'])
        ax.axis('equal')

    def plot_results(self, hat_states, hat_Ps, states, ldks):
        Rots, ps = self.get_states(states, self.N)
        hat_Rots, hat_ps = self.get_states(hat_states, self.N)

        errors = self.errors(Rots, hat_Rots, ps, hat_ps)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)', title='Robot position')

        plt.plot(ps[:, 0], ps[:, 1], linewidth=2, c='black')
        plt.plot(hat_ps[:, 0], hat_ps[:, 1], c='blue')
        ax.scatter(ldks[:, 0], ldks[:, 1], c='red')
        ax.axis('equal')
        ax.legend([r'true position', 'UKF', r'landmarks'])

        hat_Ps = self.get_cov(hat_Ps, self.N)

        ukf3sigma = 3 * np.sqrt(hat_Ps[:, 0, 0])
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (deg)',
               title='Orientation error (deg)')

        t = np.linspace(0, self.T, self.N)
        plt.plot(t, 180/np.pi*errors[:, 0], c='blue')
        plt.plot(t, 180/np.pi*ukf3sigma, c='blue', linestyle='dashed')
        plt.plot(t, 180/np.pi*(- ukf3sigma), c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1])

        ukf3sigma = 3 * np.sqrt(hat_Ps[:, 1, 1] + hat_Ps[:, 2, 2])
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (m)',
               title='Robot position error (m)')

        plt.plot(t, errors[:, 1], c='blue')
        plt.plot(t, ukf3sigma, c='blue', linestyle='dashed')
        plt.plot(t, -ukf3sigma, c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1])

    def nees(self, err, Ps, Rots, ps, name):

        neess = np.zeros((self.N, 2))
        J = np.eye(3)

        def err2nees(err, P):
            # separate orientation and position
            nees_Rot = err[0]**2 / P[0, 0]
            nees_p = err[1:3].dot(np.linalg.inv(P[1:3, 1:3]).dot(err[1:3]))/2
            return np.array([nees_Rot, nees_p])

        for n in range(10, self.N):
            # covariance need to be turned
            if name == 'STD':
                P = Ps[n][:3, :3]
            elif name == 'LEFT':
                J[1:3, 1:3] = Rots[n]
                P = J.dot(Ps[n][:3, :3]).dot(J.T)
            else:
                J[1:3, 0] = self.J.dot(ps[n])
                P = J.dot(Ps[n][:3, :3]).dot(J.T)
            neess[n] = err2nees(err[n], P)
        return neess

    def nees_print(self, ukf_nees, left_ukf_nees, right_ukf_nees, iekf_nees,
                   ekf_nees):

        t = np.linspace(0,  self.dt * self.N,  self.N)

        def f(x):
            return np.mean(x, axis=0)
        ukf_nees = f(ukf_nees)
        left_ukf_nees = f(left_ukf_nees)
        right_ukf_nees = f(right_ukf_nees)
        iekf_nees = f(iekf_nees)
        ekf_nees = f(ekf_nees)

        # plot orientation nees
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='orientation NEES',
               title='Robot orientation NEES', yscale="log")

        plt.plot(t, ukf_nees[:, 0], c='magenta')
        plt.plot(t, left_ukf_nees[:, 0], c='green')
        plt.plot(t, right_ukf_nees[:, 0], c='cyan')
        plt.plot(t, ekf_nees[:, 0], c='red')
        plt.plot(t, iekf_nees[:, 0], c='blue')
        ax.legend([r'$SO(2) \times \mathbb{R}^{2(1+L)}$ UKF',
                   r'\textbf{$SE_{1+L}(2)$ UKF (left)}',
                   r'\textbf{$SE_{1+L}(2)$ UKF (right)}', r'EKF', 
                   r'IEKF [BB17]'])
        ax.set_xlim(0, t[-1])

        # plot position nees
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='position NEES',
               title='Robot position NEES', yscale="log")

        plt.plot(t, ukf_nees[:, 1], c='magenta')
        plt.plot(t, left_ukf_nees[:, 1], c='green')
        plt.plot(t, right_ukf_nees[:, 1], c='cyan')
        plt.plot(t, ekf_nees[:, 1], c='red')
        plt.plot(t, iekf_nees[:, 1], c='blue')
        ax.legend([r'$SO(2) \times \mathbb{R}^{2(1+L)}$ UKF',
                   r'\textbf{$SE_{1+L}(2)$ UKF (left)}',
                   r'\textbf{$SE_{1+L}(2)$ UKF (right)}', r'EKF', 
                   r'IEKF [BB17]'])
        ax.set_xlim(0, t[-1])

        def g(x):
            return np.mean(x)

        print(' ')
        print(' Normalized Estimation Error Squared (NEES) w.r.t. orientation')
        print("    -SO(2) x R^(2(1+L)) UKF: % .2f " % g(ukf_nees[:, 0]))
        print("    -left SE_{1+L}(2) UKF  : % .2f " % g(left_ukf_nees[:, 0]))
        print("    -right SE_{1+L}(2) UKF : % .2f " % g(right_ukf_nees[:, 0]))
        print("    -EKF                   : % .2f " % g(ekf_nees[:, 0]))
        print("    -IEKF                  : % .2f " % g(iekf_nees[:, 0]))

        print(' ')
        print(' Normalized Estimation Error Squared (NEES) w.r.t. position')
        print("    -SO(2) x R^(2(1+L)) UKF: % .2f " % g(ukf_nees[:, 1]))
        print("    -left SE_{1+L}(2) UKF  : % .2f " % g(left_ukf_nees[:, 1]))
        print("    -right SE_{1+L}(2) UKF : % .2f " % g(right_ukf_nees[:, 1]))
        print("    -EKF                   : % .2f " % g(ekf_nees[:, 1]))
        print("    -IEKF                  : % .2f " % g(iekf_nees[:, 1]))

    def simu_f(self, odo_std, v, gyro):
        # create the map
        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return x, y

        r = v / gyro  # radius
        ldks = np.zeros((self.N_ldk, 2))
        for i in range(self.N_ldk):
            rho = r + self.min_range + 2
            th = 2 * np.pi * i / self.N_ldk
            [x, y] = pol2cart(rho, th)
            # shift y w/ r since robot starts at (0,0)
            ldks[i] = np.array([x, y + r])

        w = np.zeros(2)

        omega = []
        state = [self.STATE(
            Rot=np.eye(2),
            p=np.zeros(2),
            p_l=ldks
        )]

        for n in range(1, self.N):
            omega.append(self.INPUT(v, gyro))
            state.append(self.f(state[n-1], omega[n-1], w, self.dt))
            omega[n-1].v = omega[n-1].v + odo_std[0] * np.random.randn(1)
            omega[n-1].gyro = omega[n-1].gyro + odo_std[1] * np.random.randn(1)
        return state, omega, ldks

    def simu_h(self, states, obs_std, ldks):
        ys = np.zeros((self.N, self.N_ldk, 3))
        ys[:, :, 2] = -1
        for n in range(self.N):
            Rot = states[n].Rot
            p = states[n].p
            for i in range(self.N_ldk):
                p_l = ldks[i]
                r = np.linalg.norm(p_l - p)
                if self.max_range > r > self.min_range:
                    ys[n, i, :2] = Rot.T.dot(
                        p_l-p) + obs_std*np.random.randn(2)
                    ys[n, i, 2] = i
        return ys

    def benchmark_plot(self, ukf_err, left_ukf_err, right_ukf_err, iekf_err,
                       ekf_err, ps, ukf_ps, left_ukf_ps, right_ukf_ps, 
                       ekf_ps, iekf_ps):
        def rmse(errs):
            err = np.zeros((errs.shape[1], 2))
            err[:, 0] = np.sqrt(np.mean(errs[:, :, 0]**2, axis=0))
            err[:, 1] = np.sqrt(np.mean(errs[:, :, 1]**2
                                        + errs[:, :, 2]**2, axis=0))
            return err

        ukf_err = rmse(ukf_err)
        left_ukf_err = rmse(left_ukf_err)
        right_ukf_err = rmse(right_ukf_err)
        iekf_err = rmse(iekf_err)
        ekf_err = rmse(ekf_err)

        # get orientation error
        t = np.linspace(0, self.dt * self.N, self.N)

        # plot position
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set(xlabel='$y$ (m)', ylabel='$x$ (m)',
               title='Robot position for a Monte-Carlo run')
        plt.plot(ps[:, 0], ps[:, 1], linewidth=2, c='black')
        plt.plot(ukf_ps[:, 0], ukf_ps[:, 1], c='magenta')
        plt.plot(left_ukf_ps[:, 0], left_ukf_ps[:, 1], c='green')
        plt.plot(right_ukf_ps[:, 0], right_ukf_ps[:, 1], c='cyan')
        plt.plot(ekf_ps[:, 0], ekf_ps[:, 1], c='red')
        plt.plot(iekf_ps[:, 0], iekf_ps[:, 1], c='blue')
        ax.axis('equal')
        ax.legend([r'true position', r'$SO(2) \times \mathbb{R}^{2(1+L)}$ UKF',
                   r'\textbf{$SE_{1+L}(2)$ UKF (left)}',
                   r'\textbf{$SE_{1+L}(2)$ UKF (right)}', r'EKF', 
                   r'IEKF [BB17]'])

        # plot attitude error
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (deg)',
               title='Robot orientation error (deg)')

        # error
        plt.plot(t, 180/np.pi*ukf_err[:, 0], c='magenta')
        plt.plot(t, 180/np.pi*left_ukf_err[:, 0], c='green')
        plt.plot(t, 180/np.pi*right_ukf_err[:, 0], c='cyan')
        plt.plot(t, 180/np.pi*ekf_err[:, 0], c='red')
        plt.plot(t, 180/np.pi*iekf_err[:, 0], c='blue')
        ax.legend([r'$SO(2) \times \mathbb{R}^{2(1+L)}$ UKF',
                   r'\textbf{$SE_{1+L}(2)$ UKF (left)}',
                   r'\textbf{$SE_{1+L}(2)$ UKF (right)}', r'EKF', r'IEKF [BB17]'])
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, t[-1])

        # plot position error
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (m)',
               title='Robot position error (m)')

        # error
        plt.plot(t, ukf_err[:, 1], c='magenta')
        plt.plot(t, left_ukf_err[:, 1], c='green')
        plt.plot(t, right_ukf_err[:, 1], c='cyan')
        plt.plot(t, ekf_err[:, 1], c='red')
        plt.plot(t, iekf_err[:, 1], c='blue')
        ax.legend([r'$SO(2) \times \mathbb{R}^{2(1+L)}$ UKF',
                   r'\textbf{$SE_{1+L}(2)$ UKF (left)}',
                   r'\textbf{$SE_{1+L}(2)$ UKF (right)}', r'EKF', 
                   r'IEKF [BB17]'])
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, t[-1])
        return ukf_err, left_ukf_err, right_ukf_err, iekf_err, ekf_err

    @staticmethod
    def benchmark_print(ukf_err, left_ukf_err, right_ukf_err, iekf_err,
                        ekf_err):
        def rmse(errs):
            return np.sqrt(np.mean(errs**2))
        ukf_err_p = '{:.2f}'.format(rmse(ukf_err[:, 1]))
        left_ukf_err_p = '{:.2f}'.format(rmse(left_ukf_err[:, 1]))
        right_ukf_err_p = '{:.2f}'.format(rmse(right_ukf_err[:, 1]))
        ekf_err_p = '{:.2f}'.format(rmse(ekf_err[:, 1]))
        iekf_err_p = '{:.2f}'.format(rmse(iekf_err[:, 1]))

        ukf_err_rot = '{:.2f}'.format(180/np.pi*rmse(ukf_err[:, 0]))
        left_ukf_err_rot = '{:.2f}'.format(180/np.pi*rmse(left_ukf_err[:, 0]))
        right_ukf_err_rot = '{:.2f}'.format(
            180/np.pi*rmse(right_ukf_err[:, 0]))
        ekf_err_rot = '{:.2f}'.format(180/np.pi*rmse(ekf_err[:, 0]))
        iekf_err_rot = '{:.2f}'.format(180/np.pi*rmse(iekf_err[:, 0]))

        print(' ')
        print('Root Mean Square Error w.r.t. orientation (deg)')
        print("    -SO(2) x R^(2(1+L)) UKF: " + ukf_err_rot)
        print("    -left SE_{1+L}(2) UKF  : " + left_ukf_err_rot)
        print("    -right SE_{1+L}(2) UKF : " + right_ukf_err_rot)
        print("    -EKF                   : " + ekf_err_rot)
        print("    -IEKF                  : " + iekf_err_rot)

        print(' ')
        print('Root Mean Square Error w.r.t. position (m)')
        print("    -SO(2) x R^(2(1+L)) UKF: " + ukf_err_p)
        print("    -left SE_{1+L}(2) UKF  : " + left_ukf_err_p)
        print("    -right SE_{1+L}(2) UKF : " + right_ukf_err_p)
        print("    -EKF                   : " + ekf_err_p)
        print("    -IEKF                  : " + iekf_err_p)


class EKF:
    def __init__(self, state0, P0, f, h, Q, phi,
                 jacobian_propagation=None, H_num=None, aug=None, 
                 z=None, aug_z=None):
        self.state = state0
        self.P = P0
        self.f = f
        self.h = h
        self.Q = Q
        self.jacobian_propagation = jacobian_propagation
        self.H_num = H_num
        self.phi = phi

        self.new_state = self.state
        self.F = np.eye(self.P.shape[0])
        self.G = np.zeros((self.P.shape[0], self.Q.shape[0]))
        self.H = np.zeros((0, self.P.shape[0]))
        self.r = np.zeros(0)
        self.R = np.zeros((0, 0))

        self.TOL = 1e-9
        self.q = Q.shape[0]

        #  for augmenting state
        self.z = z
        self.aug_z = aug_z
        self.aug = aug

        self.J = np.array([[0, -1],
                           [1, 0]])

    def propagation(self, omega, dt):
        self.state_propagation(omega, dt)
        self.F, self.G = self.jacobian_propagation(omega, dt)
        self.cov_propagation()

    def state_propagation(self, omega, dt):
        w = np.zeros(self.q)
        self.new_state = self.f(self.state, omega, w, dt)

    def cov_propagation(self):
        P = self.F.dot(self.P).dot(self.F.T) + self.G.dot(self.Q).dot(self.G.T)
        self.P = (P+P.T)/2
        self.state = self.new_state

    def state_update(self):
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        # gain matrix
        K = np.linalg.solve(S, self.P.dot(self.H.T).T).T

        # innovation
        xi = K.dot(self.r)

        # update state
        self.state = self.phi(self.state, xi)

        # update covariance
        P = (np.eye(self.P.shape[0])-K.dot(self.H)).dot(self.P)
        self.P = (P+P.T)/2

        # init for next update
        self.H = np.zeros((0, self.P.shape[0]))
        self.r = np.zeros(0)
        self.R = np.zeros((0, 0))

    def ekf_jacobian_update(self, y, idxs, R):
        H_idx = np.zeros((2, 5))
        H_idx[:, 0] = -self.state.Rot.T.dot(
            self.J.dot((self.state.p_l - self.state.p)))
        H_idx[:, 1:3] = -self.state.Rot.T
        H_idx[:, 3:] = self.state.Rot.T

        H = np.zeros((y.shape[0], self.P.shape[0]))
        H[:, idxs] = H_idx

        # compute residual
        r = y - self.h(self.state)
        self.H = np.vstack((self.H, H))
        self.r = np.hstack((self.r, r))
        self.R = block_diag(self.R, R)

    def iekf_jacobian_update(self, y, idxs, R):
        H_idx = np.zeros((2, 5))
        H_idx[:, 1:3] = -self.state.Rot.T
        H_idx[:, 3:] = self.state.Rot.T

        H = np.zeros((y.shape[0], self.P.shape[0]))
        H[:, idxs] = H_idx

        # compute residual
        r = y - self.h(self.state)
        self.H = np.vstack((self.H, H))
        self.r = np.hstack((self.r, r))
        self.R = block_diag(self.R, R)

    def ekf_FG_ana(self, omega, dt):
        F = np.eye(self.P.shape[0])
        F[1:3, 0] = self.state.Rot.dot(self.J).dot(np.hstack([omega.v, 0]))*dt
        G = np.zeros((self.P.shape[0], 2))
        G[1:3, 0] = self.state.Rot.dot(np.array([1, 0]))*dt
        G[0, 1] = dt
        return F, G

    def iekf_FG_ana(self, omega, dt):
        F = np.eye(self.P.shape[0])

        G = np.zeros((self.P.shape[0], 2))
        G[1:3, 0] = self.state.Rot.dot(np.array([1, 0]))*dt
        G[0, 1] = dt

        p_temp = -self.J.dot(np.hstack([np.expand_dims(self.state.p, 1),
                                        self.state.p_l.T]))
        G[1:, 1] = np.reshape(p_temp, -1, order='F') * dt
        return F, G

    def ekf_augment(self, y, aug_idxs, R):
        self.state.p_l = np.squeeze(self.state.p_l)
        HR = np.zeros((2, 3))
        HR[:2, 0] = -self.state.Rot.T.dot(self.J.dot((self.state.p_l -
                                                      self.state.p)))
        HR[:2, 1:3] = -self.state.Rot.T
        H = np.zeros((2, self.P.shape[0]))
        H[:, aug_idxs] = HR
        HL = self.state.Rot.T
        iHL = np.linalg.inv(HL)
        P_sz = -iHL.dot(H.dot(self.P))
        P_ss = iHL.dot(H.dot(self.P).dot(H.T) + R).dot(iHL.T)
        Pa = np.zeros((self.P.shape[0] + 2, self.P.shape[0] + 2))
        Pa[:self.P.shape[0], :self.P.shape[0]] = self.P
        Pa[:self.P.shape[0], self.P.shape[0]:] = P_sz.T
        Pa[self.P.shape[0]:, :self.P.shape[0]] = P_sz
        Pa[self.P.shape[0]:, self.P.shape[0]:] = P_ss
        self.P = Pa
        self.state = self.aug_z(self.state, y)
        self.H = np.zeros((0, self.P.shape[0]))
        self.r = np.zeros(0)
        self.R = np.zeros((0, 0))

    def iekf_augment(self, y, aug_idxs, R):

        HR = np.zeros((2, 3))
        HR[:2, 1:3] = -self.state.Rot.T
        H = np.zeros((2, self.P.shape[0]))
        H[:, aug_idxs] = HR

        HL = self.state.Rot.T
        iHL = np.linalg.inv(HL)
        P_ss = iHL.dot(H.dot(self.P).dot(H.T) + R).dot(iHL.T)
        P_sz = -iHL.dot(H.dot(self.P))
        Pa = np.zeros((self.P.shape[0] + 2, self.P.shape[0] + 2))
        Pa[:self.P.shape[0], :self.P.shape[0]] = self.P
        Pa[:self.P.shape[0], self.P.shape[0]:] = P_sz.T
        Pa[self.P.shape[0]:, :self.P.shape[0]] = P_sz
        Pa[self.P.shape[0]:, self.P.shape[0]:] = P_ss
        self.P = Pa
        self.state = self.aug_z(self.state, y)
        self.H = np.zeros((0, self.P.shape[0]))
        self.r = np.zeros(0)
        self.R = np.zeros((0, 0))
