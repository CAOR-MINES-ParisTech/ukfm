import numpy as np
from ukfm.geometry.so2 import SO2
from ukfm.geometry.se2 import SE2
import matplotlib.pyplot as plt
import os


class LOCALIZATION:
    """2D Robot localization based on odometry and GNSS (robot position) 
    measurements. See a text description in :cite:`barrauInvariant2017`, Section
    IV.

    :var T: sequence time (s).
    :var odo_freq: odometry frequency (Hz).
    """

    J = np.array([[0, -1],
                  [1, 0]])
    data_dir = os.path.join(os.path.dirname(__file__), "../../examples/data/")

    class STATE:
        """State of the system.

        It represents the orientation and the position of the robot.

        .. math::

            \\boldsymbol{\\chi} \in \\mathcal{M} = \\left\\{ \\begin{matrix} 
           \\mathbf{C} \in SO(2),
            \\mathbf{p} \in \\mathbb R^2
           \\end{matrix} \\right\\}

        :ivar Rot: rotation matrix :math:`\mathbf{C}`.
        :ivar p: position vector :math:`\mathbf{p}`.
        """

        def __init__(self, Rot, p):
            self.Rot = Rot
            self.p = p

    class INPUT:
        """Input of the propagation model.

        The input are the robot velocities that can be obtained from a
        differential wheel system.

        .. math::

            \\boldsymbol{\\omega} \in \\mathcal{U} = \\left\\{ \\begin{matrix}
            \\mathbf{v} \in \\mathbb R^2,
            \\omega \in \\mathbb R 
            \\end{matrix} \\right\\}

        :ivar v: robot forward and lateral velocities :math:`\\mathbf{v}`.
        :ivar gyro: robot orientation velocity :math:`\\omega`.
        """

        def __init__(self, v, gyro):
            self.v = v
            self.gyro = gyro

    def __init__(self, T, odo_freq):
        # sequence time (s)
        self.T = 40
        # odometry frequency (Hz)
        self.odo_freq = 100
        # total number of timestamps
        self.N = T*odo_freq
        # integration step (s)
        self.dt = 1/odo_freq

    @classmethod
    def f(cls, state, omega, w, dt):
        """ Propagation function.

        .. math::

            \\mathbf{C}_{n+1}  &= \\mathbf{C}_{n} \\exp\\left(\\left(\\omega + 
            \\mathbf{w}^{(2)} \\right) dt\\right)  \\\\
            \\mathbf{p}_{n+1}  &= \\mathbf{p}_{n} + \\left( \\mathbf{v}_{n} + 
            \\mathbf{w}^{(0:2)} \\right) dt

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var w: noise :math:`\\mathbf{w}`.
        :var dt: integration step :math:`dt` (s).
        """
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO2.exp((omega.gyro + w[2])*dt)),
            p=state.p + state.Rot.dot(omega.v + w[:2])*dt
        )
        return new_state

    @classmethod
    def h(cls, state):
        """ Observation function.

        .. math::

            h\\left(\\boldsymbol{\\chi}\\right)  = \\mathbf{p}

        :var state: state :math:`\\boldsymbol{\\chi}`.
        """
        y = state.p
        return y

    @classmethod
    def phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) = 
          \\left( \\begin{matrix}
            \\mathbf{C} \\exp\\left(\\boldsymbol{\\xi}^{(0)}\\right) \\\\
            \\mathbf{p} + \\boldsymbol{\\xi}^{(1:3)}
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(2) 
        \\times \\mathbb R^2`.

        Its corresponding inverse operation is
        :meth:`~ukfm.LOCALIZATION.phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO2.exp(xi[0])),
            p=state.p + xi[1:]
        )
        return new_state

    @classmethod
    def phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}\\left(\\boldsymbol{\\chi}
          \\right) = \\left( \\begin{matrix}
            \\log\\left(\\mathbf{C} \\mathbf{\\hat{C}}^T\\right) \\\\
            \\mathbf{p} - \\mathbf{\\hat{p}} 
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(2)
        \\times \\mathbb R^2`.

        Its corresponding retraction is :meth:`~ukfm.LOCALIZATION.phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        xi = np.hstack([SO2.log(state.Rot.dot(hat_state.Rot.T)),
                        state.p - hat_state.p])
        return xi

    @classmethod
    def left_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) = 
          \\left( \\begin{matrix}
               \\mathbf{C} \\mathbf{C}_\\mathbf{T} \\\\
                \\mathbf{p} + \\mathbf{C} \\mathbf{r}
          \\end{matrix} \\right)

        where

        .. math::
                \\mathbf{T} = \\exp\\left(\\boldsymbol{\\xi}\\right) = 
                \\begin{bmatrix}
                    \\mathbf{C}_\\mathbf{T} & \\mathbf{r} \\\\
                    \\mathbf{0}^T & 1
                \\end{bmatrix}

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE(2)`
        with left multiplication.

        Its corresponding inverse operation is
        :meth:`~ukfm.LOCALIZATION.left_phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        chi = SE2.exp(xi)
        new_state = cls.STATE(
            Rot=state.Rot.dot(chi[:2, :2]),
            p=state.p + state.Rot.dot(chi[:2, 2])
        )
        return new_state

    @classmethod
    def left_phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}\\left(\\boldsymbol{\\chi}
          \\right) = \\log\\left(
              \\boldsymbol{\chi}^{-1} \\boldsymbol{\\hat{\\chi}} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE(2)` 
        with left multiplication.

        Its corresponding retraction is :meth:`~ukfm.LOCALIZATION.left_phi`.

        :arg state: state :math:`\\boldsymbol{\\chi}`.
        :arg hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        chi = cls.state2chi(state)
        hat_chi = cls.state2chi(hat_state)
        xi = SE2.log(SE2.inv(chi).dot(hat_chi))
        return xi

    @classmethod
    def right_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) = 
          \\left( \\begin{matrix}
                \\mathbf{C}_\\mathbf{T} \\mathbf{C}  \\\\
                 \\mathbf{C}_\\mathbf{T}\\mathbf{p} +  \\mathbf{r}
             \\end{matrix} \\right)

        where

        .. math::
        
                \\mathbf{T} = \\exp\\left(\\boldsymbol{\\xi}\\right) 
                = \\begin{bmatrix}
                    \\mathbf{C}_\\mathbf{T} &\\mathbf{r} \\\\
                    \\mathbf{0}^T & 1
                \\end{bmatrix}

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE(2)` 
        with right multiplication.

        Its corresponding inverse operation is 
        :meth:`~ukfm.LOCALIZATION.right_phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        chi = SE2.exp(xi)
        new_state = cls.STATE(
            Rot=chi[:2, :2].dot(state.Rot),
            p=chi[:2, 2] + chi[:2, :2].dot(state.p)
        )
        return new_state

    @classmethod
    def right_phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}\\left(\\boldsymbol{\\chi}
          \\right) = \\log\\left(
              \\boldsymbol{\\hat{\\chi}}^{-1} \\boldsymbol{\\chi} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE(2)` 
        with right multiplication.

        Its corresponding retraction is :meth:`~ukfm.LOCALIZATION.right_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        chi = cls.state2chi(state)
        hat_chi = cls.state2chi(hat_state)
        xi = SE2.log(hat_chi.dot(SE2.inv(chi)))
        return xi

    @classmethod
    def ekf_FG_ana(cls, state, omega, dt):
        F = np.eye(3)
        F[1:, 0] = state.Rot.dot(cls.J).dot(omega.v)*dt
        G = dt*np.eye(3)
        G[:2, :2] = dt*state.Rot
        return F, G

    @classmethod
    def ekf_H_ana(cls, state):
        H = np.hstack([np.zeros((2, 1)), np.eye(2)])
        return H

    @classmethod
    def iekf_FG_ana(cls, state, omega, dt):
        F = np.eye(3)
        F[1:, 0] = cls.J.dot(omega.v)*dt
        F[1, 2] = omega.gyro*dt
        F[2, 1] = -omega.gyro*dt
        G = dt*np.eye(3)
        return F, G

    @classmethod
    def iekf_H_ana(cls, state):
        H = state.Rot.dot(np.hstack([np.zeros((2, 1)), np.eye(2)]))
        return H

    def simu_f(self, odo_std, radius):
        # set input
        v = np.array([2*np.pi*radius/self.T, 0])  # forward speed (m/s)
        gyro = 2*np.pi/self.T  # angular speed (rad/s)

        # set noise to zero to compute the true trajectory
        w = np.zeros(3)

        # init variables at zero and do for loop
        omegas = []
        states = [self.STATE(np.eye(2), np.zeros(2))]

        for n in range(1, self.N):
            # true input
            omegas.append(self.INPUT(v, gyro))
            # propagate state
            states.append(self.f(states[n-1], omegas[n-1], w, self.dt))
            # noisy input
            omegas[n-1].v = omegas[n-1].v + odo_std[:2]*np.random.randn(2)
            omegas[n-1].gyro = omegas[n-1].gyro + odo_std[2]*np.random.randn(1)
        return states, omegas

    def simu_h(self, states, gps_freq, gps_std):
        # vector to identify when measurements happen
        one_hot_ys = np.zeros(self.N)
        one_hot_ys[::int(self.odo_freq/gps_freq)] = 1
        # odo_freq/gps_freq must be integer
        idxs = np.where(one_hot_ys == 1)[0]  # indexes where measurement happen

        # total number of measurements
        K = idxs.shape[0]
        ys = np.zeros((K, 2))
        for k in range(K):
            ys[k] = states[idxs[k]].p + gps_std*np.random.randn(2)
        return ys, one_hot_ys

    @classmethod
    def state2chi(cls, state):
        chi = np.eye(3)
        chi[:2, :2] = state.Rot
        chi[:2, 2] = state.p
        return chi

    @classmethod
    def get_states(cls, states, N):
        Rots = np.zeros((N, 2, 2))
        ps = np.zeros((N, 2))
        for n in range(N):
            Rots[n] = states[n].Rot
            ps[n] = states[n].p
        return Rots, ps

    def errors(self, Rots, hat_Rots, ps, hat_ps):
        errors = np.zeros((self.N, 3))
        for n in range(self.N):
            errors[n, 0] = SO2.log(Rots[n].T.dot(hat_Rots[n]))
        errors[:, 1:3] = ps-hat_ps
        return errors

    def plot_results(self, ukf_states, ukf_Ps, states, ys):

        t = np.linspace(0, self.T, self.N)

        Rots, ps = self.get_states(states, self.N)
        ukf_Rots, ukf_ps = self.get_states(ukf_states, self.N)
        errors = self.errors(Rots, ukf_Rots, ps, ukf_ps)
        # add position
        errors[:, 1] = np.sqrt(errors[:, 1]**2 + errors[:, 2]**2)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)', title="Robot position")
        plt.plot(ps[:, 0], ps[:, 1], linewidth=2, c='black')
        ax.scatter(ys[:, 0], ys[:, 1], c='red')
        plt.plot(ukf_ps[:, 0], ukf_ps[:, 1], c='blue')
        ax.legend([r'true position', r'UKF', r'GPS measurements'])
        ax.axis('equal')

        ukf3sigma = 3*np.sqrt(ukf_Ps[:, 0, 0])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (deg)',
               title="Robot orientation error (deg)")
        plt.plot(t, 180/np.pi*errors[:, 0], c='blue')
        plt.plot(t, 180/np.pi*ukf3sigma, c='blue', linestyle='dashed')
        plt.plot(t, 180/np.pi*(-ukf3sigma), c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1])

        ukf3sigma = 3*np.sqrt(ukf_Ps[:, 2, 2] + ukf_Ps[:, 1, 1])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (m)',
               title='Robot position error (m)')
        plt.plot(t, errors[:, 1], c='blue')
        plt.plot(t, ukf3sigma, c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1])
        ax.set_ylim(bottom=0)

    def plot_traj(self, states, ys):
        Rots, ps = self.get_states(states, self.N)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)', title="Robot position")
        plt.plot(ps[:, 0], ps[:, 1], linewidth=2, c='black')
        ax.scatter(ys[:, 0], ys[:, 1], c='red')
        ax.legend([r'true position',
                   r'GPS measurements'])
        ax.axis('equal')

    def benchmark_plot(self, ukf_err, left_ukf_err, right_ukf_err, iekf_err,
                       ekf_err, ps, ukf_ps, left_ukf_ps, right_ukf_ps, ekf_ps,
                       iekf_ps):
        def rmse(errs):
            err = np.zeros((errs.shape[1], 2))
            err[:, 0] = np.sqrt(np.mean(errs[:, :, 0]**2, axis=0))
            err[:, 1] = np.sqrt(np.mean(errs[:, :, 1]**2
                                        + errs[:, :, 2]**2, axis=0))
            return err

        ukf_e = rmse(ukf_err)
        left_ukf_e = rmse(left_ukf_err)
        right_ukf_e = rmse(right_ukf_err)
        iekf_e = rmse(iekf_err)
        ekf_e = rmse(ekf_err)

        # get orientation error
        t = np.linspace(0,  self.dt * self.N,  self.N)

        # plot position
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$y$ (m)', ylabel='$x$ (m)',
               title='Robot position for a Monte-Carlo run')
        plt.plot(ps[:, 0], ps[:, 1], linewidth=2, c='black')
        plt.plot(ukf_ps[:, 0], ukf_ps[:, 1], c='magenta')
        plt.plot(left_ukf_ps[:, 0], left_ukf_ps[:, 1], c='green')
        plt.plot(right_ukf_ps[:, 0], right_ukf_ps[:, 1], c='cyan')
        plt.plot(ekf_ps[:, 0], ekf_ps[:, 1], c='red')
        plt.plot(iekf_ps[:, 0], iekf_ps[:, 1], c='blue')
        ax.axis('equal')
        ax.legend([r'true position', r'$SO(2) \times \mathbb{R}^2$ UKF',
                   r'\textbf{$SE(2)$ UKF (left)}',
                   r'\textbf{$SE(2)$ UKF (right)}', r'EKF', r'IEKF [BB17]'])

        # plot attitude error
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (deg)',
               title='Robot orientation error (deg)')

        # error
        plt.plot(t, 180/np.pi*ukf_e[:, 0], c='magenta')
        plt.plot(t, 180/np.pi*left_ukf_e[:, 0], c='green')
        plt.plot(t, 180/np.pi*right_ukf_e[:, 0], c='cyan')
        plt.plot(t, 180/np.pi*ekf_e[:, 0], c='red')
        plt.plot(t, 180/np.pi*iekf_e[:, 0], c='blue')
        ax.legend([r'$SO(2) \times \mathbb{R}^2$ UKF',
                   r'\textbf{$SE(2)$ UKF (left)}',
                   r'\textbf{$SE(2)$ UKF (right)}', r'EKF', r'IEKF [BB17]'])
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, t[-1])

        # plot position error
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (m)',
               title='Robot position error (m)')

        # error
        plt.plot(t, ukf_e[:, 1], c='magenta')
        plt.plot(t, left_ukf_e[:, 1], c='green')
        plt.plot(t, right_ukf_e[:, 1], c='cyan')
        plt.plot(t, ekf_e[:, 1], c='red')
        plt.plot(t, iekf_e[:, 1], c='blue')
        ax.legend([r'$SO(2) \times \mathbb{R}^2$ UKF',
                   r'\textbf{$SE(2)$ UKF (left)}', 
                   r'\textbf{$SE(2)$ UKF (right)}',
                   r'EKF', r'IEKF [BB17]'])
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, t[-1])
        return ukf_e, left_ukf_e, right_ukf_e, iekf_e, ekf_e

    def benchmark_print(self, ukf_e, left_ukf_e, right_ukf_e, iekf_e, ekf_e):
        def rmse(errs):
            return np.sqrt(np.mean(errs**2))
        ukf_err_p = '{:.2f}'.format(rmse(ukf_e[:, 1]))
        left_ukf_err_p = '{:.2f}'.format(rmse(left_ukf_e[:, 1]))
        right_ukf_err_p = '{:.2f}'.format(rmse(right_ukf_e[:, 1]))
        ekf_err_p = '{:.2f}'.format(rmse(ekf_e[:, 1]))
        iekf_err_p = '{:.2f}'.format(rmse(iekf_e[:, 1]))

        ukf_err_rot = '{:.2f}'.format(180/np.pi*rmse(ukf_e[:, 0]))
        left_ukf_err_rot = '{:.2f}'.format(180/np.pi*rmse(left_ukf_e[:, 0]))
        right_ukf_err_rot = '{:.2f}'.format(180/np.pi*rmse(right_ukf_e[:, 0]))
        ekf_err_rot = '{:.2f}'.format(180/np.pi*rmse(ekf_e[:, 0]))
        iekf_err_rot = '{:.2f}'.format(180/np.pi*rmse(iekf_e[:, 0]))

        print(' ')
        print('Root Mean Square Error w.r.t. orientation (deg)')
        print("    -SO(2) x R^2 UKF: " + ukf_err_rot)
        print("    -left SE(2) UKF : " + left_ukf_err_rot)
        print("    -right SE(2) UKF: " + right_ukf_err_rot)
        print("    -EKF            : " + ekf_err_rot)
        print("    -IEKF           : " + iekf_err_rot)

        print(' ')
        print('Root Mean Square Error w.r.t. position (m)')
        print("    -SO(2) x R^2 UKF: " + ukf_err_p)
        print("    -left SE(2) UKF : " + left_ukf_err_p)
        print("    -right SE(2) UKF: " + right_ukf_err_p)
        print("    -EKF            : " + ekf_err_p)
        print("    -IEKF           : " + iekf_err_p)

    def nees(self, err, Ps, Rots, ps, name):

        neess = np.zeros((self.N, 2))
        J = np.eye(3)

        def err2nees(err, P):
            # separate orientation and position
            nees_Rot = err[0]**2 / P[0, 0]
            nees_p = err[1:3].dot(np.linalg.inv(P[1:3, 1:3]).dot(err[1:3]))/2
            return np.array([nees_Rot, nees_p])

        # only after 20 s
        for n in range(2000, self.N):
            # covariance need to be turned
            if name == 'STD':
                P = Ps[n]
            elif name == 'LEFT':
                J[1:3, 1:3] = Rots[n]
                P = J.dot(Ps[n]).dot(J.T)
            else:
                J[1:3, 0] = self.J.dot(ps[n])
                P = J.dot(Ps[n]).dot(J.T)
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

        m = 2000

        # plot orientation nees
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='orientation NEES',
               title='Robot orientation NEES', yscale="log")

        plt.plot(t[m:], ukf_nees[m:, 0], c='magenta')
        plt.plot(t[m:], left_ukf_nees[m:, 0], c='green')
        plt.plot(t[m:], right_ukf_nees[m:, 0], c='cyan')
        plt.plot(t[m:], ekf_nees[m:, 0], c='red')
        plt.plot(t[m:], iekf_nees[m:, 0], c='blue')
        ax.legend([r'$SO(2) \times \mathbb{R}^2$ UKF',
                   r'\textbf{$SE(2)$ UKF (left)}', 
                   r'\textbf{$SE(2)$ UKF (right)}',
                   r'EKF', r'IEKF [BB17]'])
        ax.set_xlim(t[m], t[-1])

        # plot position nees
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='position NEES',
               title='Robot position NEES', yscale="log")

        plt.plot(t[m:], ukf_nees[m:, 1], c='magenta')
        plt.plot(t[m:], left_ukf_nees[m:, 1], c='green')
        plt.plot(t[m:], right_ukf_nees[m:, 1], c='cyan')
        plt.plot(t[m:], ekf_nees[m:, 1], c='red')
        plt.plot(t[m:], iekf_nees[m:, 1], c='blue')
        ax.legend([r'$SO(2) \times \mathbb{R}^2$ UKF',
                   r'\textbf{$SE(2)$ UKF (left)}', 
                   r'\textbf{$SE(2)$ UKF (right)}',
                   r'EKF', r'IEKF [BB17]'])
        ax.set_xlim(t[m], t[-1])

        def g(x):
            # only after 20 s
            return np.mean(x[m:])

        print(' ')
        print(' Normalized Estimation Error Squared (NEES) w.r.t. orientation')
        print("    -SO(2) x R^2 UKF: % .2f " % g(ukf_nees[:, 0]))
        print("    -left SE(2) UKF : % .2f " % g(left_ukf_nees[:, 0]))
        print("    -right SE(2) UKF: % .2f " % g(right_ukf_nees[:, 0]))
        print("    -EKF            : % .2f " % g(ekf_nees[:, 0]))
        print("    -IEKF           : % .2f " % g(iekf_nees[:, 0]))

        print(' ')
        print(' Normalized Estimation Error Squared (NEES) w.r.t. position')
        print("    -SO(2) x R^2 UKF: % .2f " % g(ukf_nees[:, 1]))
        print("    -left SE(2) UKF : % .2f " % g(left_ukf_nees[:, 1]))
        print("    -right SE(2) UKF: % .2f " % g(right_ukf_nees[:, 1]))
        print("    -EKF            : % .2f " % g(ekf_nees[:, 1]))
        print("    -IEKF           : % .2f " % g(iekf_nees[:, 1]))

    @classmethod
    def load(cls, n_sequence, gps_freq, gps_std):

        f_name = os.path.join(cls.data_dir, "wifibot"
                              + str(n_sequence) + ".txt")
        data = np.genfromtxt(f_name, skip_header=1)
        t = data[:, 0]
        gyros = data[:, 1]
        vs = data[:, 2:4]
        thetas = data[:, 4]
        ps = data[:, 5:7]
        states = []
        omegas = []

        for n in range(t.shape[0]):
            states.append(cls.STATE(Rot=SO2.exp(thetas[n]),
                                    p=ps[n]))
            omegas.append(cls.INPUT(gyro=gyros[n],
                                    v=vs[n]))

        # simulate measurements
        N = t.shape[0]
        one_hot_ys = np.zeros(N)
        t_gps = t[0] + np.arange(2000)/gps_freq
        k = 1
        ys = np.zeros((t_gps.shape[0], 2))
        for n in range(N):
            if t_gps[k] <= t[n]:
                ys[k] = states[n].p + gps_std*np.random.randn(2)
                one_hot_ys[n] = 1
                k = k + 1

        ys = ys[:k]
        return states, omegas, ys, one_hot_ys, t

    @classmethod
    def plot_wifibot(cls, ukf_states, ukf_Ps, states, ys, t):

        N = t.shape[0]

        Rots, ps = cls.get_states(states, N)
        ukf_Rots, ukf_ps = cls.get_states(ukf_states, N)
        errors = np.zeros((N, 2))
        for n in range(N):
            errors[n, 0] = SO2.log(Rots[n].T.dot(ukf_Rots[n]))
        errors[:, 1] = np.linalg.norm(ps - ukf_ps, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)', title="Robot position")
        plt.plot(ps[:, 0], ps[:, 1], linewidth=2, c='black')
        ax.scatter(ys[:, 0], ys[:, 1], c='red')
        plt.plot(ukf_ps[:, 0], ukf_ps[:, 1], c='blue')
        ax.legend([r'true position', r'UKF', r'GPS measurements'])
        ax.axis('equal')

        ukf3sigma = 3*np.sqrt(ukf_Ps[:, 0, 0])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (deg)',
               title="Robot orientation error (deg)")
        plt.plot(t, 180/np.pi*errors[:, 0], c='blue')
        plt.plot(t, 180/np.pi*ukf3sigma, c='blue', linestyle='dashed')
        plt.plot(t, 180/np.pi*(- ukf3sigma), c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1])

        ukf3sigma = 3*np.sqrt(ukf_Ps[:, 2, 2] + ukf_Ps[:, 1, 1])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (m)',
               title='Robot position error (m)')
        plt.plot(t, errors[:, 1], c='blue')
        plt.plot(t, ukf3sigma, c='blue', linestyle='dashed')
        plt.plot(t, -ukf3sigma, c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1])
