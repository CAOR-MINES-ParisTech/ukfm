import numpy as np
from ukfm import SO3, SEK3
import matplotlib.pyplot as plt


class INERTIAL_NAVIGATION:
    """3D inertial navigation on flat Earth, where the vehicle obtains 
    observations of known landmarks. See a text description in
    :cite:`barrauInvariant2017`, Section V.

    :arg T: sequence time (s).
    :arg imu_freq: IMU frequency (Hz).
    """

    g = np.array([0, 0, -9.82])
    "gravity vector (m/s^2) :math:`\\mathbf{g}`."

    N_ldk = 3
    ldks = np.zeros((3, 3))
    "known landmarks :math:`\\mathbf{p}_i^l,~i=1,\\ldots,L`."
    ldks[0] = np.array([0, 2, 2])
    ldks[1] = np.array([-2, -2, -2])
    ldks[2] = np.array([2, -2, -2])

    class STATE:
        """State of the system.

        It represents the state of the vehicle.

        .. math::

            \\boldsymbol{\\chi} \in \\mathcal{M} = \\left\\{ \\begin{matrix} 
           \\mathbf{C} \in SO(3),
            \\mathbf{v} \in \\mathbb R^3,
            \\mathbf{p} \in \\mathbb R^3
           \\end{matrix} \\right\\}

        :ivar Rot: rotation matrix :math:`\mathbf{C}`.
        :ivar v: velocity vector :math:`\mathbf{v}`.
        :ivar p: position vector :math:`\mathbf{p}`.
        """

        def __init__(self, Rot, v, p):
            self.Rot = Rot
            self.v = v
            self.p = p

    class INPUT:
        """Input of the propagation model.

        The input is a measurement from an Inertial Measurement Unit (IMU).

        .. math:: 

            \\boldsymbol{\\omega} \in \\mathcal{U} = \\left\\{ \\begin{matrix}
            \\mathbf{u} \in \\mathbb R^3,
            \\mathbf{a}_b \in \\mathbb R^3 
            \\end{matrix} \\right\\}

        :ivar gyro: 3D gyro :math:`\mathbf{u}`.
        :ivar acc: 3D accelerometer (measurement in body frame)
              :math:`\mathbf{a}_b`.
        """

        def __init__(self, gyro, acc):
            self.gyro = gyro
            self.acc = acc

    def __init__(self, T, imu_freq):
        # sequence time (s)
        self.T = T
        # IMU frequency (Hz)
        self.imu_freq = imu_freq
        # total number of timestamps
        self.N = T * imu_freq
        # integration step (s)
        self.dt = 1 / imu_freq

    @classmethod
    def f(cls, state, omega, w, dt):
        """ Propagation function.

        .. math::

            \\mathbf{C}_{n+1}  &= \\mathbf{C}_{n} \\exp\\left(\\left(\\mathbf{u}
            + \\mathbf{w}^{(0:3)} \\right) dt\\right),  \\\\
            \\mathbf{v}_{n+1}  &= \\mathbf{v}_{n} + \\mathbf{a}  dt, \\\\
            \\mathbf{p}_{n+1}  &= \\mathbf{p}_{n} + \\mathbf{v}_{n} dt + 
            \mathbf{a} dt^2/2,

        where

        .. math::

            \\mathbf{a}  = \\mathbf{C}_{n} \\left( \\mathbf{a}_b + 
            \\mathbf{w}^{(3:6)} \\right) + \\mathbf{g}

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var w: noise :math:`\\mathbf{w}`.
        :var dt: integration step :math:`dt` (s).
        """
        acc = state.Rot.dot(omega.acc + w[3:6]) + cls.g
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO3.exp((omega.gyro + w[:3])*dt)),
            v=state.v + acc*dt,
            p=state.p + state.v*dt + 1/2*acc*dt**2
        )
        return new_state

    @classmethod
    def h(cls, state):
        """ Observation function.

        .. math::

            h\\left(\\boldsymbol{\\chi}\\right)  = \\begin{bmatrix} 
            \\mathbf{C}^T \\left( \\mathbf{p} - \\mathbf{p}^l_1\\right) \\\\
            \\vdots \\\\
            \\mathbf{C}^T \\left( \\mathbf{p} - \\mathbf{p}^l_L\\right)
            \\end{bmatrix}

        where :math:`\\mathbf{p}^l_i \in \\mathbb R^3,~i=1,\\ldots,L` are known 
        landmarks.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        """
        y = np.zeros(cls.N_ldk*3)
        for i in range(cls.N_ldk):
            y[3*i: 3*(i+1)] = state.Rot.T.dot(cls.ldks[i] - state.p)
        return y

    @classmethod
    def phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) =
          \\left( \\begin{matrix}
            \\mathbf{C} \\exp\\left(\\boldsymbol{\\xi}^{(0:3)}\\right) \\\\
            \\mathbf{v} + \\boldsymbol{\\xi}^{(3:6)} \\\\
            \\mathbf{p} + \\boldsymbol{\\xi}^{(6:9)}
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3) 
        \\times \\mathbb R^6`.

        Its corresponding inverse operation is 
        :meth:`~ukfm.INERTIAL_NAVIGATION.phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """

        new_state = cls.STATE(
            Rot=SO3.exp(xi[:3]).dot(state.Rot),
            v=state.v + xi[3:6],
            p=state.p + xi[6:9]
        )
        return new_state

    @classmethod
    def phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}\\left(\\boldsymbol{\\chi}
          \\right) = \\left( \\begin{matrix}
            \\log\\left(\\mathbf{C} \\mathbf{\\hat{C}}^T \\right)\\\\
            \\mathbf{v} - \\mathbf{\\hat{v}} \\\\
            \\mathbf{p} - \\mathbf{\\hat{p}} 
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3)
        \\times \\mathbb R^6`.

        Its corresponding retraction is :meth:`~ukfm.INERTIAL_NAVIGATION.phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        xi = np.hstack([SO3.log(state.Rot.dot(hat_state.Rot.T)),
                        state.v - hat_state.v,
                        state.p - hat_state.p])
        return xi

    @classmethod
    def left_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) = 
          \\left( \\begin{matrix}
            \\mathbf{C} \\mathbf{C}_\\mathbf{T} \\\\
            \\mathbf{v} + \\mathbf{C} \\mathbf{r_1} \\\\
            \\mathbf{p} + \\mathbf{C} \\mathbf{r_2}
          \\end{matrix} \\right)

        where

        .. math::
            \\mathbf{T} = \\exp\\left(\\boldsymbol{\\xi}\\right) =
            \\begin{bmatrix}
                \\mathbf{C}_\\mathbf{T} & \\mathbf{r_1}  &\\mathbf{r}_2 \\\\
                \\mathbf{0}^T & & \\mathbf{I} 
            \\end{bmatrix}

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)`
        with left multiplication.

        Its corresponding inverse operation is
        :meth:`~ukfm.INERTIAL_NAVIGATION.left_phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """

        T = SEK3.exp(xi)
        new_state = cls.STATE(
            Rot=state.Rot.dot(T[:3, :3]),
            v=state.Rot.dot(T[:3, 3]) + state.v,
            p=state.Rot.dot(T[:3, 4]) + state.p
        )
        return new_state

    @classmethod
    def left_phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}
          \\left(\\boldsymbol{\\chi}\\right) = 
          \\log\\left(
            \\boldsymbol{\chi}^{-1} \\boldsymbol{\\hat{\\chi}} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)`
        with left multiplication.

        Its corresponding retraction is
        :meth:`~ukfm.INERTIAL_NAVIGATION.left_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        chi = cls.state2chi(state)
        hat_chi = cls.state2chi(hat_state)
        xi = SEK3.log(SEK3.inv(chi).dot(hat_chi))
        return xi

    @classmethod
    def right_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) =
           \\left( \\begin{matrix}
            \\mathbf{C}_\\mathbf{T} \\mathbf{C}  \\\\
            \\mathbf{C}_\\mathbf{T}\\mathbf{v} +  \\mathbf{r_1} \\\\
            \\mathbf{C}_\\mathbf{T}\\mathbf{p} +  \\mathbf{r_2}
           \\end{matrix} \\right)

        where

        .. math::
            \\mathbf{T} = \\exp\\left(\\boldsymbol{\\xi}\\right) = 
            \\begin{bmatrix}
                \\mathbf{C}_\\mathbf{T} & \\mathbf{r_1}  &\\mathbf{r}_2 \\\\
                \\mathbf{0}^T & & \\mathbf{I} 
            \\end{bmatrix}

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)`
        with right multiplication.

        Its corresponding inverse operation is 
        :meth:`~ukfm.INERTIAL_NAVIGATION.right_phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        chi = SEK3.exp(xi)
        new_state = cls.STATE(
            Rot=chi[:3, :3].dot(state.Rot),
            v=chi[:3, :3].dot(state.v) + chi[:3, 3],
            p=chi[:3, :3].dot(state.p) + chi[:3, 4]
        )
        return new_state

    @classmethod
    def right_phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}\\left(\\boldsymbol{\\chi}
          \\right) = \\log\\left(
            \\boldsymbol{\\hat{\\chi}}^{-1} \\boldsymbol{\\chi} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)`
        with right multiplication.

        Its corresponding retraction is
        :meth:`~ukfm.INERTIAL_NAVIGATION.right_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        chi = cls.state2chi(state)
        hat_chi = cls.state2chi(hat_state)
        xi = SEK3.log(hat_chi.dot(SEK3.inv(chi)))
        return xi

    @classmethod
    def state2chi(cls, state):
        chi = np.eye(5)
        chi[:3, :3] = state.Rot
        chi[:3, 3] = state.v
        chi[:3, 4] = state.p
        return chi

    @classmethod
    def ekf_FG_ana(cls, state, omega, dt):
        F = np.eye(9)
        F[3:6, :3] = -SO3.wedge(state.Rot.dot(omega.acc)*dt)
        F[6:9, :3] = F[3:6, :3]*dt/2
        F[6:9, 3:6] = dt*np.eye(3)

        G = np.zeros((9, 6))
        G[:3, :3] = state.Rot*dt
        G[3:6, 3:6] = state.Rot*dt
        return F, G

    @classmethod
    def ekf_H_ana(cls, state):
        H = np.zeros((3 * cls.N_ldk, 9))
        for i in range(cls.N_ldk):
            H[3*i: 3*(i+1), :3] = state.Rot.T.dot(SO3.wedge(cls.ldks[i] -
                                                            state.p))
            H[3*i: 3*(i+1), 6:9] = -state.Rot.T
        return H

    @classmethod
    def iekf_FG_ana(cls, state, omega, dt):
        F = np.eye(9)
        F[3:6, :3] = SO3.wedge(cls.g)*dt
        F[6:9, :3] = F[3:6, :3]*dt/2
        F[6:9, 3:6] = dt*np.eye(3)

        G = np.zeros((9, 6))
        G[:3, :3] = state.Rot*dt
        G[3:6, 3:6] = state.Rot*dt
        G[3:6, :3] = SO3.wedge(state.v).dot(state.Rot)*dt
        G[6:9, :3] = SO3.wedge(state.p).dot(state.Rot)*dt
        return F, G

    @classmethod
    def iekf_H_ana(cls, state):
        H = np.zeros((3 * cls.N_ldk, 9))
        for i in range(cls.N_ldk):
            H[3*i: 3*(i+1), :3] = state.Rot.T.dot(SO3.wedge(cls.ldks[i]))
            H[3*i: 3*(i+1), 6:9] = -state.Rot.T
        return H

    def simu_f(self, imu_std):
        # rayon (m)
        r = 5

        # set noise to zero to compute true trajectory
        w = np.zeros(6)

        # compute acceleration from trajectory
        t = np.linspace(0, self.T, self.N)
        p = r * np.vstack([np.sin(t / self.T * 2 * np.pi),
                           np.cos(t / self.T * 2 * np.pi), np.zeros(self.N)])
        v = np.hstack([np.zeros((3, 1)), np.diff(p)]) / self.dt
        acc = np.hstack([np.zeros((3, 1)), np.diff(v)]) / self.dt

        # init variables at zero and do for loop
        omegas = []
        states = [self.STATE(np.eye(3), v[:, 0], p[:, 0])]

        for n in range(1, self.N):
            # true input
            omegas.append(self.INPUT(
                gyro=np.zeros(3),
                acc=states[n-1].Rot.T.dot(acc[:, n-1] - self.g)
            ))
            # propagate state
            states.append(self.f(states[n-1], omegas[n-1], w, self.dt))
            # noisy input
            omegas[n-1].gyro = omegas[n-1].gyro + \
                imu_std[0] * np.random.randn(3)
            omegas[n-1].acc = omegas[n-1].acc + imu_std[1] * np.random.randn(3)
        return states, omegas

    def simu_h(self, states, obs_freq, obs_std):
        # vector to know where measurement happen
        one_hot_ys = np.zeros(self.N)
        # imu_freq/obs_freq must be integer
        one_hot_ys[::int(self.imu_freq / obs_freq)] = 1
        idxs = np.where(one_hot_ys == 1)[0]  # indexes where measurement happen
        # total number of measurements
        K = idxs.shape[0]

        # measurement iteration number
        ys = np.zeros((K, 3*self.N_ldk))
        for k in range(K):
            ys[k] = self.h(states[idxs[k]]) + obs_std * \
                np.random.randn(3*self.N_ldk)
        return ys, one_hot_ys

    def errors(self, Rots, vs, ps, hat_Rots, hat_vs, hat_ps):
        errors = np.zeros((self.N, 9))
        for n in range(self.N):
            errors[n, :3] = SO3.log(Rots[n].T.dot(hat_Rots[n]))
        errors[:, 3:6] = vs - hat_vs
        errors[:, 6:9] = ps - hat_ps
        return errors

    def plot_results(self, hat_states, hat_P, states):
        Rots, vs, ps = self.get_states(states, self.N)
        hat_Rots, hat_vs, hat_ps = self.get_states(hat_states, self.N)

        errors = self.errors(Rots, vs, ps, hat_Rots, hat_vs, hat_ps)

        errors[:, 0] = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2 +
                               errors[:, 2]**2)
        errors[:, 1] = np.sqrt(errors[:, 3]**2 + errors[:, 4]**2 +
                               errors[:, 5]**2)
        errors[:, 2] = np.sqrt(errors[:, 6]**2 + errors[:, 7]**2 +
                               errors[:, 8]**2)

        t = np.linspace(0, self.T, self.N)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)',
               title="Horizontal vehicle position")
        ax.scatter(self.ldks[:, 0], self.ldks[:, 1], c='red')
        plt.plot(ps[:, 0], ps[:, 1], linewidth=2, c='black')
        plt.plot(hat_ps[:, 0], hat_ps[:, 1], c='blue')
        ax.legend([r'true trajectory', r'UKF', r'features'])

        ukf3sigma = 3 * \
            np.sqrt(hat_P[:, 0, 0] + hat_P[:, 1, 1] + hat_P[:, 2, 2])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (deg)',
               title='Attitude error (deg)')
        plt.plot(t, 180/np.pi*errors[:, 0], c='blue')
        plt.plot(t, 180/np.pi*ukf3sigma, c='blue', linestyle='dashed')
        plt.plot(t, -180/np.pi*ukf3sigma, c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1])

        ukf3sigma = 3 * \
            np.sqrt(hat_P[:, 6, 6]+ hat_P[:, 7, 7] + hat_P[:, 8, 8])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (m)',
               title="Position error (m)")

        plt.plot(t, errors[:, 2], c='blue')
        plt.plot(t, ukf3sigma, c='blue', linestyle='dashed')
        plt.plot(t, -ukf3sigma, c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1])

    def benchmark_plot(self, ukf_err, left_ukf_err, right_ukf_err, iekf_err,
                       ekf_err, ps, ukf_ps, left_ukf_ps, right_ukf_ps, ekf_ps,
                       iekf_ps):

        def rmse(errs):
            err = np.zeros((errs.shape[1], 3))
            err[:, 0] = np.sqrt(np.mean(errs[:, :, 0]**2 +
                                        errs[:, :, 1]**2 + errs[:, :, 2]**2,
                                        axis=0))
            err[:, 1] = np.sqrt(np.mean(errs[:, :, 3]**2 +
                                        errs[:, :, 4]**2 + errs[:, :, 5]**2,
                                        axis=0))
            err[:, 2] = np.sqrt(np.mean(errs[:, :, 6]**2 +
                                        errs[:, :, 7]**2 + errs[:, :, 8]**2,
                                        axis=0))
            return err

        ukf_err = rmse(ukf_err)
        left_ukf_err = rmse(left_ukf_err)
        right_ukf_err = rmse(right_ukf_err)
        iekf_err = rmse(iekf_err)
        ekf_err = rmse(ekf_err)

        # get orientation error
        t = np.linspace(0, self.dt * self.N, self.N)

        # plot position
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$y$ (m)', ylabel='$x$ (m)',
               title='Vehicle position for a Monte-Carlo run')
        plt.plot(ps[:, 0], ps[:, 1], linewidth=2, c='black')
        plt.plot(ukf_ps[:, 0], ukf_ps[:, 1], c='magenta')
        plt.plot(left_ukf_ps[:, 0], left_ukf_ps[:, 1], c='green')
        plt.plot(right_ukf_ps[:, 0], right_ukf_ps[:, 1], c='cyan')
        plt.plot(ekf_ps[:, 0], ekf_ps[:, 1], c='red')
        plt.plot(iekf_ps[:, 0], iekf_ps[:, 1], c='blue')
        ax.axis('equal')
        ax.legend([r'true position', r' $SO(3) \times \mathbb{R}^6$ UKF',
                   r'\textbf{$SE_2(3)$ UKF (left)}',
                   r'\textbf{$SE_2(3)$ UKF (right)}', r'EKF', r'IEKF [BB17]'])

        # plot attitude error
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (deg)',
               title='Vehicle orientation error (deg)')

        # error
        plt.plot(t, 180/np.pi*ukf_err[:, 0], c='magenta')
        plt.plot(t, 180/np.pi*left_ukf_err[:, 0], c='green')
        plt.plot(t, 180/np.pi*right_ukf_err[:, 0], c='cyan')
        plt.plot(t, 180/np.pi*ekf_err[:, 0], c='red')
        plt.plot(t, 180/np.pi*iekf_err[:, 0], c='blue')
        ax.legend([r' $SO(3) \times \mathbb{R}^6$ UKF',
                   r'\textbf{$SE_2(3)$ UKF (left)}',
                   r'\textbf{$SE_2(3)$ UKF (right)}',
                   r'EKF', r'IEKF [BB17]'])
        ax.set_ylim(0, 8)
        ax.set_xlim(0, t[-1])

        # plot position error
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (m)',
               title='Vehicle position error (m)')

        # error
        plt.plot(t, ukf_err[:, 2], c='magenta')
        plt.plot(t, left_ukf_err[:, 2], c='green')
        plt.plot(t, right_ukf_err[:, 2], c='cyan')
        plt.plot(t, ekf_err[:, 2], c='red')
        plt.plot(t, iekf_err[:, 2], c='blue')
        ax.legend([r' $SO(3) \times \mathbb{R}^6$ UKF',
                   r'\textbf{$SE_2(3)$ UKF (left)}',
                   r'\textbf{$SE_2(3)$ UKF (right)}',
                   r'EKF', r'IEKF [BB17]'])
        ax.set_xlim(0, t[-1])
        ax.set_ylim(bottom=0)
        return ukf_err, left_ukf_err, right_ukf_err, iekf_err, ekf_err

    @staticmethod
    def benchmark_print(ukf_err, left_ukf_err, right_ukf_err, iekf_err,
                        ekf_err):
        def rmse(errs):
            return np.sqrt(np.mean(errs**2))
        ukf_err_p = '{:.2f}'.format(rmse(ukf_err[:, 2]))
        left_ukf_err_p = '{:.2f}'.format(rmse(left_ukf_err[:, 2]))
        right_ukf_err_p = '{:.2f}'.format(rmse(right_ukf_err[:, 2]))
        ekf_err_p = '{:.2f}'.format(rmse(ekf_err[:, 2]))
        iekf_err_p = '{:.2f}'.format(rmse(iekf_err[:, 2]))

        ukf_err_rot = '{:.2f}'.format(180/np.pi*rmse(ukf_err[:, 0]))
        left_ukf_err_rot = '{:.2f}'.format(180/np.pi*rmse(left_ukf_err[:, 0]))
        right_ukf_err_rot = '{:.2f}'.format(
            180/np.pi*rmse(right_ukf_err[:, 0]))
        ekf_err_rot = '{:.2f}'.format(180/np.pi*rmse(ekf_err[:, 0]))
        iekf_err_rot = '{:.2f}'.format(180/np.pi*rmse(iekf_err[:, 0]))

        print(' ')
        print('Root Mean Square Error w.r.t. orientation (deg)')
        print("    -SO(3) x R^6 UKF  : " + ukf_err_rot)
        print("    -left SE_2(3) UKF : " + left_ukf_err_rot)
        print("    -right SE_2(3) UKF: " + right_ukf_err_rot)
        print("    -EKF              : " + ekf_err_rot)
        print("    -IEKF             : " + iekf_err_rot)

        print(' ')
        print('Root Mean Square Error w.r.t. position (m)')
        print("    -SO(3) x R^6 UKF  : " + ukf_err_p)
        print("    -left SE_2(3) UKF : " + left_ukf_err_p)
        print("    -right SE_2(3) UKF: " + right_ukf_err_p)
        print("    -EKF              : " + ekf_err_p)
        print("    -IEKF             : " + iekf_err_p)

    @classmethod
    def get_states(cls, states, N):
        Rots = np.zeros((N, 3, 3))
        vs = np.zeros((N, 3))
        ps = np.zeros((N, 3))
        for n in range(N):
            Rots[n] = states[n].Rot
            vs[n] = states[n].v
            ps[n] = states[n].p
        return Rots, vs, ps

    def nees(self, err, Ps, Rots, vs, ps, name):

        J = np.eye(9)
        neess = np.zeros((self.N, 2))

        def err2nees(err, P):
            # separate orientation and position
            nees_Rot = err[:3].dot(np.linalg.inv(P[:3, :3]).dot(err[:3]))/3
            nees_p = err[6:9].dot(np.linalg.inv(P[6:9, 6:9]).dot(err[6:9]))/3
            return np.array([nees_Rot, nees_p])

        for n in range(1, self.N):
            # covariance need to be turned
            if name == 'STD':
                P = Ps[n]
            elif name == 'LEFT':
                J[3:6, 3:6] = Rots[n]
                J[6:9, 6:9] = Rots[n]
                P = J.dot(Ps[n]).dot(J.T)
            else:
                J[3:6, :3] = SO3.wedge(vs[n])
                J[6:9, :3] = SO3.wedge(ps[n])
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

        # plot orientation nees
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='orientation NEES',
               title='Vehicle orientation NEES', yscale="log")
        plt.plot(t, ukf_nees[:, 0], c='magenta')
        plt.plot(t, left_ukf_nees[:, 0], c='green')
        plt.plot(t, right_ukf_nees[:, 0], c='cyan')
        plt.plot(t, ekf_nees[:, 0], c='red')
        plt.plot(t, iekf_nees[:, 0], c='blue')
        ax.legend([r'$SO(3) \times \mathbb{R}^6$ UKF',
                   r'\textbf{$SE_2(3)$ UKF (left)}',
                   r'\textbf{$SE_2(3)$ UKF (right)}', r'EKF', r'IEKF [BB17]'])
        ax.set_xlim(0, t[-1])

        # plot position nees
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='position NEES',
               title='Vehicle position NEES', yscale="log")

        plt.plot(t, ukf_nees[:, 1], c='magenta')
        plt.plot(t, left_ukf_nees[:, 1], c='green')
        plt.plot(t, right_ukf_nees[:, 1], c='cyan')
        plt.plot(t, ekf_nees[:, 1], c='red')
        plt.plot(t, iekf_nees[:, 1], c='blue')
        ax.legend([r'$SO(3) \times \mathbb{R}^6$ UKF',
                   r'\textbf{$SE_2(3)$ UKF (left)}',
                   r'\textbf{$SE_2(3)$ UKF (right)}', r'EKF', r'IEKF [BB17]'])
        ax.set_xlim(0, t[-1])

        def g(x):
            return np.mean(x)

        print(' ')
        print(' Normalized Estimation Error Squared (NEES) w.r.t. orientation')
        print("    -SO(3) x R^6 UKF  : % .2f " % g(ukf_nees[:, 0]))
        print("    -left SE_2(3) UKF : % .2f " % g(left_ukf_nees[:, 0]))
        print("    -right SE_2(3) UKF: % .2f " % g(right_ukf_nees[:, 0]))
        print("    -EKF              : % .2f " % g(ekf_nees[:, 0]))
        print("    -IEKF             : % .2f " % g(iekf_nees[:, 0]))

        print(' ')
        print(' Normalized Estimation Error Squared (NEES) w.r.t. position')
        print("    -SO(3) x R^6 UKF  : % .2f " % g(ukf_nees[:, 1]))
        print("    -left SE_2(3) UKF : % .2f " % g(left_ukf_nees[:, 1]))
        print("    -right SE_2(3) UKF: % .2f " % g(right_ukf_nees[:, 1]))
        print("    -EKF              : % .2f " % g(ekf_nees[:, 1]))
        print("    -IEKF             : % .2f " % g(iekf_nees[:, 1]))
