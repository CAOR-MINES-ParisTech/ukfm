import numpy as np
from ukfm import SO3
import matplotlib.pyplot as plt


class ATTITUDE:
    """3D attitude estimation from an IMU equipped with gyro, accelerometer and 
    magnetometer. See text description in :cite:`kokUsing2017`, Section IV.

    :arg T: sequence time (s).
    :arg imu_freq: IMU frequency (Hz).
    """

    g = np.array([0, 0, -9.82])
    "gravity vector (m/s^2) :math:`\\mathbf{g}`."
    b = np.array([0.33, 0, -0.95])
    "normed magnetic field in Sweden :math:`\\mathbf{b}`."

    class STATE:
        """State of the system.

        It represents the orientation of the platform.

        .. math::

            \\boldsymbol{\\chi} \in \\mathcal{M} = \\left\\{ 
           \\mathbf{C} \in SO(3) \\right\\}

        :ivar Rot: rotation matrix :math:`\mathbf{C}`.
        """

        def __init__(self, Rot):
            self.Rot = Rot

    class INPUT:
        """Input of the propagation model.

        The input is the gyro measurement from an Inertial Measurement Unit
        (IMU).

        .. math:: 

            \\boldsymbol{\\omega} \in \\mathcal{U} = \\left\\{ 
            \\mathbf{u} \in \\mathbb R^3 \\right\\}

        :ivar gyro: 3D gyro :math:`\mathbf{u}`.
        """

        def __init__(self, gyro):
            self.gyro = gyro

    def __init__(self, T, imu_freq):
        # sequence time (s)
        self.T = T
        # IMU frequency (Hz)
        self.imu_freq = imu_freq
        # total number of timestamps
        self.N = T*imu_freq
        # integration step (s)
        self.dt = 1/imu_freq

    @classmethod
    def f(cls, state, omega, w, dt):
        """ Propagation function.

        .. math::

            \\mathbf{C}_{n+1}  = \\mathbf{C}_{n} 
            \\exp\\left(\\left(\\mathbf{u} + \\mathbf{w} \\right) 
            dt \\right)

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var w: noise :math:`\\mathbf{w}`.
        :var dt: integration step :math:`dt` (s).
        """
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO3.exp((omega.gyro + w)*dt)),
        )
        return new_state

    @classmethod
    def h(cls, state):
        """ Observation function.

        .. math::

            h\\left(\\boldsymbol{\\chi}\\right)  = \\begin{bmatrix} 
            \\mathbf{C}^T \\mathbf{g} \\\\
            \\mathbf{C}^T \\mathbf{b}
            \\end{bmatrix}

        :var state: state :math:`\\boldsymbol{\\chi}`.
        """
        y = np.hstack([state.Rot.T.dot(cls.g),
                       state.Rot.T.dot(cls.b)])
        return y

    @classmethod
    def phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) = 
            \\mathbf{C} \\exp\\left(\\boldsymbol{\\xi}\\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3)`
        with left multiplication.

        Its corresponding inverse operation is :meth:`~ukfm.ATTITUDE.phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO3.exp(xi))
        )
        return new_state

    @classmethod
    def phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}\\left(\\boldsymbol{\\chi}
          \\right) = \\log\\left(
            \\boldsymbol{\chi}^{-1} \\boldsymbol{\\hat{\\chi}} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3)`
        with left multiplication.

        Its corresponding retraction is :meth:`~ukfm.ATTITUDE.phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        xi = SO3.log(state.Rot.T.dot(hat_state.Rot))
        return xi

    @classmethod
    def right_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) = 
            \\exp\\left(\\boldsymbol{\\xi}\\right) \\mathbf{C} 

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3)` 
        with right multiplication.

        Its corresponding inverse operation is 
        :meth:`~ukfm.ATTITUDE.right_phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        new_state = cls.STATE(
            Rot=SO3.exp(xi).dot(state.Rot)
        )
        return new_state

    @classmethod
    def right_phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}\\left(\\boldsymbol{\\chi}
          \\right) = \\log\\left(
            \\boldsymbol{\\hat{\\chi}}\\boldsymbol{\chi}^{-1} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3)` 
        with right multiplication.

        Its corresponding retraction is :meth:`~ukfm.ATTITUDE.right_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        xi = SO3.log(hat_state.Rot.dot(state.Rot.T))
        return xi

    @classmethod
    def ekf_FG_ana(cls, state, omega, dt):
        F = np.eye(3)
        G = dt*state.Rot
        return F, G

    @classmethod
    def ekf_H_ana(cls, state):
        H = np.vstack([state.Rot.T.dot(SO3.wedge(cls.g)),
                       state.Rot.T.dot(SO3.wedge(cls.b))])
        return H

    def simu_f(self, imu_std):
        # The robot is 2 s stationary and then have constant angular velocity
        # around gravity

        n_T = 0  # increment for angular velocity
        omega_T = np.zeros(3)  # first velocity (robot is first stationary)
        omega_move = np.array([0, 0, 10/180*np.pi])

        # set noise to zero to compute true trajectory
        w = np.zeros(3)

        # init variables at zero and do for loop
        omegas = []
        states = [self.STATE(Rot=np.eye(3))]
        for n in range(1, self.N):
            # change true input
            if n_T > 2:
                omega_T = omega_move
            n_T = n_T + self.dt
            # true input
            omegas.append(self.INPUT(omega_T))
            # propagate state
            states.append(self.f(states[n-1], omegas[n-1], w, self.dt))
            # noisy input
            omegas[n-1].gyro = omegas[n-1].gyro + imu_std[0]*np.random.randn(3)
        return states, omegas

    def simu_h(self, state, imu_std):
        # total number of timestamps
        y = np.zeros((self.N, 6))
        for n in range(self.N):
            y[n, :3] = state[n].Rot.T.dot(
                self.g + imu_std[1]*np.random.randn(3))
            y[n, 3:] = state[n].Rot.T.dot(
                self.b + imu_std[2]*np.random.randn(3))
        return y

    def plot_results(self, hat_states, hat_Ps, states, omegas):
        Rots, rpys = self.get_states(states, self.N)
        hat_Rots, hat_rpys = self.get_states(hat_states, self.N)
        errors = self.errors(Rots, hat_Rots)
        t = np.linspace(0, self.T, self.N)

        ukf3sigma = 3*np.vstack([np.sqrt(hat_Ps[:, 0, 0]),
                                 np.sqrt(hat_Ps[:, 1, 1]),
                                 np.sqrt(hat_Ps[:, 2, 2])])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='orientation (deg)', 
        title='Orientation')

        plt.plot(t, 180/np.pi*rpys[:, 0], c='red')
        plt.plot(t, 180/np.pi*rpys[:, 1], c='yellow')
        plt.plot(t, 180/np.pi*rpys[:, 2], linewidth=2, c='black')
        ax.legend([r'roll', 'pitch', 'yaw'])
        ax.set_xlim(0, t[-1]) 

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='Roll error (deg)', 
            title='Roll error  (deg)')
        plt.plot(t, 180/np.pi*errors[:, 0], c='blue')
        plt.plot(t, 180/np.pi*ukf3sigma[0, :],   c='blue', linestyle='dashed')
        plt.plot(t, -180/np.pi*ukf3sigma[0, :],  c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1])

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set(xlabel='$t$ (s)', ylabel='Pitch error (deg)', 
        title='Pitch error  (deg)')
        plt.plot(t, 180/np.pi*errors[:, 1], c='blue')
        plt.plot(t, 180/np.pi*ukf3sigma[1, :],   c='blue', linestyle='dashed')
        plt.plot(t, -180/np.pi*ukf3sigma[1, :],  c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1]) 

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='Yaw error (deg)', 
            title='Yaw error  (deg)')
        plt.plot(t, 180/np.pi*errors[:, 2], c='blue')
        plt.plot(t, 180/np.pi*ukf3sigma[2, :],   c='blue', linestyle='dashed')
        plt.plot(t, -180/np.pi*ukf3sigma[2, :],  c='blue', linestyle='dashed')
        ax.legend([r'UKF', r'$3\sigma$ UKF'])
        ax.set_xlim(0, t[-1]) 

    @classmethod
    def get_states(cls, states, N):
        Rots = np.zeros((N, 3, 3))
        rpys = np.zeros((N, 3))
        for n in range(N):
            Rots[n] = states[n].Rot
            rpys[n] = SO3.to_rpy(states[n].Rot)
        return Rots, rpys

    @classmethod
    def errors(cls, Rots, hat_Rots):
        N = Rots.shape[0]
        errors = np.zeros((N, 3))
        # get true states and estimates, and orientation error
        for n in range(N):
            errors[n] = SO3.log(Rots[n].dot(hat_Rots[n].T))
        return errors

    def benchmark_print(self, left_ukf_err, right_ukf_err, ekf_err):
        def rmse(errs):
            return np.sqrt(np.mean(errs ** 2))

        def f(x):
            # average over Monte-Carlo and angles
            return np.sqrt(np.mean(np.sum(x ** 2, axis=2), axis=0))
        
        t = np.linspace(0, self.T, self.N)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='error (deg)',
               title="Orientation error(deg)")
        plt.plot(t, 180/np.pi*f(left_ukf_err), c='green')
        plt.plot(t, 180/np.pi*f(right_ukf_err), c='cyan')
        plt.plot(t, 180/np.pi*f(ekf_err), c='red')
        ax.legend([r'\textbf{left UKF}', r'\textbf{right UKF}', r'EKF'])
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, t[-1]) 

        left_ukf_err_rot = '{:.2f}'.format(180/np.pi*rmse(left_ukf_err))
        right_ukf_err_rot = '{:.2f}'.format(180/np.pi*rmse(right_ukf_err))
        ekf_err_rot = '{:.2f}'.format(180/np.pi*rmse(ekf_err))

        print(' ')
        print('Root Mean Square Error w.r.t. orientation (deg)')
        print("    -left UKF    : " + left_ukf_err_rot)
        print("    -right UKF   : " + right_ukf_err_rot)
        print("    -EKF         : " + ekf_err_rot)

    def nees(self, err, Ps, Rots, name):
        neess = np.zeros(self.N)
        def err2nees(err, P):
            return err.dot(np.linalg.inv(P).dot(err))/3

        for n in range(1, self.N):
            # covariance need to be turned
            if name == 'LEFT':
                P = Rots[n].T.dot(Ps[n]).dot(Rots[n])
            else:
                P = Ps[n]
            neess[n] = err2nees(err[n], P)
        return neess

    def nees_print(self, left_ukf_nees, right_ukf_nees, ekf_nees):

        t = np.linspace(0,  self.dt * self.N,  self.N)
        def f(x):
            return np.mean(x, axis=0)
        left_ukf_nees = f(left_ukf_nees)
        right_ukf_nees = f(right_ukf_nees)
        ekf_nees = f(ekf_nees)

        # plot orientation nees
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel='$t$ (s)', ylabel='orientation NEES',
                title='Orientation NEES', yscale="log")

        plt.plot(t, left_ukf_nees, c='green')
        plt.plot(t, right_ukf_nees, c='cyan')
        plt.plot(t, ekf_nees, c='red')
        ax.legend([r'\textbf{left UKF}', r'\textbf{right UKF}', r'EKF'])
        ax.set_xlim(0, t[-1]) 

        def g(x):
            return np.mean(x)

        print(' ')
        print(' Normalized Estimation Error Squared (NEES) w.r.t. orientation')
        print("    -left UKF    : % .2f " % g(left_ukf_nees))
        print("    -right UKF   : % .2f " % g(right_ukf_nees))
        print("    -EKF         : % .2f " % g(ekf_nees))
