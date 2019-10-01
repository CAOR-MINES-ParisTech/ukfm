import numpy as np
from ukfm import SO3, SE3
import matplotlib.pyplot as plt
import os


class IMUGNSS:
    """IMU-GNSS sensor-fusion on the KITTI dataset. The model is the standard 3D
    kinematics model based on inertial inputs and kinematics equations.
    """

    g = np.array([0, 0, -9.82])
    "gravity vector (m/s^2) :math:`\\mathbf{g}`."

    data_dir = os.path.join(os.path.dirname(__file__), "../../examples/data/")

    f_gps = "KittiGps_converted.txt"
    f_imu = "KittiEquivBiasedImu.txt"

    class STATE:
        """State of the system.

        It represents the state of a moving vehicle with IMU biases.

        .. math::

            \\boldsymbol{\\chi} \in \\mathcal{M} = \\left\\{ \\begin{matrix} 
           \\mathbf{C} \in SO(3),
            \\mathbf{v} \in \\mathbb R^3,
            \\mathbf{p} \in \\mathbb R^3,
            \\mathbf{b}_g \in \\mathbb R^3,
            \\mathbf{b}_a \in \\mathbb R^3
           \\end{matrix} \\right\\}

        :ivar Rot: rotation matrix :math:`\mathbf{C}`.
        :ivar v: velocity vector :math:`\mathbf{v}`.
        :ivar p: position vector :math:`\mathbf{p}`.
        :ivar b_gyro: gyro bias :math:`\mathbf{b}_g`.
        :ivar b_acc: accelerometer bias :math:`\mathbf{b}_a`.
        """

        def __init__(self, Rot, v, p, b_gyro, b_acc):
            self.Rot = Rot
            self.v = v
            self.p = p
            self.b_gyro = b_gyro
            self.b_acc = b_acc

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

    @classmethod
    def f(cls, state, omega, w, dt):
        """ Propagation function.

        .. math::

          \\mathbf{C}_{n+1}  &= \\mathbf{C}_{n} \\exp\\left(\\left(\\mathbf{u}
          - \mathbf{b}_g + \\mathbf{w}^{(0:3)} \\right) dt\\right)  \\\\
          \\mathbf{v}_{n+1}  &= \\mathbf{v}_{n} + \\mathbf{a}  dt, \\\\
          \\mathbf{p}_{n+1}  &= \\mathbf{p}_{n} + \\mathbf{v}_{n} dt 
          + \mathbf{a} dt^2/2 \\\\
          \\mathbf{b}_{g,n+1}  &= \\mathbf{b}_{g,n} 
          + \\mathbf{w}^{(6:9)}dt \\\\
          \\mathbf{b}_{a,n+1}  &= \\mathbf{b}_{a,n} + 
          \\mathbf{w}^{(9:12)} dt     

        where

        .. math::

            \\mathbf{a}  = \\mathbf{C}_{n} 
            \\left( \\mathbf{a}_b -\mathbf{b}_a 
            + \\mathbf{w}^{(3:6)} \\right) + \\mathbf{g}

        Ramdom-walk noises on biases are not added as the Jacobian w.r.t. these 
        noise are trivial. This spares some computations of the UKF.  

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var w: noise :math:`\\mathbf{w}`.
        :var dt: integration step :math:`dt` (s).
        """
        gyro = omega.gyro - state.b_gyro + w[:3]
        acc = state.Rot.dot(omega.acc - state.b_acc + w[3:6]) + cls.g
        new_state = cls.STATE(
            Rot=state.Rot.dot(SO3.exp(gyro*dt)),
            v=state.v + acc*dt,
            p=state.p + state.v*dt + 1/2*acc*dt**2,
            # noise is not added on biases
            b_gyro=state.b_gyro,
            b_acc=state.b_acc
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

          \\varphi\\left(\\boldsymbol{\\chi}, 
          \\boldsymbol{\\xi}\\right) = \\left( \\begin{matrix}
            \\mathbf{C} \\exp\\left(\\boldsymbol{\\xi}^{(0:3)}\\right) \\\\
            \\mathbf{v} + \\boldsymbol{\\xi}^{(3:6)} \\\\
            \\mathbf{p} + \\boldsymbol{\\xi}^{(6:9)} \\\\
            \\mathbf{b}_g + \\boldsymbol{\\xi}^{(9:12)} \\\\
            \\mathbf{b}_a + \\boldsymbol{\\xi}^{(12:15)}
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3)
        \\times \\mathbb R^{15}`.

        Its corresponding inverse operation is :meth:`~ukfm.IMUGNSS.phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        new_state = cls.STATE(
            Rot=SO3.exp(xi[:3]).dot(state.Rot),
            v=state.v + xi[3:6],
            p=state.p + xi[6:9],
            b_gyro=state.b_gyro + xi[9:12],
            b_acc=state.b_acc + xi[12:15]
        )
        return new_state

    @classmethod
    def phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}
          \\left(\\boldsymbol{\\chi}\\right) = \\left( \\begin{matrix}
            \\log\\left(\\mathbf{C} \\mathbf{\\hat{C}}^T \\right)\\\\
            \\mathbf{v} - \\mathbf{\\hat{v}} \\\\
            \\mathbf{p} - \\mathbf{\\hat{p}} \\\\
            \\mathbf{b}_g - \\mathbf{\\hat{b}}_g \\\\
            \\mathbf{b}_a - \\mathbf{\\hat{b}}_a
           \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SO(3)
        \\times \\mathbb R^{15}`.

        Its corresponding retraction is :meth:`~ukfm.IMUGNSS.phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        xi = np.hstack([SO3.log(hat_state.Rot.dot(state.Rot.T)),
                        hat_state.v - state.v,
                        hat_state.p - state.p,
                        hat_state.b_gyro - state.b_gyro,
                        hat_state.b_acc - state.b_acc])
        return xi

    @classmethod
    def up_phi(cls, state, xi):
        """Retraction used for updating state and infering Jacobian.

        The retraction :meth:`~ukfm.IMUGNSS.phi` applied on the position state.
        """
        new_state = cls.STATE(
            Rot=state.Rot,
            v=state.v,
            p=xi + state.p,
            b_gyro=state.b_gyro,
            b_acc=state.b_acc
        )
        return new_state

    @classmethod
    def left_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, 
          \\boldsymbol{\\xi}\\right) = \\left( \\begin{matrix}
            \\mathbf{C} \\mathbf{C}_\\mathbf{T} \\\\
            \\mathbf{v} + \\mathbf{C} \\mathbf{r_1} \\\\
            \\mathbf{p} + \\mathbf{C} \\mathbf{r_2} \\\\
            \\mathbf{b}_g + \\boldsymbol{\\xi}^{(9:12)} \\\\
            \\mathbf{b}_a + \\boldsymbol{\\xi}^{(12:15)}
          \\end{matrix} \\right)

        where

        .. math::
            \\mathbf{T} = \\exp\\left(\\boldsymbol{\\xi}^{(0:9)}\\right) 
            = \\begin{bmatrix}
                \\mathbf{C}_\\mathbf{T} & \\mathbf{r_1}  &\\mathbf{r}_2 \\\\
                \\mathbf{0}^T & & \\mathbf{I} 
            \\end{bmatrix}

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)
        \\times \\mathbb{R}^6` with left multiplication.

        Its corresponding inverse operation is 
        :meth:`~ukfm.IMUGNSS.left_phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        dR = SO3.exp(xi[:3])
        J = SO3.left_jacobian(xi[:3])
        new_state = cls.STATE(
            Rot=state.Rot.dot(dR),
            v=state.Rot.dot(J.dot(xi[3:6])) + state.v,
            p=state.Rot.dot(J.dot(xi[6:9])) + state.p,
            b_gyro=state.b_gyro + xi[9:12],
            b_acc=state.b_acc + xi[12:15]
        )
        return new_state

    @classmethod
    def left_phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::

          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}
          \\left(\\boldsymbol{\\chi}\\right) = \\left( \\begin{matrix}
            \\log\\left(
            \\boldsymbol{\chi}^{-1} \\boldsymbol{\\hat{\\chi}} 
            \\right) \\\\
            \\mathbf{b}_g - \\mathbf{\\hat{b}}_g \\\\
            \\mathbf{b}_a - \\mathbf{\\hat{b}}_a
           \end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)`
        with left multiplication.

        Its corresponding retraction is :meth:`~ukfm.IMUGNSS.left_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        dR = state.Rot.T.dot(hat_state.Rot)
        phi = SO3.log(dR)
        J = SO3.inv_left_jacobian(phi)
        dv = state.Rot.T.dot(hat_state.v - state.v)
        dp = state.Rot.T.dot(hat_state.p - state.p)
        xi = np.hstack([phi,
                        J.dot(dv),
                        J.dot(dp),
                        hat_state.b_gyro - state.b_gyro,
                        hat_state.b_acc - state.b_acc])
        return xi

    @classmethod
    def left_H_ana(cls, state):
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        return H

    @classmethod
    def right_phi(cls, state, xi):
        """Retraction.

        .. math::

          \\varphi\\left(\\boldsymbol{\\chi}, \\boldsymbol{\\xi}\\right) 
          = \\left( \\begin{matrix}
            \\mathbf{C}_\\mathbf{T} \\mathbf{C}  \\\\
            \\mathbf{C}_\\mathbf{T}\\mathbf{v} +  \\mathbf{r_1} \\\\
            \\mathbf{C}_\\mathbf{T}\\mathbf{p} +  \\mathbf{r_2} \\\\
            \\mathbf{b}_g + \\boldsymbol{\\xi}^{(9:12)} \\\\
            \\mathbf{b}_a + \\boldsymbol{\\xi}^{(12:15)}
           \\end{matrix} \\right)

        where

        .. math::
            \\mathbf{T} = \\exp\\left(\\boldsymbol{\\xi}^{(0:9)}\\right)
             = \\begin{bmatrix}
                \\mathbf{C}_\\mathbf{T} & \\mathbf{r_1}  &\\mathbf{r}_2 \\\\
                \\mathbf{0}^T & & \\mathbf{I} 
            \\end{bmatrix}

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)
        \\times \\mathbb{R}^6` with right multiplication.

        Its corresponding inverse operation is 
        :meth:`~ukfm.IMUGNSS.right_phi_inv`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var xi: state uncertainty :math:`\\boldsymbol{\\xi}`.
        """
        dR = SO3.exp(xi[:3])
        J = SO3.left_jacobian(xi[:3])
        new_state = cls.STATE(
            Rot=dR.dot(state.Rot),
            v=dR.dot(state.v) + J.dot(xi[3:6]),
            p=dR.dot(state.p) + J.dot(xi[6:9]),
            b_gyro=state.b_gyro + xi[9:12],
            b_acc=state.b_acc + xi[12:15]
        )
        return new_state

    @classmethod
    def right_phi_inv(cls, state, hat_state):
        """Inverse retraction.

        .. math::
        
          \\varphi^{-1}_{\\boldsymbol{\\hat{\\chi}}}
          \\left(\\boldsymbol{\\chi}\\right) = \\left( \\begin{matrix}
            \\log\\left( \\boldsymbol{\\hat{\\chi}}^{-1} 
            \\boldsymbol{\\chi} \\right) \\\\
            \\mathbf{b}_g - \\mathbf{\\hat{b}}_g \\\\
            \\mathbf{b}_a - \\mathbf{\\hat{b}}_a
          \\end{matrix} \\right)

        The state is viewed as a element :math:`\\boldsymbol{\chi} \\in SE_2(3)
        \\times \\mathbb{R}^6` with right multiplication.

        Its corresponding retraction is :meth:`~ukfm.IMUGNSS.right_phi`.

        :var state: state :math:`\\boldsymbol{\\chi}`.
        :var hat_state: noise-free state :math:`\\boldsymbol{\hat{\\chi}}`.
        """
        dR = hat_state.Rot.dot(state.Rot.T)
        phi = SO3.log(dR)
        J = SO3.inv_left_jacobian(phi)
        dv = hat_state.v - dR*state.v
        dp = hat_state.p - dR*state.p
        xi = np.hstack([phi,
                        J.dot(dv),
                        J.dot(dp),
                        hat_state.b_gyro - state.b_gyro,
                        hat_state.b_acc - state.b_acc])
        return xi

    @classmethod
    def right_up_phi(cls, state, xi):
        """Retraction used for updating state and infering Jacobian.

        The retraction :meth:`~ukfm.IMUGNSS.right_phi` applied on the position 
        state.
        """
        chi = SE3.exp(xi)
        new_state = cls.STATE(
            Rot=chi[:3, :3].dot(state.Rot),
            v=state.v,
            p=chi[:3, 3] + state.p,
            b_gyro=state.b_gyro,
            b_acc=state.b_acc
        )
        return new_state

    @classmethod
    def load(cls, gps_freq):
        data_gps = np.genfromtxt(os.path.join(
            cls.data_dir, cls.f_gps), delimiter=',', skip_header=1)
        data_imu = np.genfromtxt(os.path.join(
            cls.data_dir, cls.f_imu), delimiter=' ', skip_header=1)
        data_imu = data_imu[120:]
        t = data_imu[:, 0]
        t0 = t[0]
        t = t - t0
        N = t.shape[0]

        omegaX = data_imu[:, 5]
        omegaY = data_imu[:, 6]
        omegaZ = data_imu[:, 7]
        accelX = data_imu[:, 2]
        accelY = data_imu[:, 3]
        accelZ = data_imu[:, 4]

        omegas = []
        for n in range(N):
            omegas.append(cls.INPUT(
                gyro=np.array([omegaX[n], omegaY[n], omegaZ[n]]),
                acc=np.array([accelX[n], accelY[n], accelZ[n]])))
        t_gps = data_gps[:, 0] - t0
        N_gps = t_gps.shape[0]

        # vector to know where GPS measurement happen
        one_hot_ys = np.zeros(N)
        k = 1
        ys = np.zeros((N_gps, 3))
        for n in range(1, N):
            if t_gps[k] <= t[n]:
                ys[k] = data_gps[k, 1:]
                one_hot_ys[n] = 1
                k += 1
            if k >= N_gps:
                break
        return omegas, ys, one_hot_ys, t

    @classmethod
    def plot_results(cls, hat_states, ys):
        N = len(hat_states)
        hat_Rots, hat_vs, hat_ps, hat_b_gyros, hat_b_accs = cls.get_states(
            hat_states)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set(xlabel='$x$ (m)', ylabel='$y$ (m)', title="Robot position")
        ax.scatter(ys[:, 0], ys[:, 1], c='red')
        plt.plot(hat_ps[:, 0], hat_ps[:, 1], c='blue')
        ax.legend(['UKF', r'GPS measurements'])
        ax.axis('equal')

    @classmethod
    def get_states(cls, states):
        N = len(states)
        Rots = np.zeros((N, 3, 3))
        vs = np.zeros((N, 3))
        ps = np.zeros((N, 3))
        b_gyros = np.zeros((N, 3))
        b_accs = np.zeros((N, 3))
        for n in range(N):
            Rots[n] = states[n].Rot
            vs[n] = states[n].v
            ps[n] = states[n].p
            b_gyros[n] = states[n].b_gyro
            b_accs[n] = states[n].b_acc
        return Rots, vs, ps, b_gyros, b_accs
