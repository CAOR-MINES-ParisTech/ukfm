import numpy as np
from scipy.linalg import block_diag

class UKF:
    """The Unscented Kalman Filter on (parallelizable) Manifolds.

    This filter is the implementation described in :cite:`brossardCode2019` . It
    is well adapted to relatively small systems and for understanding the
    methodology of **UKF-M**, otherwise see :meth:`~ukfm.JUKF`. Noise covariance
    parameters are assumed static for convenience, i.e. :math:`\\mathbf{Q}_n =
    \\mathbf{Q}`, and :math:`\\mathbf{R}_n = \\mathbf{R}`.

    :arg f: propagation function :math:`f`.
    :arg h: observation function :math:`h`.
    :arg phi: retraction :math:`\\boldsymbol{\\varphi}`.
    :arg phi_inv: inverse retraction :math:`\\boldsymbol{\\varphi}^{-1}`.
    :ivar Q: propagation noise covariance matrix (static) :math:`\\mathbf{Q}`.
    :ivar R: observation noise covariance matrix (static) :math:`\\mathbf{R}`.
    :arg alpha: sigma point parameters. Must be 1D array with 3 values.
    :ivar state: state :math:`\\boldsymbol{\\hat{\\chi}}_n`, initialized at 
        ``state0``.
    :ivar P: state uncertainty covariance :math:`\\mathbf{P}_n`, initialized at
        ``P0``.
    """

    TOL = 1e-9 # tolerance parameter (avoid numerical issue)

    def __init__(self, f, h, phi, phi_inv, Q, R, alpha, state0, P0):
        self.f = f
        self.h = h
        self.phi = phi
        self.phi_inv = phi_inv
        self.Q = Q
        self.R = R
        self.state = state0
        self.P = P0
        
        # Cholesky decomposition of Q
        self.cholQ = np.linalg.cholesky(Q).T
        
        # variable dimensions
        self.d = P0.shape[0]
        self.q = Q.shape[0]
        self.l = R.shape[0]

        self.Id_d = np.eye(self.d)
        
        # sigma point weights
        self.weights = self.WEIGHTS(P0.shape[0], Q.shape[0], alpha)

    class WEIGHTS:
        """Sigma point weights.

        Weights are computed as:

        .. math::

          \\lambda &= (\\alpha^2 - 1) \\mathrm{dim}, \\\\
          w_j &= 1/(2(\\mathrm{dim} + \\lambda)), \\\\
          w_m &= \\lambda/(\\lambda + \\mathrm{dim}), \\\\
          w_0 &= \\lambda/(\\lambda + \\mathrm{dim}) + 3 - \\alpha^2,

        where :math:`\\alpha` is a parameter set between :math:`10^{-3}` and
        :math:`1`, and :math:`\\mathrm{dim}` is the dimension of the
        sigma-points (:math:`d` or :math:`q`).
        
        This variable contains sigma point weights for propagation (w.r.t. state
        uncertainty and noise) and for update.
        """

        def __init__(self, d, q, alpha):
            # propagation w.r.t. state
            self.d = self.W(d, alpha[0])
            # propagation w.r.t. noise
            self.q = self.W(q, alpha[1])
            # update w.r.t. state
            self.u = self.W(d, alpha[2])

        class W:
            def __init__(self, l, alpha):
                m = (alpha**2 - 1) * l
                self.sqrt_d_lambda = np.sqrt(l + m)
                self.wj = 1/(2*(l + m))
                self.wm = m/(m + l)
                self.w0 = m/(m + l) + 3 - alpha**2

        
    def propagation(self, omega, dt):
        """UKF propagation step.

        .. math::
        
          \\boldsymbol{\\hat{\\chi}}_{n} &\\leftarrow 
          \\boldsymbol{\\hat{\\chi}}_{n+1} = 
          f\\left(\\boldsymbol{\\hat{\\chi}}_{n}, \\boldsymbol{\\omega}_{n}, 
          \\mathbf{0}\\right) \\\\
          \\mathbf{P}_{n} &\\leftarrow \\mathbf{P}_{n+1} \\\\

        Mean state and covariance are propagated.

        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var dt: integration step :math:`dt` (s).
        """

        P = self.P + self.TOL*self.Id_d

        # update mean
        w = np.zeros(self.q)
        new_state = self.f(self.state, omega, w, dt)

        # compute covariance w.r.t. state uncertainty
        w_d = self.weights.d

        # set sigma points
        xis = w_d.sqrt_d_lambda * np.linalg.cholesky(P).T
        new_xis = np.zeros((2*self.d, self.d))

        # retract sigma points onto manifold
        for j in range(self.d):
            s_j_p = self.phi(self.state, xis[j])
            s_j_m = self.phi(self.state, -xis[j])
            new_s_j_p = self.f(s_j_p, omega, w, dt)
            new_s_j_m = self.f(s_j_m, omega, w, dt)
            new_xis[j] = self.phi_inv(new_state, new_s_j_p)
            new_xis[self.d + j] = self.phi_inv(new_state, new_s_j_m)
        
        # compute covariance
        new_xi = w_d.wj * np.sum(new_xis, 0)
        new_xis = new_xis - new_xi
        new_P = w_d.wj * new_xis.T.dot(new_xis) + \
            w_d.w0*np.outer(new_xi, new_xi)

        # compute covariance w.r.t. noise
        w_q = self.weights.q
        new_xis = np.zeros((2*self.q, self.d))

        # retract sigma points onto manifold
        for j in range(self.q):
            w_p = w_q.sqrt_d_lambda * self.cholQ[j]
            w_m = -w_q.sqrt_d_lambda * self.cholQ[j]
            new_s_j_p = self.f(self.state, omega, w_p, dt)
            new_s_j_m = self.f(self.state, omega, w_m, dt)
            new_xis[j] = self.phi_inv(new_state, new_s_j_p)
            new_xis[self.q + j] = self.phi_inv(new_state, new_s_j_m)

        # compute covariance
        new_xi = w_q.wj * np.sum(new_xis, 0)
        new_xis = new_xis - new_xi
        Q = w_q.wj * new_xis.T.dot(new_xis) + w_q.w0*np.outer(new_xi, new_xi)

        # sum covariances
        self.P = new_P + Q
        self.state = new_state

    def update(self, y):
        """UKF update step.

        .. math::
        
          \\boldsymbol{\\hat{\\chi}}_{n} &\\leftarrow \\boldsymbol{\\hat{\\chi}}
          _{n}^{+} \\\\
          \\mathbf{P}_{n} &\\leftarrow \\mathbf{P}_{n}^{+} \\\\

        :var y: 1D array (vector) measurement :math:`\\mathbf{y}_n`.
        """
        P = self.P + self.TOL*self.Id_d

        # set sigma points
        w_d = self.weights.d
        xis = w_d.sqrt_d_lambda * np.linalg.cholesky(P).T

        # compute measurement sigma_points
        ys = np.zeros((2*self.d, self.l))
        hat_y = self.h(self.state)
        for j in range(self.d):
            s_j_p = self.phi(self.state, xis[j])
            s_j_m = self.phi(self.state, -xis[j])
            ys[j] = self.h(s_j_p)
            ys[self.d + j] = self.h(s_j_m)

        # measurement mean
        y_bar = w_d.wm * hat_y + w_d.wj * np.sum(ys, 0)

        # prune mean before computing covariance
        ys = ys - y_bar
        hat_y = hat_y - y_bar

        # compute covariance and cross covariance matrices
        P_yy = w_d.w0*np.outer(hat_y, hat_y) + w_d.wj*ys.T.dot(ys) + self.R
        P_xiy = w_d.wj*np.hstack([xis.T, -xis.T]).dot(ys)

        # Kalman gain
        K = np.linalg.solve(P_yy, P_xiy.T).T
        # update state
        xi_plus = K.dot(y - y_bar)
        self.state = self.phi(self.state, xi_plus)

        # update covariance
        self.P = P - K.dot(P_yy).dot(K.T)
        # avoid non-symmetric matrix
        self.P = (self.P + self.P.T)/2


class JUKF:
    """The Unscented Kalman Filter on (parallelizable) Manifolds, that infers 
    Jacobian.

    This filter is an alternative implementation to the method described in
    :cite:`brossardCode2019`, with exactly the same results. It spares
    computational time for systems when only a part of the state is involved in
    a propagation or update step. It can also be used for state augmentation.
    Only noise covariance parameter for propagation is assumed static for
    convenience, i.e. :math:`\\mathbf{Q}_n = \\mathbf{Q}`.

    :arg f: propagation function :math:`f`.
    :arg h: observation function :math:`h`.
    :arg phi: retraction :math:`\\boldsymbol{\\varphi}`.
    :ivar Q: propagation noise covariance matrix (static) :math:`\\mathbf{Q}`.
    :arg alpha: sigma point parameters. Must be 1D array with 5 values.
    :ivar state: state :math:`\\boldsymbol{\\hat{\\chi}}_n`, initialized at 
        ``state0``.
    :ivar P: state uncertainty covariance :math:`\\mathbf{P}_n`, initialized at
        ``P0``.
    :arg red_phi: reduced retraction for propagation.
    :arg red_phi_inv: reduced inverse retraction for propagation.
    :arg red_idxs: indices corresponding to the reduced uncertainty.
    :arg up_phi: retraction for update.
    :arg up_idxs: indices corresponding to the state uncertainty for update.
    :arg aug_z: augmentation function :math:`z`. (optional)
    :arg aug_phi: retraction for augmenting state. (optional)
    :arg aug_phi_inv: inverse retraction for augmenting state. (optional)
    :arg aug_idxs: indices corresponding to the state uncertainty for state
        augmentation. (optional)
    :arg aug_q: state uncertainty dimension for augmenting state. (optional)
    """

    def __init__(self, f, h, phi, Q, alpha,  state0, P0, red_phi, 
        red_phi_inv, red_idxs, up_phi, up_idxs,
        aug_z=None, aug_phi=None, aug_phi_inv=None, aug_idxs=np.array([0]), 
        aug_q=1):
        self.state = state0
        self.P = P0
        self.f = f
        self.h = h
        self.Q = Q
        self.cholQ = np.linalg.cholesky(Q).T
        self.phi = phi

        self.new_state = self.state
        self.F = np.eye(self.P.shape[0])
        self.G = np.zeros((self.P.shape[0], self.Q.shape[0]))
        self.H = np.zeros((0, self.P.shape[0]))
        self.r = np.zeros(0)
        self.R = np.zeros((0, 0))

        self.TOL = 1e-9
        self.red_idxs = red_idxs
        self.red_d = red_idxs.shape[0]
        self.up_idxs = up_idxs
        self.up_d = up_idxs.shape[0]
        self.q = Q.shape[0]

        # reducing state during propagation
        self.red_phi = red_phi
        self.red_phi_inv = red_phi_inv
        self.red_idxs = red_idxs

        # reducing state during update
        self.up_idxs = up_idxs
        self.up_phi = up_phi

        # for augmenting state
        self.aug_z = aug_z
        self.aug_d = aug_idxs.shape[0]
        self.aug_idxs = aug_idxs
        self.aug_phi = aug_phi
        self.aug_phi_inv = aug_phi_inv
        self.aug_q = aug_q

        self.weights = self.WEIGHTS(self.red_d, Q.shape[0], self.up_d, 
            self.aug_d, self.aug_q, alpha)

    class WEIGHTS:
        """Sigma point weights.

        Weights are computed as:

        .. math::

          \\lambda &= (\\alpha^2 - 1) \\mathrm{dim}, \\\\
          w_j &= 1/(2(\\mathrm{dim} + \\lambda)), \\\\
          w_m &= \\lambda/(\\lambda + \\mathrm{dim}), \\\\
          w_0 &= \\lambda/(\\lambda + \\mathrm{dim}) + 3 - \\alpha^2,

        where :math:`\\alpha` is a parameter set between :math:`10^{-3}` and 
        :math:`1`, and :math:`\\mathrm{dim}` the dimension of the sigma-points.
        
        This variable contains sigma point weights for propagation (w.r.t. state
        uncertainty and noise), update and state augmentation.
        """
        def __init__(self, red_d, q, up_d, aug_d, aug_q, alpha):
            self.red_d = self.W(red_d, alpha[0])
            self.q = self.W(q, alpha[1])
            self.up_d = self.W(up_d, alpha[2])
            self.aug_d = self.W(aug_d, alpha[3])
            self.aug_q = self.W(aug_q, alpha[4])

        class W:
            def __init__(self, l, alpha):
                m = (alpha**2 - 1) * l
                self.sqrt_d_lambda = np.sqrt(l + m)
                self.wj = 1/(2*(l + m))
                self.wm = m/(m + l)
                self.w0 = m/(m + l) + 3 - alpha**2

    def F_num(self, omega, dt):
        """Numerical Jacobian computation of :math:`\mathbf{F}`.

        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var dt: integration step :math:`dt` (s).
        """
        P = self.P[np.ix_(self.red_idxs, self.red_idxs)] 
        self.F = np.eye(self.P.shape[0])
        # variable sizes
        d = P.shape[0]
        P = P + self.TOL*np.eye(d)
        w = np.zeros(self.q)

        w_d = self.weights.red_d
        
        # set sigma points
        xis = w_d.sqrt_d_lambda * np.linalg.cholesky(P).T
        new_xis = np.zeros((2*d, d))

        # retract sigma points onto manifold
        for j in range(d):
            s_j_p = self.red_phi(self.state, xis[j])
            s_j_m = self.red_phi(self.state, -xis[j])
            new_s_j_p = self.f(s_j_p, omega, w, dt)
            new_s_j_m = self.f(s_j_m, omega, w, dt)
            new_xis[j] = self.red_phi_inv(self.new_state, new_s_j_p)
            new_xis[d + j] = self.red_phi_inv(self.new_state, new_s_j_m)

        # compute covariance
        new_xi = w_d.wj * np.sum(new_xis, 0)
        new_xis = new_xis - new_xi

        Xi = w_d.wj * new_xis.T.dot(np.vstack([xis, -xis]))
        self.F[np.ix_(self.red_idxs, self.red_idxs)] = \
             np.linalg.solve(P, Xi.T).T  # Xi*P_red^{-1}

    def propagation(self, omega, dt):
        """UKF propagation step.

        .. math::

          \\boldsymbol{\\hat{\\chi}}_{n} &\\leftarrow \\boldsymbol{\\hat{\\chi}}
          _{n+1} = f\\left(\\boldsymbol{\\hat{\\chi}}_{n}, 
          \\boldsymbol{\\omega}_{n}, \\mathbf{0}\\right) \\\\
          \\mathbf{P}_{n} &\\leftarrow \\mathbf{P}_{n+1} = \\mathbf{F} 
          \\mathbf{P}_{n} \\mathbf{F}^T + \\mathbf{G} \\mathbf{Q} 
          \\mathbf{G}^T  \\\\

        Mean state and covariance are propagated. Covariance is propagated as 
        an EKF, where Jacobian :math:`\\mathbf{F}` and :math:`\\mathbf{G}` are 
        *numerically* inferred.

        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var dt: integration step :math:`dt` (s).
        """

        self.state_propagation(omega, dt)
        self.F_num(omega, dt)
        self.G_num(omega, dt)
        self.cov_propagation()

    def state_propagation(self, omega, dt):
        """Propagate mean state.
        
        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var dt: integration step :math:`dt` (s).
        """
        w = np.zeros(self.q)
        self.new_state = self.f(self.state, omega, w, dt)

    def G_num(self, omega, dt):
        """Numerical Jacobian computation of :math:`\mathbf{G}`.

        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var dt: integration step :math:`dt` (s).
        """
        w_q = self.weights.q
        new_xis = np.zeros((2*self.q, self.red_d))

        # retract sigma points onto manifold
        for j in range(self.q):
            w_p = w_q.sqrt_d_lambda * self.cholQ[j]
            w_m = -w_q.sqrt_d_lambda * self.cholQ[j]
            new_s_j_p = self.f(self.state, omega, w_p, dt)
            new_s_j_m = self.f(self.state, omega, w_m, dt)
            new_xis[j] = self.red_phi_inv(self.new_state, new_s_j_p)
            new_xis[self.q + j] = self.red_phi_inv(self.new_state, new_s_j_m)

        # compute covariance
        new_xi = w_q.wj * np.sum(new_xis, 0)
        new_xis = new_xis - new_xi
        Xi = w_q.wj * new_xis.T.dot(np.vstack([self.cholQ, -self.cholQ])) \
            *w_q.sqrt_d_lambda
        self.G = np.zeros((self.P.shape[0], self.q))
        self.G[self.red_idxs] = np.linalg.solve(self.Q, Xi.T).T  # Xi*P_red^{-1}

    def cov_propagation(self):
        """Covariance propagation.

        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var dt: integration step :math:`dt` (s).
        """
        P = self.F.dot(self.P).dot(self.F.T) + self.G.dot(self.Q).dot(self.G.T)
        self.P = (P+P.T)/2
        self.state = self.new_state

    def update(self, y, R):
        """State update, where Jacobian is computed.

        :var y: 1D array (vector) measurement :math:`\\mathbf{y}_n`.
        :var R:  measurement covariance :math:`\\mathbf{R}_n`.
        """
        self.H_num(y, self.up_idxs, R)
        self.state_update()

    def H_num(self, y, idxs, R):
        """Numerical Jacobian computation of :math:`\mathbf{H}`.

        :var y: 1D array (vector) measurement :math:`\\mathbf{y}_n`.
        :var idxs: indices corresponding to the state uncertainty for update.
        :var R:  measurement covariance :math:`\\mathbf{R}_n`.
        """

        P = self.P[np.ix_(idxs, idxs)]
        # set variable size
        d = P.shape[0]
        l = y.shape[0]

        P = P + self.TOL*np.eye(d)

        # set sigma points
        w_u = self.weights.up_d
        xis = w_u.sqrt_d_lambda * np.linalg.cholesky(P).T

        # compute measurement sigma_points
        y_mat = np.zeros((2*d, l))
        hat_y = self.h(self.state)
        for j in range(d):
            s_j_p = self.up_phi(self.state, xis[j])
            s_j_m = self.up_phi(self.state, -xis[j])
            y_mat[j] = self.h(s_j_p)
            y_mat[d + j] = self.h(s_j_m)

        # measurement mean
        y_bar = w_u.wm * hat_y + w_u.wj * np.sum(y_mat, 0)
        # prune mean before computing covariance
        y_mat = y_mat - y_bar

        Y = w_u.wj*y_mat.T.dot(np.vstack([xis, -xis]))
        H_idx = np.linalg.solve(P, Y.T).T # Y*P_red^{-1}

        H = np.zeros((y.shape[0], self.P.shape[0]))
        H[:, idxs] = H_idx

        # compute residual
        r = y - y_bar

        self.H = np.vstack((self.H, H))
        self.r = np.hstack((self.r, r))
        self.R = block_diag(self.R, R)

    def state_update(self):
        """State update, once Jacobian is computed.
        """

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

    def aug(self, y, aug_idxs, R):
        """State augmentation.

        :var y: 1D array (vector) measurement :math:`\\mathbf{y}_n`.
        :var aug_idxs: indices corresponding to the state augmentation
            uncertainty.
        :var R:  measurement covariance :math:`\\mathbf{R}_n`.
        """

        P = self.P[np.ix_(aug_idxs, aug_idxs)] + self.TOL*np.eye(self.aug_d)

        # augment state mean
        aug_state = self.aug_z(self.state, y)

        # compute Jacobian and covariance from state
        # set sigma points w.r.t. state
        w_d = self.weights.aug_d
        xis = w_d.sqrt_d_lambda * np.linalg.cholesky(P).T

        # compute measurement sigma_points
        zs = np.zeros((2*self.aug_d, self.aug_q))
        for j in range(self.aug_d):
            s_j_p = self.aug_phi(self.state, xis[j])
            s_j_m = self.aug_phi(self.state, -xis[j])
            z_j_p = self.aug_z(s_j_p, y)
            z_j_m = self.aug_z(s_j_m, y)
            zs[j] = self.aug_phi_inv(aug_state, z_j_p)
            zs[self.aug_d + j] = self.aug_phi_inv(aug_state, z_j_m)
            
        # measurement mean
        z_bar = w_d.wj * np.sum(zs, 0)

        # prune mean before computing covariance
        zs = zs - z_bar
        P_ss = w_d.wj * zs.T.dot(zs) + w_d.w0*np.outer(z_bar, z_bar)

        Xi = w_d.wj * zs.T.dot(np.vstack([xis, -xis]))
        H = np.zeros((self.aug_q, self.P.shape[0]))
        H[:, aug_idxs] = np.linalg.solve(P, Xi.T).T  # Xi*P^{-1}

        # compute covariance from measurement
        # set sigma points w.r.t. noise
        w_q = self.weights.aug_q
        y_mat = w_q.sqrt_d_lambda * np.linalg.cholesky(R).T

        # compute measurement sigma_points
        zs = np.zeros((2*R.shape[0], self.aug_q))
        for j in range(R.shape[0]):
            y_j_p = y + y_mat[j]
            y_j_m = y - y_mat[j]
            z_j_p = self.aug_z(aug_state, y_j_p)
            z_j_m = self.aug_z(aug_state, y_j_m)
            zs[j] = self.aug_phi_inv(aug_state, z_j_p)
            zs[self.aug_q + j] = self.aug_phi_inv(aug_state, z_j_m)

        # measurement mean
        z_bar = w_q.wj * np.sum(zs, 0)

        # prune mean before computing covariance
        zs = zs - z_bar
        P_zz = w_q.wj * zs.T.dot(zs) + w_q.w0*np.outer(z_bar, z_bar)

        # compute augmented covariance
        P_sz = H.dot(self.P)
        P2 = np.zeros((self.P.shape[0] + 2, self.P.shape[0] + 2))
        P2[:self.P.shape[0], :self.P.shape[0]] = self.P
        P2[:self.P.shape[0], self.P.shape[0]:] = P_sz.T
        P2[self.P.shape[0]:, :self.P.shape[0]] = P_sz
        P2[self.P.shape[0]:, self.P.shape[0]:] = P_ss + P_zz
        self.P = P2

        self.state = aug_state

        # init for next update
        self.H = np.zeros((0, self.P.shape[0]))
        self.r = np.zeros(0)
        self.R = np.zeros((0, 0))
