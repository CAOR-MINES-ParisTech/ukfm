import numpy as np

class EKF:
    """The Extended Kalman Filter on (parallelizable) manifolds. 
    
    This is the counterpart of the UKF on (parallelizable) manifolds, where 
    functions for computing Jacobian need to be provided.

    :arg model: model that contains propagation function :math:`f` and 
        observation function :math:`h`.
    :arg phi: retraction :math:`\\boldsymbol{\\varphi}`.
    :arg FG_ana: function for computing Jacobian :math:`\\mathbf{F}` and 
        :math:`\\mathbf{G}` during propagation.
    :arg H_ana: function for computing Jacobian :math:`\\mathbf{H}` during 
        update.
    :ivar Q: propagation noise covariance matrix (static) :math:`\\mathbf{Q}`.
    :ivar R: observation noise covariance matrix (static) :math:`\\mathbf{R}`.
    :ivar state: state :math:`\\boldsymbol{\\hat{\\chi}}_n`, initialized at 
        ``state0``.
    :ivar P: state uncertainty covariance :math:`\\mathbf{P}_n`, initialized at 
        ``P0``.
    """

    def __init__(self, model, FG_ana, H_ana, phi, Q, R, state0, P0):
        self.model = model
        self.Q = Q
        self.R = R
        self.FG_ana = FG_ana
        self.H_ana = H_ana
        self.phi = phi
        self.state = state0
        self.P = P0

        self.w0 = np.zeros(Q.shape[0])
        self.Id_d = np.eye(self.P.shape[0]) 

    def propagation(self, omega, dt):
        """EKF propagation step.

        .. math::

          \\boldsymbol{\\hat{\\chi}}_{n} &\\leftarrow
          \\boldsymbol{\\hat{\\chi}}_{n+1} =
          f\\left(\\boldsymbol{\\hat{\\chi}}_{n}, 
          \\boldsymbol{\\omega}_{n}, \\mathbf{0}\\right) \\\\
          \\mathbf{P}_{n} &\\leftarrow \\mathbf{P}_{n+1} = \\mathbf{F} 
          \\mathbf{P}_{n} \\mathbf{F}^T 
          + \\mathbf{G} \\mathbf{Q} \\mathbf{G}^T  \\\\

        Mean state and covariance are propagated.

        :var omega: input :math:`\\boldsymbol{\\omega}`.
        :var dt: integration step :math:`dt` (s).
        """
        # propagate covariance
        F, G = self.FG_ana(self.state, omega, dt)
        self.P = F.dot(self.P).dot(F.T) + G.dot(self.Q).dot(G.T)
        # propagate mean state
        self.state = self.model.f(self.state, omega, self.w0, dt)

    def update(self, y):
        """EKF update step.

        .. math::
        
          \\boldsymbol{\\hat{\\chi}}_{n} &\\leftarrow 
          \\boldsymbol{\\hat{\\chi}}_{n}^{+} \\\\
          \\mathbf{P}_{n} &\\leftarrow \\mathbf{P}_{n}^{+} \\\\

        :var y: 1D array (vector) measurement :math:`\\mathbf{y}_n`.
        """
        # Observability matrix
        H = self.H_ana(self.state)

        # measurement uncertainty matrix
        S = H.dot(self.P).dot(H.T) + self.R

        # gain matrix
        K = np.linalg.solve(S, self.P.dot(H.T).T).T

        # innovation
        xi = K.dot(y-self.model.h(self.state))

        # update state
        self.state = self.phi(self.state, xi)

        # update covariance
        P = (np.eye(self.P.shape[0])-K.dot(H)).dot(self.P)
        # avoid non-symmetric matrix
        self.P = (P+P.T)/2
