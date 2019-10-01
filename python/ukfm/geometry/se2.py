import numpy as np
from ukfm.geometry.so2 import SO2


class SE2:
    """Homogeneous transformation matrix in :math:`SE(2)`

    .. math::
    
        SE(2) &= \\left\\{ \\mathbf{T}=
                \\begin{bmatrix}
                    \\mathbf{C} & \\mathbf{r} \\\\
                    \\mathbf{0}^T & 1
                \\end{bmatrix} \\in \\mathbb{R}^{3 \\times 3} ~\\middle|~ 
                \\mathbf{C} \\in SO(2), \\mathbf{r} \\in \\mathbb{R}^2 
                \\right\\} \\\\
        \\mathfrak{se}(2) &= \\left\\{ \\boldsymbol{\\Xi} =
        \\boldsymbol{\\xi}^\\wedge \\in \\mathbb{R}^{3 \\times 3} ~\\middle|
        ~
            \\boldsymbol{\\xi}=
            \\begin{bmatrix}
                \\phi \\\\  \\boldsymbol{\\rho}
            \\end{bmatrix} \\in \\mathbb{R}^3, \\boldsymbol{\\rho} \\in 
            \\mathbb{R}^2, \\phi \in \\mathbb{R} \\right\\}

    """

    @classmethod
    def Ad(cls, chi):
        """Adjoint matrix of the transformation.

        .. math::
        
            \\text{Ad}(\\mathbf{T}) =
            \\begin{bmatrix}
                \\mathbf{C} & 1^\\wedge \\mathbf{r} \\\\
                \\mathbf{0}^T & 1
            \\end{bmatrix}
            \\in \\mathbb{R}^{3 \\times 3}

        """
        Rot = chi[:2, :2]
        Jp = np.array([chi[1, 2], -chi[0, 2]])
        Ad = np.vstack([np.hstack([Jp, Rot]),
                        [1, 0, 0]])
        return Ad

    @classmethod
    def exp(cls, xi):
        """Exponential map for :math:`SE(2)`, which computes a transformation
        from a tangent vector:

        .. math::
            \\mathbf{T}(\\boldsymbol{\\xi}) =
            \\exp(\\boldsymbol{\\xi}^\\wedge) =
            \\begin{bmatrix}
                \\exp(\\phi ^\\wedge) & \\mathbf{J} \\boldsymbol{\\rho}  \\\\
                \\mathbf{0} ^ T & 1
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SE2.log`.
        """
        chi = np.eye(3)
        chi[:2, :2] = SO2.exp(xi[0])
        chi[:2, 2] = SO2.left_jacobian(xi[0]).dot(xi[1:3])
        return chi

    @classmethod
    def inv(cls, chi):
        """Inverse map for :math:`SE(2)`.

        .. math::
        
            \\mathbf{T}^{-1} =
            \\begin{bmatrix}
                \\mathbf{C}^T  & -\\mathbf{C}^T \\boldsymbol{\\rho}
                    \\\\
                \\mathbf{0} ^ T & 1
            \\end{bmatrix}

        """
        chi_inv = np.eye(3)
        chi_inv[:2, :2] = chi[:2, :2].T
        chi_inv[:2, 2] = -chi[:2, :2].T.dot(chi[:2, 2])
        return chi_inv

    @classmethod
    def log(cls, chi):
        """Logarithmic map for :math:`SE(2)`, which computes a tangent vector 
        from a transformation:

        .. math::
        
            \\boldsymbol{\\xi}(\\mathbf{T}) =
            \\ln(\\mathbf{T})^\\vee =
            \\begin{bmatrix}
            \\ln(\\boldsymbol{C}) ^\\vee \\\\
                \\mathbf{J} ^ {-1} \\mathbf{r}
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SE2.log`.
        """
        phi = SO2.log(chi[:2, :2])
        xi = np.hstack([phi, SO2.inv_left_jacobian(phi).dot(chi[:2, 2])])
        return xi
