import numpy as np
from ukfm.geometry.so3 import SO3


class SE3:
    """Homogeneous transformation matrix in :math:`SE(3)`.

    .. math::

        SE(3) &= \\left\\{ \\mathbf{T}=
                \\begin{bmatrix}
                    \\mathbf{C} & \\mathbf{r} \\\\
                        \\mathbf{0}^T & 1
                \\end{bmatrix} \\in \\mathbb{R}^{4 \\times 4} ~\\middle|~ 
                \\mathbf{C} \\in SO(3), \\mathbf{r} \\in \\mathbb{R}^3 
                \\right\\} \\\\
        \\mathfrak{se}(3) &= \\left\\{ \\boldsymbol{\\Xi} =
        \\boldsymbol{\\xi}^\\wedge \\in \\mathbb{R}^{4 \\times 4} ~\\middle|~
         \\boldsymbol{\\xi}=
            \\begin{bmatrix}
                \\boldsymbol{\\phi} \\\\ \\boldsymbol{\\rho}
            \\end{bmatrix} \\in \\mathbb{R}^6, \\boldsymbol{\\rho} \\in 
            \\mathbb{R}^3, \\boldsymbol{\\phi} \in \\mathbb{R}^3 \\right\\}
    """

    @classmethod
    def exp(cls, xi):
        """Exponential map for :math:`SE(3)`, which computes a transformation 
        from a tangent vector:

        .. math::

            \\mathbf{T}(\\boldsymbol{\\xi}) =
            \\exp(\\boldsymbol{\\xi}^\\wedge) =
            \\begin{bmatrix}
                \\exp(\\boldsymbol{\\phi}^\\wedge) & \\mathbf{J} 
                \\boldsymbol{\\rho}  \\\\
                \\mathbf{0} ^ T & 1
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SE3.log`.
        """
        chi = np.eye(4)
        chi[:3, :3] = SO3.exp(xi[:3])
        chi[:3, 3] = SO3.left_jacobian(xi[:3]).dot(xi[3:])
        return chi

    @classmethod
    def inv(cls, chi):
        """Inverse map for :math:`SE(3)`.

        .. math::

            \\mathbf{T}^{-1} =
            \\begin{bmatrix}
                \\mathbf{C}^T  & -\\mathbf{C}^T \\boldsymbol{\\rho} 
                    \\\\
                \\mathbf{0} ^ T & 1
            \\end{bmatrix}

        """
        chi_inv = np.eye(4)
        chi_inv[:3, :3] = chi[:3, :3].T
        chi_inv[:3, 3] = -chi[:3, :3].T.dot(chi[:3, 3])
        return chi_inv

    @classmethod
    def log(cls, chi):
        """Logarithmic map for :math:`SE(3)`, which computes a tangent vector 
        from a transformation:

        .. math::
        
            \\boldsymbol{\\xi}(\\mathbf{T}) =
            \\ln(\\mathbf{T})^\\vee =
            \\begin{bmatrix}
                \\mathbf{J} ^ {-1} \\mathbf{r} \\\\
                \\ln(\\boldsymbol{C}) ^\\vee
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SE3.exp`.
        """
        phi = SO3.log(chi[:3, :3])
        xi = np.hstack([phi, SO3.inv_left_jacobian(phi).dot(chi[:3, 3])])
        return xi
