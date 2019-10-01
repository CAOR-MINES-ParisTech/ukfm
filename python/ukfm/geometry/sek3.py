import numpy as np
from ukfm.geometry.so3 import SO3


class SEK3:
    """Homogeneous transformation matrix in :math:`SE_k(3)`.

    .. math::

        SE_k(3) &= \\left\\{ \\mathbf{T}=
                \\begin{bmatrix}
                    \\mathbf{C} & \\mathbf{r_1} & \\cdots &\\mathbf{r}_k \\\\
                    \\mathbf{0}^T & & \\mathbf{I} &
                \\end{bmatrix} \\in \\mathbb{R}^{(3+k) \\times (3+k)} 
                ~\\middle|~ \\mathbf{C} \\in SO(3), \\mathbf{r}_1 
                \\in \\mathbb{R}^3, \cdots, \\mathbf{r}_k \\in 
                \\mathbb{R}^3 \\right\\} \\\\
        \\mathfrak{se}_k(3) &= \\left\\{ \\boldsymbol{\\Xi} =
        \\boldsymbol{\\xi}^\\wedge \\in \\mathbb{R}^{(3+k) 
        \\times (3+k)} ~\\middle|~
         \\boldsymbol{\\xi}=
            \\begin{bmatrix}
                \\boldsymbol{\\phi} \\\\ \\boldsymbol{\\rho}_1  \\\\ 
                \\vdots  \\\\ \\boldsymbol{\\rho}_k
            \\end{bmatrix} \\in \\mathbb{R}^{3+3k}, \\boldsymbol{\\phi} 
            \in \\mathbb{R}^3, \\boldsymbol{\\rho}_1 \\in \\mathbb{R}^3, 
            \\cdots, \\boldsymbol{\\rho}_k \\in \\mathbb{R}^3 \\right\\}

    """
    @classmethod
    def exp(cls, xi):
        """Exponential map for :math:`SE_k(3)`, which computes a transformation 
        from a tangent vector:

        .. math::

            \\mathbf{T}(\\boldsymbol{\\xi}) =
            \\exp(\\boldsymbol{\\xi}^\\wedge) =
            \\begin{bmatrix}
                \\exp(\\boldsymbol{\\phi}^\\wedge) & \\mathbf{J} 
                \\boldsymbol{\\rho}_1 & \\cdots  & \\mathbf{J} 
                \\boldsymbol{\\rho}_k  \\\\
                \\mathbf{0} ^ T & & \\mathbf{I} &
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SEK3.log`.
        """
        k = int(xi.shape[0]/3 - 1)
        Xi = np.reshape(xi[3:], (3, k), 'F')
        chi = np.eye(3+k)
        chi[:3, :3] = SO3.exp(xi[:3])
        chi[:3, 3:] = SO3.left_jacobian(xi[:3]).dot(Xi)
        return chi

    @classmethod
    def inv(cls, chi):
        """Inverse map for :math:`SE_k(3)`.

        .. math::

            \\mathbf{T}^{-1} =
            \\begin{bmatrix}
                \\mathbf{C}^T  & -\\mathbf{C}^T \\boldsymbol{\\rho}_1  &
                    \\cdots & & -\\mathbf{C}^T \\boldsymbol{\\rho}_k \\\\
                \\mathbf{0} ^ T & & \\mathbf{I} &
            \\end{bmatrix}

        """
        k = chi.shape[0] - 3
        chi_inv = np.eye(3+k)
        chi_inv[:3, :3] = chi[:3, :3].T
        chi_inv[:3, 3:] = -chi_inv[:3, :3].dot(chi[:3, 3:])
        return chi_inv

    @classmethod
    def log(cls, chi):
        """Logarithmic map for :math:`SE_k(3)`, which computes a tangent vector 
        from a transformation:

        .. math::
        
            \\boldsymbol{\\xi}(\\mathbf{T}) =
            \\ln(\\mathbf{T})^\\vee =
            \\begin{bmatrix}
                \\ln(\\boldsymbol{C}) ^\\vee \\\\
                \\mathbf{J} ^ {-1} \\mathbf{r}_1 \\\\ \\vdots \\\\
                \\mathbf{J} ^ {-1} \\mathbf{r}_k
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SEK3.exp`.
        """
        phi = SO3.log(chi[:3, :3])
        Xi = SO3.inv_left_jacobian(phi).dot(chi[:3, 3:])
        xi = np.hstack([phi, Xi.flatten('F')])
        return xi
