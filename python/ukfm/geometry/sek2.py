import numpy as np
from ukfm.geometry.so2 import SO2


class SEK2:
    """Homogeneous transformation matrix in :math:`SE_k(2)`

    .. math::
    
        SE_k(2) &= \\left\\{ \\mathbf{T}=
                \\begin{bmatrix}
                    \\mathbf{C} & \\mathbf{r_1} & \\cdots &\\mathbf{r}_k
                        \\\\
                    \\mathbf{0}^T & &\\mathbf{I}&
                \\end{bmatrix} \\in \\mathbb{R}^{(2+k) \\times (2+k)} 
                ~\\middle|~ \\mathbf{C} \\in SO(2), \\mathbf{r}_1 
                \\in \\mathbb{R}^2, \cdots, \\mathbf{r}_k \\in 
                \\mathbb{R}^2 \\right\\} \\\\
        \\mathfrak{se}_k(2) &= \\left\\{ \\boldsymbol{\\Xi} =
        \\boldsymbol{\\xi}^\\wedge \\in \\mathbb{R}^{(2+k) 
        \\times (2+k)} ~\\middle|~
            \\boldsymbol{\\xi}=
            \\begin{bmatrix}
                \\phi \\\\ \\boldsymbol{\\rho}_1 \\\\ \\vdots \\\\ 
                \\boldsymbol{\\rho}_k
            \\end{bmatrix} \\in \\mathbb{R}^{1+2k}, \\boldsymbol{\\rho}_1 
            \\in \\mathbb{R}^2, \\cdots, \\boldsymbol{\\rho}_k \\in 
            \\mathbb{R}^2, \\phi \in \\mathbb{R} \\right\\}

    """

    TOL = 1e-8

    @classmethod
    def exp(cls, xi):
        """Exponential map for :math:`SE_k(2)`, which computes a transformation 
        from a tangent vector:

        .. math::

            \\mathbf{T}(\\boldsymbol{\\xi}) =
            \\exp(\\boldsymbol{\\xi}^\\wedge) =
            \\begin{bmatrix}
                \\exp(\\phi ^\\wedge) & \\mathbf{J} \\boldsymbol{\\rho}_1 & 
                \\cdots && \\mathbf{J} \\boldsymbol{\\rho}_k  \\\\
                \\mathbf{0} ^ T & & \\mathbf{I} &
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SEK2.log`.
        """
        k = int((xi.shape[0]-1)/2)
        Xi = np.reshape(xi[1:], (2, k), order='F')
        chi = np.eye(2+k)
        chi[:2, :2] = SO2.exp(xi[0])
        chi[:2, 2:] = SO2.left_jacobian(xi[0]).dot(Xi)
        return chi

    @classmethod
    def inv(cls, chi):
        """Inverse map for :math:`SE_k(2)`.

        .. math::

            \\mathbf{T}^{-1} =
            \\begin{bmatrix}
                \\mathbf{C}^T  & -\\mathbf{C}^T \\boldsymbol{\\rho}_1  &
                    \\cdots & & -\\mathbf{C}^T \\boldsymbol{\\rho}_k \\\\
                \\mathbf{0} ^ T & & \\mathbf{I} &
            \\end{bmatrix}

        """
        k = chi.shape[0] - 2
        chi_inv = np.eye(2+k)
        chi_inv[:2, :2] = chi[:2, :2].T
        chi_inv[:2, 2:] = -chi[:2, :2].T.dot(chi[:2, 2:])
        return chi_inv

    @classmethod
    def log(cls, chi):
        """Logarithmic map for :math:`SE_k(2)`, which computes a tangent vector 
        from a transformation:

        .. math::

            \\boldsymbol{\\xi}(\\mathbf{T}) =
            \\ln(\\mathbf{T})^\\vee =
            \\begin{bmatrix}
            \\ln(\\boldsymbol{C}) ^\\vee \\\\
                \\mathbf{J} ^ {-1} \\mathbf{r}_1 \\\\
                \\vdots \\\\
                \\mathbf{J} ^ {-1} \\mathbf{r}_k
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SEK2.log`.
        """
        phi = SO2.log(chi[:2, :2])
        Xi = SO2.inv_left_jacobian(phi).dot(chi[:2, 2:])
        xi = np.hstack([phi, Xi.flatten('F')])
        return xi
