import numpy as np


class SO2:
    """Rotation matrix in :math:`SO(2)`.

    .. math::

        SO(2) &= \\left\\{ \\mathbf{C} 
        \\in \\mathbb{R}^{2 \\times 2} ~\\middle|~ 
        \\mathbf{C}\\mathbf{C}^T = \\mathbf{1}, \\det \\mathbf{C} = 
        1 \\right\\} \\\\
        \\mathfrak{so}(2) &= \\left\\{ \\boldsymbol{\\Phi} = \\phi^\\wedge \\in 
        \\mathbb{R}^{2 \\times 2} ~\\middle|~ \\phi \\in \\mathbb{R} \\right\\}

    """

    TOL = 1e-8

    @classmethod
    def exp(cls, phi):
        """Exponential map for :math:`SO(2)`, which computes a transformation 
        from a tangent vector:

        .. math::

            \\mathbf{C}(\\phi) =
            \\exp(\\phi^\\wedge) =
            \\cos \\phi \\mathbf{1} + \\sin \\phi 1^\\wedge =
            \\begin{bmatrix}
                \\cos \\phi  & -\\sin \\phi  \\\\
                \\sin \\phi & \\cos \\phi
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SO2.log`.
        """
        Rot = np.empty((2, 2))
        Rot[0, 0] = np.cos(phi)
        Rot[1, 1] = np.cos(phi)
        Rot[0, 1] = -np.sin(phi)
        Rot[1, 0] = np.sin(phi)
        return Rot

    @classmethod
    def inv_left_jacobian(cls, phi):
        """:math:`SO(2)` inverse left Jacobian.

        .. math::

            \\mathbf{J}^{-1}(\\phi) =
            \\begin{cases}
                \\mathbf{1} - \\frac{1}{2} \\phi^\wedge, & \\text{if } \\phi 
                \\text{ is small} \\\\
                \\frac{\\phi}{2} \\cot \\frac{\\phi}{2} \\mathbf{1} -
                \\frac{\\phi}{2} 1^\\wedge, & \\text{otherwise}
            \\end{cases}

        """
        if np.linalg.norm(phi) < cls.TOL:
            J = np.eye(2) - 1/2 * cls.wedge(phi)
        else:
            half_theta = phi/2
            cot = 1 / np.tan(half_theta)
            J = half_theta * cot * np.eye(2) - half_theta * cls.wedge(1)
        return J

    @classmethod
    def left_jacobian(cls, phi):
        """:math:`SO(2)` left Jacobian.

        .. math::

            \\mathbf{J}(\\phi) =
            \\begin{cases}
                \\mathbf{1} + \\frac{1}{2} \\phi^\wedge, & \\text{if } \\phi 
                \\text{ is small} \\\\
                \\frac{\\sin \\phi}{\\phi} \\mathbf{1} -
                \\frac{1 - \\cos \\phi}{\\phi} 1^\\wedge, & \\text{otherwise}
            \\end{cases}

        """
        if np.linalg.norm(phi) < cls.TOL:
            # Near |phi|==0, use first order Taylor expansion
            J = np.eye(2) + 1/2 * cls.wedge(phi)
        else:
            J = np.sin(phi)/phi * np.eye(2) + \
                (1-np.cos(phi))/phi * cls.wedge(1)
        return J

    @classmethod
    def log(cls, Rot):
        """Logarithmic map for :math:`SO(2)`, which computes a tangent vector 
        from a transformation:

        .. math::

            \\phi(\\mathbf{C}) =
            \\ln(\\mathbf{C})^\\vee =
            \\text{atan2}(C_{1,0}, C_{0,0})

        This is the inverse operation to :meth:`~ukfm.SO2.exp`.
        """
        phi = np.arctan2(Rot[1, 0], Rot[0, 0])
        return phi

    @classmethod
    def wedge(cls, phi):
        """:math:`SO(2)` wedge (skew-symmetric) operator.

        .. math::
        
            \\boldsymbol{\\Phi} =
            \\phi^\\wedge =
            \\begin{bmatrix}
                0 & -\\phi \\\\
                \\phi & 0
            \\end{bmatrix}
            
        """
        Phi = np.array([[0, -phi],
                        [phi, 0]])
        return Phi
