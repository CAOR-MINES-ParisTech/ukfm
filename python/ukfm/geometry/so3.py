import numpy as np


class SO3:
    """Rotation matrix in :math:`SO(3)`

    .. math::

        SO(3) &= \\left\\{ \\mathbf{C} \\in \\mathbb{R}^{3 \\times 3} 
        ~\\middle|~ \\mathbf{C}\\mathbf{C}^T = \\mathbf{1}, \\det
            \\mathbf{C} = 1 \\right\\} \\\\
        \\mathfrak{so}(3) &= \\left\\{ \\boldsymbol{\\Phi} = 
        \\boldsymbol{\\phi}^\\wedge \\in \\mathbb{R}^{3 \\times 3} 
        ~\\middle|~ \\boldsymbol{\\phi} = \\phi \\mathbf{a} \\in \\mathbb{R}
        ^3, \\phi = \\Vert \\boldsymbol{\\phi} \\Vert \\right\\}

    """

    #  tolerance criterion
    TOL = 1e-8
    Id_3 = np.eye(3)

    @classmethod
    def Ad(cls, Rot):
        """Adjoint matrix of the transformation.

        .. math::

            \\text{Ad}(\\mathbf{C}) = \\mathbf{C}
            \\in \\mathbb{R}^{3 \\times 3}

        """
        return Rot

    @classmethod
    def exp(cls, phi):
        """Exponential map for :math:`SO(3)`, which computes a transformation 
        from a tangent vector:

        .. math::

            \\mathbf{C}(\\boldsymbol{\\phi}) =
            \\exp(\\boldsymbol{\\phi}^\wedge) =
            \\begin{cases}
                \\mathbf{1} + \\boldsymbol{\\phi}^\wedge, 
                & \\text{if } \\phi \\text{ is small} \\\\
                \\cos \\phi \\mathbf{1} +
                (1 - \\cos \\phi) \\mathbf{a}\\mathbf{a}^T +
                \\sin \\phi \\mathbf{a}^\\wedge, & \\text{otherwise}
            \\end{cases}

        This is the inverse operation to :meth:`~ukfm.SO3.log`.
        """
        angle = np.linalg.norm(phi)
        if angle < cls.TOL:
            # Near |phi|==0, use first order Taylor expansion
            Rot = cls.Id_3 + SO3.wedge(phi)
        else:
            axis = phi / angle
            c = np.cos(angle)
            s = np.sin(angle)
            Rot = c * cls.Id_3 + (1-c)*np.outer(axis,
                                                axis) + s * cls.wedge(axis)
        return Rot

    @classmethod
    def inv_left_jacobian(cls, phi):
        """:math:`SO(3)` inverse left Jacobian

        .. math::

            \\mathbf{J}^{-1}(\\boldsymbol{\\phi}) =
            \\begin{cases}
                \\mathbf{1} - \\frac{1}{2} \\boldsymbol{\\phi}^\wedge, &
                    \\text{if } \\phi \\text{ is small} \\\\
                \\frac{\\phi}{2} \\cot \\frac{\\phi}{2} \\mathbf{1} +
                \\left( 1 - \\frac{\\phi}{2} \\cot \\frac{\\phi}{2} 
                \\right) \\mathbf{a}\\mathbf{a}^T -
                \\frac{\\phi}{2} \\mathbf{a}^\\wedge, & 
                \\text{otherwise}
            \\end{cases}

        """
        angle = np.linalg.norm(phi)
        if angle < cls.TOL:
            # Near |phi|==0, use first order Taylor expansion
            J = np.eye(3) - 1/2 * cls.wedge(phi)
        else:
            axis = phi / angle
            half_angle = angle/2
            cot = 1 / np.tan(half_angle)
            J = half_angle * cot * cls.Id_3 + \
                (1 - half_angle * cot) * np.outer(axis, axis) -\
                half_angle * cls.wedge(axis)
        return J

    @classmethod
    def left_jacobian(cls, phi):
        """:math:`SO(3)` left Jacobian.

        .. math::

            \\mathbf{J}(\\boldsymbol{\\phi}) =
            \\begin{cases}
                \\mathbf{1} + \\frac{1}{2} \\boldsymbol{\\phi}^\wedge, &
                    \\text{if } \\phi \\text{ is small} \\\\
                \\frac{\\sin \\phi}{\\phi} \\mathbf{1} +
                \\left(1 - \\frac{\\sin \\phi}{\\phi} \\right) 
                \\mathbf{a}\\mathbf{a}^T +
                \\frac{1 - \\cos \\phi}{\\phi} \\mathbf{a}^\\wedge, & 
                \\text{otherwise}
            \\end{cases}

        """
        angle = np.linalg.norm(phi)
        if angle < cls.TOL:
            # Near |phi|==0, use first order Taylor expansion
            J = cls.Id_3 - 1/2 * SO3.wedge(phi)
        else:
            axis = phi / angle
            s = np.sin(angle)
            c = np.cos(angle)
            J = (s / angle) * cls.Id_3 + \
                (1 - s / angle) * np.outer(axis, axis) +\
                ((1 - c) / angle) * cls.wedge(axis)
        return J

    @classmethod
    def log(cls, Rot):
        """Logarithmic map for :math:`SO(3)`, which computes a tangent vector 
        from a transformation:

        .. math::

            \\phi &= \\frac{1}{2} 
            \\left( \\mathrm{Tr}(\\mathbf{C}) - 1 \\right) \\\\
            \\boldsymbol{\\phi}(\\mathbf{C}) &=
            \\ln(\\mathbf{C})^\\vee =
            \\begin{cases}
                \\mathbf{1} - \\boldsymbol{\\phi}^\wedge, 
                & \\text{if } \\phi \\text{ is small} \\\\
                \\left( \\frac{1}{2} \\frac{\\phi}{\\sin \\phi} 
                \\left( \\mathbf{C} - \\mathbf{C}^T \\right) 
                \\right)^\\vee, & \\text{otherwise}
            \\end{cases}

        This is the inverse operation to :meth:`~ukfm.SO3.log`.
        """
        cos_angle = 0.5 * np.trace(Rot) - 0.5
        # Clip np.cos(angle) to its proper domain to avoid NaNs from rounding
        # errors
        cos_angle = np.min([np.max([cos_angle, -1]), 1])
        angle = np.arccos(cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.linalg.norm(angle) < cls.TOL:
            phi = cls.vee(Rot - cls.Id_3)
        else:
            # Otherwise take the matrix logarithm and return the rotation vector
            phi = cls.vee((0.5 * angle / np.sin(angle)) * (Rot - Rot.T))
        return phi

    @classmethod
    def to_rpy(cls, Rot):
        """Convert a rotation matrix to RPY Euler angles 
        :math:`(\\alpha, \\beta, \\gamma)`."""

        pitch = np.arctan2(-Rot[2, 0], np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

        if np.linalg.norm(pitch - np.pi/2) < cls.TOL:
            yaw = 0
            roll = np.arctan2(Rot[0, 1], Rot[1, 1])
        elif np.linalg.norm(pitch + np.pi/2.) < 1e-9:
            yaw = 0.
            roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
        else:
            sec_pitch = 1. / np.cos(pitch)
            yaw = np.arctan2(Rot[1, 0] * sec_pitch, Rot[0, 0] * sec_pitch)
            roll = np.arctan2(Rot[2, 1] * sec_pitch, Rot[2, 2] * sec_pitch)

        rpy = np.array([roll, pitch, yaw])
        return rpy

    @classmethod
    def vee(cls, Phi):
        """:math:`SO(3)` vee operator as defined by 
        :cite:`barfootAssociating2014`.

        .. math::

            \\phi = \\boldsymbol{\\Phi}^\\vee

        This is the inverse operation to :meth:`~ukfm.SO3.wedge`.
        """
        phi = np.array([Phi[2, 1], Phi[0, 2], Phi[1, 0]])
        return phi

    @classmethod
    def wedge(cls, phi):
        """:math:`SO(3)` wedge operator as defined by 
        :cite:`barfootAssociating2014`.

        .. math::

            \\boldsymbol{\\Phi} =
            \\boldsymbol{\\phi}^\\wedge =
            \\begin{bmatrix}
                0 & -\\phi_3 & \\phi_2 \\\\
                \\phi_3 & 0 & -\\phi_1 \\\\
                -\\phi_2 & \\phi_1 & 0
            \\end{bmatrix}

        This is the inverse operation to :meth:`~ukfm.SO3.vee`.
        """
        Phi = np.array([[0, -phi[2], phi[1]],
                        [phi[2], 0, -phi[0]],
                        [-phi[1], phi[0], 0]])
        return Phi

    @classmethod
    def from_rpy(cls, roll, pitch, yaw):
        """Form a rotation matrix from RPY Euler angles 
        :math:`(\\alpha, \\beta, \\gamma)`.

        .. math:: 
        
            \\mathbf{C} = \\mathbf{C}_z(\\gamma) \\mathbf{C}_y(\\beta)
            \\mathbf{C}_x(\\alpha)

        """
        return cls.rotz(yaw).dot(cls.roty(pitch).dot(cls.rotx(roll)))

    @classmethod
    def rotx(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the x-axis.

        .. math::

            \\mathbf{C}_x(\\phi) = 
            \\begin{bmatrix}
                1 & 0 & 0 \\\\
                0 & \\cos \\phi & -\\sin \\phi \\\\
                0 & \\sin \\phi & \\cos \\phi
            \\end{bmatrix}

        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[1., 0., 0.],
                         [0., c, -s],
                         [0., s,  c]])

    @classmethod
    def roty(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the y-axis.

        .. math::

            \\mathbf{C}_y(\\phi) = 
            \\begin{bmatrix}
                \\cos \\phi & 0 & \\sin \\phi \\\\
                0 & 1 & 0 \\\\
                \\sin \\phi & 0 & \\cos \\phi
            \\end{bmatrix}

        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[c,  0., s],
                         [0., 1., 0.],
                         [-s, 0., c]])

    @classmethod
    def rotz(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the z-axis.

        .. math::
        
            \\mathbf{C}_z(\\phi) = 
            \\begin{bmatrix}
                \\cos \\phi & -\\sin \\phi & 0 \\\\
                \\sin \\phi  & \\cos \\phi & 0 \\\\
                0 & 0 & 1
            \\end{bmatrix}

        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[c, -s,  0.],
                         [s,  c,  0.],
                         [0., 0., 1.]])
