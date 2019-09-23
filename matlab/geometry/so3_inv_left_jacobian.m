function [J] = so3_inv_left_jacobian(phi)
%SO3_INV_LEFT_JACOBIAN inverse of Jacobian for SO(3)
%
% Syntax:  [J] = so3_inv_left_jacobian(phi)
%
% Inputs:
%    phi - vector of length 3
%
% Outputs:
%    J - Jacobian, matrix of size 3x3

TOL = 1e-9;

angle = norm(phi);
if angle < TOL
    % Near |phi|==0, use first order Taylor expansion
    J = eye(3) - 1/2 * so3_wedge(phi);
else
    axis = phi / angle;
    half_angle = angle/2;
    J = half_angle * cot(half_angle) * eye(3) + ...
        (1 - half_angle * cot(half_angle)) * (axis*axis') -...
        half_angle * so3_wedge(axis);
end
end
