function [phi] = so3_log(Rot)
%SO3_LOG logarithm
%
% Syntax:  [phi] = so3_log(Rot)
%
% Inputs:
%    Rot - rotation matrix
%
% Outputs:
%    phi - vector

TOL = 1e-9;
cos_angle = 0.5 * trace(Rot) - 0.5;
% Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
cos_angle = min(max(cos_angle, -1), 1);
angle = acos(cos_angle);

% If angle is close to zero, use first-order Taylor expansion
if norm(angle) < TOL
    phi = so3_vee(Rot - eye(3));
else
    % Otherwise take the matrix logarithm and return the rotation vector
    phi = so3_vee((0.5 * angle / sin(angle)) * (Rot - Rot'));
end
end

