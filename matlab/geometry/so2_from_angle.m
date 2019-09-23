function [Rot] = so2_from_angle(phi)
%SO2_FROM_ANGLE Form a rotation matrix given an angle in radians
%
% Syntax:  [Rot] = so2_from_angle(phi)
%
% Inputs:
%    phi - scalar
%
% Outputs:
%    Rot - rotation matrix

Rot = so2_exp(phi);
end

