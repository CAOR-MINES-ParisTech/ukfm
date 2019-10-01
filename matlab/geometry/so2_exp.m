function [Rot] = so2_exp(phi)
%SO2_EXP exponential
%
% Syntax:  [Rot] = so2_exp(phi)
%
% Inputs:
%    phi - scalar
%
% Outputs:
%    Rot - rotation matrix

Rot = [cos(phi), -sin(phi);
    sin(phi), cos(phi)];
end

