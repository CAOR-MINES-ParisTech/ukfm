function [phi] = so2_log(Rot)
%SO2_LOG logarithm map for SO(2)
%
% Syntax:  [phi] = so2_log(Rot)
%
% Inputs:
%    Rot - rotation matrix
%
% Outputs:
%    phi - scalar

phi = atan2(Rot(2, 1), Rot(1, 1));
end
