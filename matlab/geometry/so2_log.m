function [phi] = so2_log(Rot)
%SO2_LOG logarithm
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
