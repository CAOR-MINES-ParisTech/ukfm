function [Rot] = so3_from_rpy(rpy)
%SO3_FROM_RPY Convert RPY Euler angles to a rotation matrix
%
% Syntax:  [Rot] = so3_from_rpy(rpy)
%
% Inputs:
%    rpy - roll, pitch, yaw
%
% Outputs:
%    Rot - rotation matrix

c = cos(rpy);
s = sin(rpy);
Rotx = [[1, 0, 0]; [0, c(1), -s(1)]; [0, s(1), c(1)]]; 
Roty = [[c(2), 0, s(2)]; [0, 1, 0]; [-s(2), 0, c(2)]]; 
Rotz = [[c(3), -s(3), 0]; [s(3), c(3), 0]; [0, 0, 1]]; 
Rot = Rotz*Roty*Rotx;
end

