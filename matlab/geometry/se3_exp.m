function [chi] = se3_exp(xi)
%SE3_EXP exponential map for SE(3)
%
% Syntax:  [chi] = se3_exp(xi)
%
% Inputs:
%    phi - vector of length 6
%
% Outputs:
%    chi - state matrix

chi = [se3_exp(xi(1:3)), so3_left_jacobian(xi(1:3))*xi(4:6);
    zeros(1, 3) 1];
end

