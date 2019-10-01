function [chi] = se_k_3_exp(xi)
%SE_K_3_EXP exponential
%
% Syntax:  [chi] = se_k_3_exp(xi)
%
% Inputs:
%    phi - vector
%
% Outputs:
%    chi - matrix

k = length(xi)/3 - 1;
Xi = reshape(xi(4:end), 3, k);
chi = [so3_exp(xi(1:3)) so3_left_jacobian(xi(1:3))*Xi;
    zeros(k, 3) eye(k)];
end

