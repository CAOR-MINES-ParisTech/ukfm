function [chi] = se_k_2_exp(xi)
%SE_K_2_EXP exponential
%
% Syntax:  [chi] = se_k_2_exp(xi)
%
% Inputs:
%    phi - vector
%
% Outputs:
%    chi - matrix

k = (length(xi)-1)/2;
Xi = reshape(xi(2:end), 2, k);
chi = [so2_exp(xi(1)), so2_left_jacobian(xi(1))*Xi;
     zeros(k, 2) eye(k)];
end
