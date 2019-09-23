function [chi] = se_k_2_exp(xi)
%SE_K_2_EXP exponential map for SE_k(2)
%
% Syntax:  [chi] = se_k_2_exp(xi)
%
% Inputs:
%    phi - vector of length 1 + 2*k
%
% Outputs:
%    chi - state matrix

k = (length(xi)-1)/2;
Xi = reshape(xi(2:end), 2, k);
chi = [so2_exp(xi(1)), so2_left_jacobian(xi(1))*Xi;
     zeros(k, 2) eye(k)];
end
