function [nees] = attitude_nees(errs, Ps, hat_Rots, name)
%ATTITUDE_NEES
%
% Syntax: [errors] = attitude_nees(Rots, hat_Rots, name)
%
% Inputs:
%    errs -errors
%    Ps - covariance matrices
%    hat_Rots - estimated rotation matrices
%    name - filter name
%
% Outputs:
%    nees - Normalized Estimation Error Squared

N = length(errs);
nees = zeros(N, 1);
for n = 2:N
    if name == "LEFT"
        J = hat_Rots{n};
        P = J'*squeeze(Ps(n, :, :))*J;
    else
        P = squeeze(Ps(n, :, :));
    end
    nees(n) = (errs(:, n)'*inv(P)*errs(:, n))/3;
end
end

