function [nees] = localization_nees(errs, Ps, hat_Rots, hat_ps, name)
%LOCALIZATION_NEES
%
% Syntax: [errors] = localization_nees(Rots, Ps, hat_Rots, ps, name)
%
% Inputs:
%    errs -errors
%    Ps - covariance matrices
%    hat_Rots - estimated rotation matrices
%    hat_ps - estimated positions
%    name - filter name
%
% Outputs:
%    nees - Normalized Estimation Error Squared

N = length(errs);
nees = zeros(2, N);
J0 = [[0, -1]; [0, 1]];
J = eye(3);
for n = 2000:N
    if name == "STD"
        P = squeeze(Ps(n, :, :));
    elseif name == "LEFT"
        J(2:3, 2:3) = hat_Rots{n};
        P = J*squeeze(Ps(n, :, :))*J';
    else
        J(2:3, 1) = J0*(hat_ps(n, :)');
        P = J*squeeze(Ps(n, :, :))*J';
    end
            
    nees(1, n) = errs(1, n)^2/P(1, 1);
    nees(2, n) = (errs(2:3, n)'*inv(P(2:3, 2:3))*errs(2:3, n))/2;
end
end

