function [nees] = inertial_navigation_nees(errs, Ps, hat_Rots, hat_vs, ...
    hat_ps, name)
%INERTIAL_NAVIGATION_NEES
%
% Syntax: [errors] = inertial_navigation_nees(Rots, Ps, hat_Rots, hat_vs, ...
%    hat_ps, name)
%
% Inputs:
%    errs -errors
%    Ps - covariance matrices
%    hat_Rots - estimated rotation matrices
%    hat_vs - estimated velocities
%    hat_ps - estimated positions
%    name - filter name
%
% Outputs:
%    nees - Normalized Estimation Error Squared

N = length(errs);
nees = zeros(2, N);
J = eye(9);
for n = 2:N
    if name == "STD"
        P = squeeze(Ps(n, :, :));
    elseif name == "LEFT"
        J(4:6, 4:6) = hat_Rots{n};
        J(7:9, 7:9) = hat_Rots{n};
        P = J*squeeze(Ps(n, :, :))*J';
    else
        J(4:6, 1:3) = so3_wedge(hat_vs(n, :)');
        J(7:9, 1:3) = so3_wedge(hat_ps(n, :)');
        P = J*squeeze(Ps(n, :, :))*J';
    end
    nees(1, n) = (errs(1:3, n)'*inv(P(1:3, 1:3))*errs(1:3, n))/3;
    nees(2, n) = (errs(7:9, n)'*inv(P(7:9, 7:9))*errs(7:9, n))/3;
end
end

