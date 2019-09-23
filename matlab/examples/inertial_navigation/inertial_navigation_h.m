function [y] = inertial_navigation_h(state)
%INERTIAL_NAVIGATION_H Measurement
%
% Syntax: [y] = inertial_navigation_h(state)
%
% Inputs.
%    state - state
%
% Outputs:
%    y - measurement

% landmarks
ldk = [[0; 2; 2], [-2; -2; -2], [2; -2; -2]];
N_ldk = size(ldk, 2);
y = zeros(3*N_ldk, 1);
Rot = state.Rot;
p = state.p;
for n_ldk = 1:N_ldk
    % observation measurement
    y(3*n_ldk-2: 3*n_ldk) = Rot'*(ldk(:, n_ldk)-p);
end
end