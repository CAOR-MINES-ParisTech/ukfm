function [states, omegas] = pendulum_simu_f(T, freq, model_noise_std)
%PENDULUM_SIMU_F Perform simulation
%
% Syntax: [states, omegas] = pendulum_simu_f(T, freq, model_noise_std)
%
% Inputs:
%    T - sequence time duration
%    freq - model frequency
%    model_noise_std - model noise standard deviation
%
% Outputs:
%    states - states
%    omegas- noisy inputs

% total number of timestamps
N = T*freq;
% integration step (s)
dt = 1/freq;
% set noise to zero to compute the true trajectory
w = zeros(6, 1);
rpy = [57.3/180*pi; 40/180*pi; 0];
% init variables at zero and do for loop
omegas(N) = struct;
states(N) = struct;
states(1).Rot = so3_from_rpy(rpy);
states(1).u = [-10/180*pi; 30/180*pi; 0];
for n = 2:N
    w(1:3) = model_noise_std(1)*randn(3, 1);
    w(4:6) = model_noise_std(2)*randn(3, 1);
    % propagate state
    states(n) = pendulum_f(states(n-1), omegas(n-1), w, dt); 
end
end

