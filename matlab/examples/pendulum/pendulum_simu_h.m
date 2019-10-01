function [ys, one_hot_ys] = pendulum_simu_h(states, T, freq, obs_freq, ...
    obs_noise_std)
%PENDULUM_SIMU_H Perform measurement simulation
%
% Syntax: [ys, one_hot_ys] = pendulum_simu_h(states, T, freq, ...
%   obs_freq, obs_noise_std)
%
% Inputs.
%    states - states
%    T - sequence time duration
%    freq - model frequency
%    obs_freq - observation frequency
%    obs_noise_std - observation noise standard deviation
%
% Outputs:
%    ys - noisy measurement for the sequence, matrix. The n-th measurement
%    is the n-th column. Do not confuse measurement number and timestamp
%    one_hot_ys - one hot encoding to know where measurements happen

% total number of timestamps
N = freq*T;
% vector to know where measurements happen
one_hot_ys = zeros(N, 1);
one_hot_ys(1:freq/obs_freq:end) = 1; % freq/obs_freq must be integer
idxs = find(one_hot_ys); % indexes where measurement happen
% total number of measurements
N_y = length(idxs);
% measurement iteration number
n_y = 1;
ys = zeros(2, N_y);
for n = 1:N_y
    ys(:, n_y) = pendulum_h(states(idxs(n))) + obs_noise_std*randn(2, 1);
    n_y = n_y + 1;
end
end
