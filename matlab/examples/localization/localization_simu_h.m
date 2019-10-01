function [ys, one_hot_ys] = localization_simu_h(states, T, odo_freq, ...
    gps_freq, gps_noise_std)
%LOCALIZATION_SIMU_H Perform measurement simulation
%
% Syntax: [ys, one_hot_ys] = localization_simu_h(states, T, odo_freq, ...
%   gps_freq, gps_noise_std)
%
% Inputs.
%    states - states
%    T - time duration
%    odo_freq - odometry frequency
%    gps_freq - GPS frequency
%    gps_noise_std - GPS noise standard deviation
%
% Outputs:
%    ys - noisy measurement for the sequence, matrix. The k-th measurement
%    is the k-th column. Do not confuse measurement number and timestamp
%    one_hot_ys - one hot encoding to know where measurements happen

% total number of timestamps
N = odo_freq*T;

% vector to identify when measurements happen
one_hot_ys = zeros(N, 1);
one_hot_ys(1:odo_freq/gps_freq:end) = 1; % odo_freq/gps_freq must be integer
idxs = find(one_hot_ys); % indexes where measurement happen

% total number of measurements
K = length(idxs);

% measurement iteration number
k = 1;
ys = zeros(2, K);
for n = 1:K
    ys(:, k) = states(idxs(n)).p + gps_noise_std*randn(2, 1);
    k = k + 1;
end
end
