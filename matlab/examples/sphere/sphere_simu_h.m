function [y] = sphere_simu_h(state, T, odo_freq, ...
    gps_freq, gps_noise_std)
%LOCALIZATION_SIMU_H Perform measurement simulation
%
% Syntax: [y, one_hot_y] = localization_simu_h(state, T, odo_freq, ...
%   gps_freq, gps_noise_std)
%
% Inputs.
%    state - state, array of structure
%    T - sequence time duration
%    odo_freq - odometry frequency
%    gps_freq - GPS frequency
%    gps_noise_std - GPS noise standard deviation
%
% Outputs:
%    y - noisy measurements

% total number of timestamps
N = odo_freq*T;

% vector to know where GPS measurement happen
one_hot_y = zeros(N, 1);
one_hot_y(1:odo_freq/gps_freq:end) = 1; % odo_freq/gps_freq must be integer
idxs = find(one_hot_y); % indexes where measurement happen
% total number of measurements
N_y = length(idxs);

% measurement iteration number
n_y = 1;
y = zeros(2, N_y);
for n = 1:N_y
    % GPS measurement
    y(:, n_y) = state(idxs(n)).p + gps_noise_std*randn(2, 1);
    n_y = n_y + 1;
end
end
