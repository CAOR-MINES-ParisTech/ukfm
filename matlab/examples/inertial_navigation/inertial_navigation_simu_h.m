function [ys, one_hot_ys] = inertial_navigation_simu_h(states, T, ...
    imu_freq, obs_freq, obs_noise_std)
%INERTIAL_NAVIGATION_SIMU_H Perform measurement simulation
%
% Syntax: [ys, one_hot_ys] = inertial_navigation_simu_h(states, T, ...
%     imu_freq, obs_freq, obs_noise_std)
%
% Inputs.
%    states - states
%    T - sequence time duration
%    imu_freq - IMU frequency
%    obs_freq - observation frequency
%    obs_noise_std - observation noise standard deviation
%
% Outputs:
%    ys - noisy measurement for the sequence, matrix. The n-th measurement
%    is the n-th column. Do not confuse measurement number and timestamp
%    one_hot_ys - one hot encoding to know where measurements happen

% total number of timestamps
N = imu_freq*T;

% landmarks
ldk = [[0; 2; 2], [-2; -2; -2], [2; -2; -2]];
N_ldk = size(ldk, 2);
% vector to know where measurement happen
one_hot_ys = zeros(N, 1);
one_hot_ys(1:imu_freq/obs_freq:end) = 1; 
% imu_freq/obs_freq must be integer
idxs = find(one_hot_ys); % indexes where measurement happen
% total number of measurements
K = length(idxs);

% measurement iteration number
ys = zeros(3*N_ldk, K);
for k = 1:K
    Rot = states(idxs(k)).Rot;
    p = states(idxs(k)).p;
    for n_ldk = 1:N_ldk
        % observation measurement
        ys(3*n_ldk-2: 3*n_ldk, k) = Rot'*(ldk(:, n_ldk)-p) + ...
            obs_noise_std*randn(3, 1);
    end
end
end