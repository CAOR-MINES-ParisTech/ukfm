function [ys] = attitude_simu_h(states, T, imu_freq, imu_noise_std)
%ATTITUDE_SIMU_H Perform measurement simulation 
%
% Syntax: [ys] = attitude_simu_h(states, T, imu_freq, imu_noise_std)
%
% Inputs.
%    states - state
%    T - sequence time duration
%    imu_freq - IMU frequency
%    imu_noise_std - IMU noise standard deviation
%
% Outputs:
%    ys - noisy measurement for the sequence, matrix.

% gravity for accelerometer measurement (m/s^2)
g = [0; 0; -9.81];
% normed magnetic fielf in Sweden for magnetometer measurement 
b = [0.33; 0; -0.95];

% total number of timestamps
N = imu_freq*T;

ys = zeros(6, N);
for n = 1:N
    ys(1:3, n) = states(n).Rot'*g + imu_noise_std(2)*randn(3, 1);
    ys(4:6, n) = states(n).Rot'*b + imu_noise_std(3)*randn(3, 1);
end
end