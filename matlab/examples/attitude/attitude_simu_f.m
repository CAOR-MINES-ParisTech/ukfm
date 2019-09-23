function [states, omegas] = attitude_simu_f(T, imu_freq, imu_noise_std)
%ATTITUDE_SIMU_F Perform simulation
%
% Syntax: [states, omegas] = attitude_simu_f(T, imu_freq, imu_noise_std)
%
% Inputs:
%    T - sequence time duration
%    imu_freq - IMU frequency
%    imu_noise_std - IMU noise standard deviation
%
% Outputs:
%    states - true state for the sequence
%    omegas - noisy input of the sequence

% total number of timestamps
N = T*imu_freq;
% integration step (s)
dt = 1/imu_freq;

% The robot is 2 s stationnary and then have constant angular velocity
% around gravity

n_T = 0; % increment for angular velocity
omega_T = zeros(3, 1); % first velocity (robot is first stationnary)
omega_move = [0; 0; 10/180*pi];

% set noise to zero to compute true trajectory
w = zeros(3, 1);

% init variables at zero and do for loop
omegas(N) = struct;
states(N) = struct;
states(1).Rot = eye(3);
for n = 2:N
    % change true input
    if n_T > 2
        omega_T = omega_move;
    end
    n_T = n_T + dt;
    % true input
    omegas(n-1).gyro = omega_T;
    % propagate state
    states(n) = attitude_f(states(n-1), omegas(n-1), w, dt); 
    % noisy input
    omegas(n-1).gyro = omegas(n-1).gyro + imu_noise_std(1)*randn(3, 1); 
end
end