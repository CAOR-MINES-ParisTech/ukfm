function [states, omegas] = inertial_navigation_simu_f(T, imu_freq, ...
    imu_noise_std)
%INERTIAL_NAVIGATION_SIMU_F Perform simulation
%
% Syntax: [state, omega] = inertial_navigation_simu_f(T, imu_freq,
% imu_noise_std)
%
% Inputs:
%    T - sequence time duration 
%    imu_freq - IMU frequency 
%    imu_noise_std - IMU noise standard deviation
%
% Outputs:
%    states - true state for the sequence, array of structure
%    omegas - noisy input of the sequence array of structure
%    timestamp n is at column n-1

% total number of timestamps
N = T*imu_freq;
% integration step (s)
dt = 1/imu_freq;
% rayon (m)
r = 5;

% set noise to zero to compute true trajectory
w = zeros(6, 1);

% gravity for accelerometer measurement (m/s^2)
g = [0; 0; -9.82];

% compute acceleration from trajectory
t = 0:dt:T-dt;
p = r*[sin(t/T*2*pi); cos(t/T*2*pi); zeros(1, N)];
v = [zeros(3,1),diff(p,1,2)]/dt;
acc = [zeros(3,1),diff(v,1,2)]/dt;


% init variables at zero and do for loop
omegas(N) = struct;
states(N) = struct;
states(1).Rot = eye(3);
states(1).v = v(:, 1);
states(1).p = p(:, 1);
for n = 2:N
    % true input
    omegas(n-1).gyro = zeros(3, 1);
    omegas(n-1).acc = states(n-1).Rot'*(acc(:, n-1) - g);
    % propagate state
    states(n) = inertial_navigation_f(states(n-1), omegas(n-1), w, dt); 
    % noisy input
    omegas(n-1).gyro = omegas(n-1).gyro + imu_noise_std(1)*randn(3, 1);
    omegas(n-1).acc = omegas(n-1).acc + imu_noise_std(2)*randn(3, 1);
end
end
