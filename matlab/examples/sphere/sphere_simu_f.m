function [state, omega] = sphere_simu_f(T, freq, obs_noise_std, radius)
%SPHERE_SIMU_F Perform simulation
%
% Syntax: [state, omega] = sphere_simu_f(T, freq, obs_noise_std, radius)
%
% Inputs:
%    T - sequence time duration
%    odo_freq - odometry frequency
%    odo_noise_std - odometry noise standard deviation
%    radius - radius of the trajectory (m)
%
% Outputs:
%    state - true state for the sequence, described as a array of structure
%    omega - noisy input of the sequence, described as a array of structure
%    timestamp n is at column n-1

% total number of timestamps
N = T*odo_freq;
% integration step (s)
dt = 1/odo_freq;

% the robot moves along a circle with constant true input
v = [2*pi*radius/T; 0]; % forward speed (m/s)
gyro = 2*pi/T; % angular speed (rad/s). 
% We thus do one turn during the simulation

% set noise to zero to compute the true trajectory
w = zeros(3, 1);

% init variables at zero and do for loop
omega(N) = struct;
state(N) = struct;
state(1).Rot = eye(2);
state(1).p = zeros(2, 1);
for n = 2:N
    % true input
    omega(n-1).v = v;
    omega(n-1).gyro = gyro;    
    % propagate state
    state(n) = localization_f(state(n-1), omega(n-1), w, dt); 
    % noisy input
    omega(n-1).v = omega(n-1).v + odo_noise_std(1:2).*randn(2, 1);
    omega(n-1).gyro = omega(n-1).gyro + odo_noise_std(3)*randn(1);
end
end

