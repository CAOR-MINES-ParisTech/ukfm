function [states, omegas] = localization_simu_f(T, odo_freq, ...
    odo_noise_std, radius)
%LOCALIZATION_SIMU_F Perform simulation
%
% Syntax: [states, omegas] = localization_simu_f(T, odo_freq, ...
%   odo_noise_std, radius)
%
% Inputs:
%    T - time duration
%    odo_freq - odometry frequency
%    odo_noise_std - odometry noise standard deviation
%    radius - radius of the trajectory (m)
%
% Outputs:
%    states - states
%    omegas - noisy inputs

% total number of timestamps
N = T*odo_freq;
% integration step (s)
dt = 1/odo_freq;

% set input
v = [2*pi*radius/T; 0]; % forward speed (m/s)
gyro = 2*pi/T; % angular speed (rad/s)

% set noise to zero to compute the true trajectory
w = zeros(3, 1);

% init variables at zero and do for loop
omegas(N) = struct;
states(N) = struct;
states(1).Rot = eye(2);
states(1).p = zeros(2, 1);
for n = 2:N
    % true input
    omegas(n-1).v = v;
    omegas(n-1).gyro = gyro;    
    % propagate state
    states(n) = localization_f(states(n-1), omegas(n-1), w, dt); 
    % noisy input
    omegas(n-1).v = omegas(n-1).v + odo_noise_std(1:2).*randn(2, 1);
    omegas(n-1).gyro = omegas(n-1).gyro + odo_noise_std(3)*randn(1);
end
end

