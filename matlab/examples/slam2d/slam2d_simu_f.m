function [states, omegas, ldks] = slam2d_simu_f(T, odo_freq, ...
    odo_noise_std, v, gyro)
%SLAM2D_SIMU_F Perform simulation
%
% Syntax: [states, omegas, ldks] = slam2d_simu_f(T, odo_freq, ...
%   odo_noise_std, v, gyro)
%
% Inputs:
%    T - sequence time duration
%    odo_freq - odometry frequency
%    odo_noise_std - odometry noise standard deviation
%    radius - radius of the trajectory (m)
%
% Outputs:
%    states - true state
%    omegas - noisy inputs
%    ldks - landmarks

% total number of timestamps
N = T*odo_freq;
% integration step (s)
dt = 1/odo_freq;
w = zeros(2, 1);
N_ldk = 20; %number of landmarks
omegas(N) = struct;
states(N) = struct;
states(1).Rot = eye(2);
states(1).p = zeros(2, 1);
states(1).p_l = [];
for n = 2:N
    omegas(n-1).v = v;
    omegas(n-1).gyro = gyro;
    states(n) = slam2d_f(states(n-1), omegas(n-1), w, dt); 
    
    omegas(n-1).v = omegas(n-1).v + 0*odo_noise_std(1)*randn(1);
    omegas(n-1).gyro = omegas(n-1).gyro + 0*odo_noise_std(2)*randn(1);
end

% create the map
r = v/gyro; % radius
rmax = 5; % max range
rmin = 1; % min range
ldks = zeros(2, N_ldk);
for i = 1: N_ldk
    rho = r +  rmin + 2;
    th = 2*pi*i/N_ldk;
    [x, y] = pol2cart(th ,rho);
    ldks(:,i) = [x; y + r]; % shift y w/ r since robot starts at (0,0)
end
end

