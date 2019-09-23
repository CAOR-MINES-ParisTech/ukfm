%% Attitude Estimation with an IMU - Example
% Goals of this script
% 
% * applying the UKF for estimating 3D attitude from an IMU.
%
% _We assume the reader is already familiar with the tutorial._
%
% Attitude estimation with an Inertial Measurement Unit (IMU). The filter
% fuses measurements comming from gyro, accelerometer and magnetometer. The
% IMU does not have any bias. We reproduce the simulation with the
% parameters based on [KHSchon17].

%% Initialization 
% Start by cleaning the workspace.
clear all;
close all;

%% Simulation
% The true trajectory is computed along with noisy inputs after we define
% the noise standard deviation affecting the (accurate) IMU, where the
% platform is 2 s stationary and then have constant angular velocity around
% gravity.

% sequence time (s)
T = 100; 
% IMU frequency (Hz)
imu_freq = 100; 
% IMU standard-deviation noise (noise is isotropic)
imu_noise_std = [5/180*pi; % gyro (rad/s)
                0.4;       % accelerometer (m/s^2)
                0.2];      % magnetometer
            
% total number of timestamps
N = T*imu_freq;
% time between succesive timestamps (s)
dt = 1/imu_freq;

% simulate true trajectory and noised input
[states, omegas] = attitude_simu_f(T, imu_freq, imu_noise_std);
% simulate accelerometer and magnetometer measurements
ys = attitude_simu_h(states, T, imu_freq, imu_noise_std);         
%%
% The state and the input contain the following variables:
%
%   states(n).Rot  % 3d orientation (matrix)
%   omegas(n).gyro % robot angular velocities 
%
% A measurement ys(:, k) contains accelerometer and magnetometer
% measurement.

%% Filter Design and Initialization
% We choose in this example to embed the state in $SO(3)$ with left
% multiplication, such that:
%
% * the retraction $\varphi(.,.)$ is the $SO(3)$ exponential map for
%   orientation where the state multiplies the uncertainty on the left.
%
% * the inverse retraction $\varphi^{-1}(.,.)$ is the $SO(3)$
%   logarithm for orientation.

% propagation noise matrix
ukf_Q = imu_noise_std(1).^2*eye(3);
% measurement noise matrix
ukf_R = blkdiag(imu_noise_std(2).^2*eye(3), imu_noise_std(3).^2*eye(3));
% initial error matrix
ukf_P0 = zeros(3, 3); % The state is perfectly initialized
% sigma point parameters
ukf_alpha = [1e-3, 1e-3, 1e-3];

% asses UKF function
ukf_f = @attitude_f;
ukf_h = @attitude_h;
ukf_phi = @attitude_phi;
ukf_phi_inv = @attitude_phi_inv;
ukf_weights = ukf_set_weight(length(ukf_P0), length(ukf_R), ukf_alpha);
ukf_cholQ = chol(ukf_Q);

% initialize with true state
ukf_state = states(1);
ukf_P = ukf_P0;

% variables for recording estimates
ukf_states = ukf_state;
ukf_Ps = zeros(N, 3, 3);
ukf_Ps(1, :, :) = ukf_P;

%% Filtering
% The UKF proceeds as a standard Kalman filter with a simple for loop.
for n = 2:N
    % propagation
    [ukf_state, ukf_P] = ukf_propagation(ukf_state, ukf_P, omegas(n-1), ...
        ukf_f, dt, ukf_phi, ukf_phi_inv, ukf_cholQ, ukf_weights);
    % update
   [ukf_state, ukf_P] = ukf_update(ukf_state, ukf_P, ys(:, n), ukf_h, ...
       ukf_phi, ukf_R, ukf_weights);
    % save estimates
    ukf_states(n) = ukf_state;
    ukf_Ps(n, :, :) = ukf_P;
end

%% Results
% We plot the orientation as function of time along with the orientation
% error.
attitude_results_plot(ukf_states, ukf_Ps, states, omegas, dt);

%%
% We see the true trajectory starts by a small stationary step following by
% constantly turning around the gravity vector (only the yaw is
% increasing). As yaw is not observable with an accelerometer only, it is
% expected that yaw error would be stronger than roll or pitch errors.
%
% As UKF estimates the covariance of the error, we have plotted the 95%
% confident interval ($3\sigma$). We expect the error keeps behind this
% interval, and in this situation the filter covariance output matches
% especially well the error.
%
% A cruel aspect of these curves is the absence of comparision. Is the
% filter good ? It would be nice to compare it, e.g., to an extended Kalman
% filter.

%% Conclusion
% We have seen in this script how well works the UKF on parallelizable
% manifolds for estimating orientation from an IMU.
%
% You can now:
%
% * address the UKF for the same problem with different noise parameters.
% 
% * add outliers in acceleration or magnetometer measurements.
%
% * benchmark the UKF with different function errors and compare it to the
%   extended Kalman filter in the benchmarks folder.