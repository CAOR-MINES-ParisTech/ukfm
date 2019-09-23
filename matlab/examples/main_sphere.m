%% Attitude Estimation with an IMU
% Goals of this script
% 
% * applying the UKF for estimating 3D attitude from an IMU
%
% _We assume the reader is already familiar with the approach 
% described in main_localization.m._
%
% Attitude estimation with an Inertial Measurement Unit (IMU). The filter
% fuses measurements comming from gyro, accelerometer and magnetometer. The
% IMU does not have any bias.
%
% We reproduce the simulation with the parameters based on [1].
%
% [1] Manon Kok, Jeroen D. Hol and Thomas B. Schön (2017), "Using 
% Inertial Sensors for Position and Orientation Estimation", Foundations 
% and Trends® in Signal Processing: Vol. 11: No. 1-2, pp 1-153.

%% Initialization and Simulation
% Start by cleaning the workspace.
clear all;
close all;

% sequence time (s)
T = 100; 
% IMU frequency (Hz)
imu_freq = 100; 
% IMU standard-deviation noise (noise is isotropic)
imu_noise_std = [1/180*pi; % gyro (rad/s)
                0.1;       % accelerometer (m/s^2)
                0.1];      % magnetometer
            
% total number of timestamps
N = T*imu_freq;
% time between succesive timestamps (s)
dt = 1/imu_freq;

% simulate true trajectory and noised input
[true_state, omega] = attitude_simu_f(T, imu_freq, imu_noise_std);
% simulate accelerometer and magnetometer measurements
y = attitude_simu_h(true_state, T, imu_freq, imu_noise_std);         

%% UKF design
% We choose in this example to embed the robot state in $SO(3)$ with a left
% invariant error:
%
% * the function $\varphi(.,.)$ is the $S3(2)$ exponential map for
%  orientation where the state multiplies the uncertainty on the left
% * the function $\varphi^{-1}(.,.)$ is the $SO(3)$ logarithm for
% orientation 

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
ukf_state = true_state(1);
ukf_P = ukf_P0;

% variables for recording estimates
ukf_states = ukf_state;
ukf_Ps = zeros(N, 3, 3);
ukf_Ps(1, :, :) = ukf_P;

%% Filtering
for n = 2:N
    % propagation
    [ukf_state, ukf_P] = ukf_propagation(ukf_state, ukf_P, omega(n-1), ...
        ukf_f, dt, ukf_phi, ukf_phi_inv, ukf_cholQ, ukf_weights);
    % update
   [ukf_state, ukf_P] = ukf_update(ukf_state, ukf_P, y(:, n), ukf_h, ...
       ukf_phi, ukf_R, ukf_weights);
    % save estimates
    ukf_states(n) = ukf_state;
    ukf_Ps(n, :, :) = ukf_P;
end

%% Results
% We plot the orientation as function of time along with the orientation
% error
attitude_results_plot(ukf_states, ukf_Ps, true_state, omega, dt)

%%
% As UKF estimates the covariance of the error, we have plotted the 95%
% confident interval ($3\sigma$). We expect the error keeps behind this
% interval, and in this situation the filter covariance output matches
% especially well the error. 
%
% We see the true trajectory starts by a small stationnary step following
% by constantly turning around the gravity vector (only the yaw is
% increasing). As yaw is not observable with an accelerometer only, it is
% expected that yaw error would be stronger than roll or pitch errors.

%% Where to go next ?
% We have seen in this script how well works the UKF on parallelizable manifolds
% for estimating orientation from an IMU.
%
% You can now
% 
% * enter more in depth with the theory in [2]
% * adress the UKF for the same problem with diffenrent noise parameters,
%   add outliers in acceleration or magnetometer, or for others problems
% * benchmark the UKF with different function errors and compare it to the
%   extended Kalman filter in the benchmark folder.