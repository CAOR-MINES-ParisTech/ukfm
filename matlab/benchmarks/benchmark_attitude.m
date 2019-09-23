%% 3D Attitude Estimation - Benchmark
% 
% Goals of this script:
% 
% * implement two different UKFs on the 3D attitude estimation example.
% 
% * design the Extended Kalman Filter (EKF) for the given problem.
% 
% * compare the different algorithms with Monte-Carlo simulations.
% 
% _We assume the reader is already familiar with the considered problem
% described in the related example._
% 
% For the given problem, two different UKFs emerge, defined respectively
% as:
% 
% # The state is embedded in $SO(3)$ with left multiplication, i.e.
% 
% * the retraction $\varphi(.,.)$ is the $SO(3)$ exponential
%   where uncertainty is multiplied on the left by the state.
% 
% * the inverse retraction $\varphi^{-1}(.,.)$ is the $SO(3)$
%   logarithm.
% 
% # The state is embedded in $SO(3)$ with right multiplication, i.e.
% 
% * the retraction $\varphi(.,.)$ is the $SO(3)$ exponential
% where
%   uncertainty is multiplied on the right by the state.
% 
% * the inverse retraction $\varphi^{-1}(.,.)$ is the $SO(3)$
%   logarithm.
% 
% We tests the different with the same noise parameter setting and on
% simulation with moderate initial heading error. We will see how perform
% the filters compared to the extended Kalman filter.

%% Initialization
% Start by cleaning the workspace.
clear all;
close all;

%% Simulation Setting
% We compare the different filters on a large number of Monte-Carlo runs.

% Monte-Carlo runs
N_mc = 1;

% sequence time (s)
T = 100; 
% IMU frequency (Hz)
imu_freq = 100; 
% IMU standard-deviation noise (noise is isotropic)
imu_noise_std = [5/180*pi; % gyro (rad/s)
                0.4;       % accelerometer (m/s^2)
                0.3];      % magnetometer
            
% total number of timestamps
N = T*imu_freq;
% time between succesive timestamps (s)
dt = 1/imu_freq;

% simulate true trajectory and noised input
[true_state, omega] = attitude_simu_f(T, imu_freq, imu_noise_std);
% simulate accelerometer and magnetometer measurements
y = attitude_simu_h(true_state, T, imu_freq, imu_noise_std); 

%% Filter Design
% Additionnaly to the UKFs, we compare them to an EKF. The EKF has the same
% uncertainty representation as the UKF with right uncertainty representation.

% propagation noise matrix
Q = imu_noise_std(1).^2*eye(3);
% measurement noise matrix
R = blkdiag(imu_noise_std(2).^2*eye(3), imu_noise_std(3).^2*eye(3));
% initial error matrix
P0 = (10/180*pi)^2 * eye(3); % The state is perfectly initialized

% sigma point parameters
ukf_alpha = [1e-3, 1e-3, 1e-3];

% asses UKF function
ukf_f = @attitude_f;
ukf_h = @attitude_h;
ukf_left_phi = @attitude_phi;
ukf_left_phi_inv = @attitude_phi_inv;
ukf_right_phi = @attitude_right_phi;
ukf_right_phi_inv = @attitude_right_phi_inv;
ukf_weights = ukf_set_weight(length(P0), length(R), ukf_alpha);
ukf_cholQ = chol(Q);


%%
% We set error variables before launching Monte-Carlo simulations
ukf_left_err = zeros(2, N_mc);
ukf_right_err = zeros(size(ukf_left_err));
ekf_err = zeros(size(ukf_left_err));

%% Monte-Carlo runs
% We run the Monte-Carlo through a for loop.

for k = 1:N_mc
    disp("Monte-Carlo iteration(s): " + num2str(k) + "/" + num2str(N_mc));
    % simulate true trajectory and noised input
    [true_state, omega] = attitude_simu_f(T, imu_freq, imu_noise_std);
    % simulate accelerometer and magnetometer measurements
    y = attitude_simu_h(true_state, T, imu_freq, imu_noise_std);    

    % initialize filter with true state
    ekf_state = true_state(1);
    ukf_left_state = ekf_state(1);
    ukf_right_state = ekf_state(1);

    state0
    
    ukf_left_P = state0.Rot*P0*state0.Rot';
    ukf_right_P = P0;
    ekf_P = P0;
    
    % variables for recording estimates of the Monte-Carlo run
    ukf_left_states = ukf_left_state;
    ukf_right_states = ukf_right_state;
    ekf_states = ekf_state;
    
    ukf_left_Ps = zeros(N, 3, 3);
    ukf_right_Ps = zeros(N, 3, 3);
    ekf_Ps = zeros(N, 3, 3);
    
    ukf_left_Ps(1, :, :) = ukf_left_P;
    ukf_right_Ps(1, :, :) = ukf_right_P;
    ekf_Ps(1, :, :) = ekf_P;

    % filtering loop
    for n = 2:N
        % propagation;
        [ukf_left_state, ukf_left_P] = ukf_propagation(ukf_left_state, ...
            ukf_left_P, omega(n-1), ukf_f, dt, ukf_left_phi, ...
            ukf_left_phi_inv, ukf_cholQ, ukf_weights);
        [ukf_right_state, ukf_right_P] = ukf_propagation(...
            ukf_right_state, ukf_right_P, omega(n-1), ...
            ukf_f, dt, ukf_right_phi, ukf_right_phi_inv, ukf_cholQ, ...
            ukf_weights);
        [ekf_state, ekf_P] = attitude_ekf_propagation(ekf_state, ...
            ekf_P, omega(n-1), dt, Q);
   
        % update
       [ukf_left_state, ukf_left_P] = ukf_update(ukf_left_state, ...
           ukf_left_P, y(:, n), ukf_h, ukf_left_phi, R, ukf_weights);           
       [ukf_right_state, ukf_right_P] = ukf_update(ukf_right_state, ...
           ukf_right_P, y(:, n), ukf_h, ukf_right_phi, ...
           R, ukf_weights);
       [ekf_state, ekf_P] = attitude_ekf_update(ekf_state, ...
           ekf_P, y(:, n), R);

        % save estimates
        ukf_left_states(n) = ukf_left_state;
        ukf_right_states(n) = ukf_right_state;
        ekf_states(n) = ekf_state;
        
        ukf_left_Ps(n, :, :) = ukf_left_P;
        ukf_right_Ps(n, :, :) = ukf_right_P;
        ekf_Ps(n, :, :) = ekf_P;
        
    end
    % record errors
    ukf_left_err(:, k) = localization_error(true_state, ukf_left_states);
    ukf_right_err(:, k) = localization_error(true_state, ukf_right_states);
    ekf_err(:, k) = localization_error(true_state, ekf_states);
end

%% Results
% We compare the algorithms by first visualizing the results averaged over
% Monte-Carlo sequences.

benchmark_attitude_helper;

%%
% We compute the Root Mean Squared Error (RMSE) averaged over all the
% Monte-Carlo. All the curves have the same shape. Filters obtain the same
% performances.

%%
% We finally compare the filters in term of consistency (Normalized Estimation
% Error Squared, NEES), as in the localization benchmark.

disp(' ')
disp('Root Mean Square Error w.r.t. orientation (deg)');
disp("    -left UKF    : " + ukf_left_err_rot);
disp("    -right UKF   : " + ukf_right_err_rot);
disp("    -EKF         : " + ekf_err_rot);
%%
% All the filters obtain the same NEES and are consistent.

%%
% *Which filter is the best ?* For the considered problem, *left UKF*,
% *right UKF*, and *EKF* obtain the same performances. This is expected as
% when the state consists of an orientation only, left and right UKF are the
% same. The EKF obtains similar results as it is also based on a retraction
% build on $SO(3)$ (not with Euler angles). This does not hold when the
% state include orientation, velocity and position.

%% Conclusion
% This script compares two UKFs and one EKF for the problem of attitude
% estimation. All the filters obtain similar performances as the state involves
% only the orientation of  the platform.
% 
% You can now:
%
% * compare the filter in different noise setting to see if filters still get
%   the same performances.
%  
% * address the problem of 3D inertial navigation, where the state is defined as
%   the oriention of the vehicle along with its velocity and its position, see
%   the Examples section.

