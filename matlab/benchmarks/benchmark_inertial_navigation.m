%% Navigation on Flat Earth - Benchmark
%
% Goals of this script:
% 
% - implement different UKFs on the navigation on flat Earth example.
% 
% - design the Extended Kalman Filter (EKF) and the Invariant Extended.
%   Kalman Filter (IEKF) [BB17] for the given problem.
% 
% - compare the different algorithms with Monte-Carlo simulations.
% 
% _We assume the reader is already familiar with the considered
% problem described in the related example._
% 
% This script searches to estimate the 3D attitude, the velocity, and the
% position of a rigid body in space from inertial sensors and relative
% observations of points having known locations. For the given problem, three
% different UKFs emerge, defined respectively as:
% 
% # The state is embedded  in $SO(3) \times R^6$, i.e.
% 
%    - the retraction $\varphi(.,.)$ is the $SO(3)$ exponential for
%      orientation, and the standard vector addition for robot velocity and
%      position.
% 
%    - the inverse retraction $\varphi^{-1}(.,.)$ is the $SO(3)$
%      logarithm for orientation and the standard vector subtraction for velocity
%      and position.
% 
%    - it is the $SO(3) \times R^6$ UKF that was defined in the
%      example.
% 
% # The state is embedded in $SE_2(3)$ with left multiplication, i.e.
% 
%    - the retraction $\varphi(.,.)$ is the $SE_2(3)$ exponential,
%      where the state multiplies on the left the error
%      $\xi$.
% 
%    - the inverse retraction $\varphi^{-1}(.,.)$ is the $SE_2(3)$
%      logarithm.
%    
%    - we call it the left UKF (rather than $SE_2(3)$ left UKF).
% 
% # The state is embedded in $SE_2(3)$ with right multiplication, i.e.
% 
%    - the retraction $\varphi(.,.)$ is the $SE_2(3)$ exponential,
%      where state multiplies on the right the error
%      $\xi$.
% 
%    - the inverse retraction $\varphi^{-1}(.,.)$ is the $SE_2(3)$
%      logarithm.
% 
%    - this right UKF corresponds to the Invariant Extended Kalman Filter (IEKF)
%      recommended in [BB17].
% 
% 
%     _The exponential and logarithm of $SE_2(3)$ are quickly derived from 
%     the $SE(3)` exponential and logarithm, see Lie Groups documentation._

%% Initialization
% Start by cleaning the workspace.
clear all;
close all;

%% Simulation Setting
%We compare the different filters on a large number of Monte-Carlo runs.

% Monte-Carlo runs
N_mc = 1;

% sequence time (s)
T = 40;
% IMU frequency (Hz)
imu_freq = 100;
% observation frequency (Hz)
obs_freq = 1;
% IMU standard-deviation noise (noise is isotropic)
imu_noise_std = [0.01; % gyro (rad/s), ~ 0.6 deg/s
                 0.01]; % accelerometer (m/s^2)
% number of observed landmark
N_ldk = 3;
% observation noise standard deviation (m)
obs_noise_std = 0.1;

% total number of timestamps
N = T*imu_freq;
% time between succesive timestamps (s)
dt = 1/imu_freq;

%%
% The vehicle drives a 10-meter diameter circle
% in 30 seconds and observes three features  every second while receiving
% high-frequency inertial measurements (100 Hz).

%% Filter Design
% Additionally to the three UKFs, we compare them to an EKF and an IEKF. The EKF
% has the same uncertainty representation as the UKF with $SO(3) \times
% R^6$ uncertainty representation, whereas the IEKF has the same
% uncertainty representation as the UKF with right $SE_2(3)$ retraction.
% As we have five similar methods, the code is redundant.

% propagation noise matrix
Q = blkdiag(imu_noise_std(1)^2*eye(3), imu_noise_std(2)^2*eye(3));
% measurement noise matrix
R = obs_noise_std.^2 * eye(3*N_ldk);
% initial error matrix such that the state is not perfectly initialized
init_rot_std = 15/sqrt(3)*pi/180;
init_p_std = 1/sqrt(3);
P0 = blkdiag(init_rot_std^2*eye(3), zeros(3, 3), init_p_std^2 * eye(3));

% sigma point parameter
ukf_alpha = [1e-1 1e-1 1e-1];

% define the UKF functions
ukf_f = @inertial_navigation_f;
ukf_h = @inertial_navigation_h;

ukf_phi = @inertial_navigation_phi;
ukf_left_phi = @inertial_navigation_left_phi;
ukf_right_phi = @inertial_navigation_right_phi;

ukf_phi_inv = @inertial_navigation_phi_inv;
ukf_left_phi_inv = @inertial_navigation_left_phi_inv;
ukf_right_phi_inv = @inertial_navigation_right_phi_inv;

ukf_weights = ukf_set_weight(length(P0), length(Q), ukf_alpha);
ukf_cholQ = chol(Q);

%%
% We set error variables before launching Monte-Carlo simulations
ukf_err = zeros(2, N_mc);
ukf_left_err = zeros(size(ukf_err));
ukf_right_err = zeros(size(ukf_err));
iekf_err = zeros(size(ukf_err));
ekf_err = zeros(size(ukf_err));

ukf_nees = zeros(N_mc, N, 2);
left_ukf_nees = zeros(size(ukf_nees));
right_ukf_nees = zeros(size(ukf_nees));
iekf_nees = zeros(size(ukf_nees));
ekf_nees = zeros(size(ukf_nees));

%% Monte-Carlo runs
% We run the Monte-Carlo through a for loop.

for n_mc = 1:N_mc
    disp("Monte-Carlo iteration(s): " + num2str(n_mc) + "/" + num2str(N_mc));
   % simulate true trajectory and noised input
    [true_state, omega] = inertial_navigation_simu_f(T, imu_freq, ...
        imu_noise_std);
    % simulate amers measurements
    [y, one_hot_y] = inertial_navigation_simu_h(true_state, T, imu_freq, ...
        obs_freq, obs_noise_std);

    % initialize filter with innacurate state
    ukf_state = true_state(1);
    % we sample an initial error from the true distribution. This
    % is the correct manner to compare the filters but requires more
    % Monte-Carlo samples than a static values
    init_rot_err = so3_exp(init_rot_std*ones(3, 1));
    init_p_err = init_p_std*ones(3, 1);
    ukf_state.Rot = init_rot_err * ukf_state.Rot;
    ukf_state.p = ukf_state.p + init_p_err;
    
    ukf_left_state = ukf_state(1);
    ukf_right_state = ukf_state(1);
    ekf_state = ukf_state(1);
    iekf_state = ukf_state(1);
    
    ukf_P = P0;
    ukf_left_P = P0;
    ukf_right_P = P0;
    ekf_P = P0;
    iekf_P = P0;
    
    % IEKF and right UKF covariance need to be turned [BB17]
    J = eye(9);
    J(7:9,1:3) = so3_wedge(iekf_state.p);
    ukf_right_P = J*ukf_right_P*J';
    iekf_P = J*iekf_P*J';
    
    % variables for recording estimates of the Monte-Carlo run
    ukf_states = ukf_state;
    ukf_left_states = ukf_left_state;
    ukf_right_states = ukf_right_state;
    iekf_states = iekf_state;
    ekf_states = ekf_state;
    
    ukf_Ps = zeros(N, 9, 9);
    ukf_left_Ps = zeros(N, 9, 9);
    ukf_right_Ps = zeros(N, 9, 9);
    ekf_Ps = zeros(N, 9, 9);
    iekf_Ps = zeros(N, 9, 9);
    
    ukf_Ps(1, :, :) = ukf_P;
    ukf_left_Ps(1, :, :) = ukf_left_P;
    ukf_right_Ps(1, :, :) = ukf_right_P;
    ekf_Ps(1, :, :) = ekf_P;
    iekf_Ps(1, :, :) = iekf_P;
    
    % measurement iteration number
    k = 2;

    % filtering loop
    for n = 2:N
        % propagation
        [ukf_state, ukf_P] = ukf_propagation(ukf_state, ukf_P, omega(n-1), ...
            ukf_f, dt, ukf_phi, ukf_phi_inv, ukf_cholQ, ukf_weights);
        [ukf_left_state, ukf_left_P] = ukf_propagation(ukf_left_state, ...
            ukf_left_P, omega(n-1), ukf_f, dt, ukf_left_phi, ...
            ukf_left_phi_inv, ukf_cholQ, ukf_weights);
        [ukf_right_state, ukf_right_P] = ukf_propagation(...
            ukf_right_state, ukf_right_P, omega(n-1), ...
            ukf_f, dt, ukf_right_phi, ukf_right_phi_inv, ukf_cholQ, ...
            ukf_weights);
        [ekf_state, ekf_P] = inertial_navigation_ekf_propagation(ekf_state, ...
            ekf_P, omega(n-1), dt, Q);
        [iekf_state, iekf_P] = inertial_navigation_iekf_propagation(...
            iekf_state, iekf_P, omega(n-1), dt, Q);
   
        % update only if a measurement is received
        if one_hot_y(n) == 1
           [ukf_state, ukf_P] = ukf_update(ukf_state, ukf_P, ...
               y(:, k), ukf_h, ukf_phi, R, ukf_weights);
           [ukf_left_state, ukf_left_P] = ukf_update(ukf_left_state, ...
               ukf_left_P, y(:, k), ukf_h, ukf_left_phi, R, ukf_weights);           
           [ukf_right_state, ukf_right_P] = ukf_update(ukf_right_state, ...
               ukf_right_P, y(:, k), ukf_h, ukf_right_phi, ...
               R, ukf_weights);
           [ekf_state, ekf_P] = inertial_navigation_ekf_update(ekf_state, ...
               ekf_P, y(:, k), R);
           [iekf_state, iekf_P] = inertial_navigation_iekf_update(iekf_state, ...
               iekf_P, y(:, k), R);
            k = k + 1;
        end
        % save estimates
        ukf_states(n) = ukf_state;
        ukf_left_states(n) = ukf_left_state;
        ukf_right_states(n) = ukf_right_state;
        ekf_states(n) = ekf_state;
        iekf_states(n) = iekf_state;
        
        ukf_Ps(n, :, :) = ukf_P;
        ukf_left_Ps(n, :, :) = ukf_left_P;
        ukf_right_Ps(n, :, :) = ukf_right_P;
        ekf_Ps(n, :, :) = ekf_P;
        iekf_Ps(n, :, :) = iekf_P;
        
    end
    % record errors
    ukf_left_err(:, n_mc) = inertial_navigation_error(true_state, ukf_left_states);
    ukf_right_err(:, n_mc) = inertial_navigation_error(true_state, ukf_right_states);
    ukf_err(:, n_mc) = inertial_navigation_error(true_state, ukf_states);
    iekf_err(:, n_mc) = inertial_navigation_error(true_state, iekf_states);
    ekf_err(:, n_mc) = inertial_navigation_error(true_state, ekf_states);
end

%% Results
% We first visualize the trajectory results for the last run, where the vehicle
% start in the above center of the plot. As simulations have random process,
% they just give us an indication but not a proof of filter performances. We
% then plot the orientation and position errors averaged over Monte-Carlo.

benchmark_inertial_navigation_helper;

%%
% The novel retraction on $SE_2(3)$ resolves the problem encountered by
% the $SO(3) \times R^6$ UKF and particularly the EKF.
%
% We confirm these plots by computing statistical results averaged over all the
% Monte-Carlo. We compute the Root Mean Squared Error (RMSE) for each method
% both for the orientation and the position.

model.benchmark_print(ukf_err, left_ukf_err, right_ukf_err, iekf_err, ekf_err)

%%
% For the considered Monte-Carlo, we have first observed that EKF is not working
% very well. Then, it happens that IEKF, left UKF and right UKF are the best
% in the first instants of the trajectory, that is confirmed with RMSE.

%%
% We now compare the filters in term of consistency (NEES).

model.nees_print(ukf_nees, left_ukf_nees, right_ukf_nees, iekf_nees, ekf_nees)

%%
% The $SO(3) \times R^6$ UKF and EKF are too optimistic. Left
% UKF, right UKF and IEKF obtain similar NEES and are really consistent filters
% after some secondes. A (tricky) way for solving this issue just consists in
% inflating initial covariance $\mathbf{P}_0$

%%
% *Which filter is the best ?* For the considered problem, *left
% $SE_2(3)$ UKF*, *right $SE_2(3)$ UKF*, and the IEKF obtain
% roughly similar accurate results, where the $SE_2(3)$ UKFs seem better,
% and that clearly outperform standard UKF. One could expect encounter
% situations, typically long term experiments, where the right UKF outperforms
% the left UKF.


%% Conclusion
% This script compares different algorithms on the inertial navigation on flat
% Earth example. The left UKF and the right UKF, build on $SE_2(3)$
% retraction, outperform the EKF and seem slightly better than the IEKF.
%
% You can now:
% 
% - confirm (or infirm) the obtained results on massive Monte-Carlo
%   simulations. Another relevant comparision consists in testing the filters
%   when propagation noise is very low (standard deviation of $10^{-4}$),
%   as suggested in [BB17]. 
%
% - address the problem of 2D SLAM, where the UKF is, among other, leveraged to
%   augment the state when a novel landmark is observed.
