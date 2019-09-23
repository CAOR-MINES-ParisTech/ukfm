%% 2D Robot Localization - Benchmark
% 
% Goals of this script:
% 
% * implement different UKFs on the 2D robot localization example.
% 
% * design the Extended Kalman Filter (EKF) and the Invariant Extended
%   Kalman Filter (IEKF) [BB17] for the given problem.
% 
% * compare the different algorithms with Monte-Carlo simulations.
% 
% _We assume the reader is already familiar with the considered problem
% described in the tutorial._
% 
% We previously designed an UKF with a standard uncertainty representation.
% An advantage of the versatility of the UKF is to speed up implementation,
% tests, and comparision UKF with different uncertainty representations.
% Indeed, for the given problem, three different UKFs emerge, defined
% respectively as:
% 
% # The state is embedded in $SO(2) \times R^2$, as in the example, where:
% 
%    * the retraction $\varphi(.,.)$ is the $SO(2)$ exponential map
%      for orientation and the standard vector addition for robot position.
% 
%    * the inverse retraction $\varphi^{-1}(.,.)$ is the $SO(2)$
%      logarithm for orientation and the standard vector subtraction for
%      position.
% 
% # The state is embedded in $SE(2)$ with left multiplication, i.e.
% 
%    * the retraction $\varphi(.,.)$ is the $SE(2)$ exponential,
%      where the state multiplies on the left the retraction
%      $\xi$.
% 
%    * the inverse retraction $\varphi^{-1}(.,.)$ is the $SE(2)$
%      logarithm.
% 
%    * this left UKF on $SE(2)$ corresponds to the Invariant Extended
%      Kalman Filter (IEKF) recommended in [BB17]. We
%      have theoretical reason to choose this retraction.
% 
% # The state is embedded in $SE(2)$ with right multiplication, i.e.
% 
%    * the retraction $\varphi(.,.)$ is the $SE(2)$ exponential,
%      where the state multiplies on the right the retraction
%      $\xi$.
% 
%    * the inverse retraction $\varphi^{-1}(.,.)$ is the $SE(2)$
%      logarithm.
% 
% We tests the different filters with the same noise parameter setting and
% on simulation with strong initial heading error. We will see how perform
% the filters compared to extended Kalman filters.

%% Initialization
% Start by cleaning the workspace.
clear all;
close all;

%% 
% We compare the different filters on a large number of Monte-Carlo runs.

% Monte-Carlo runs
N_mc = 10;

%% Simulation Settion
% sequence time (s)
T = 40; 
% odometry frequency (Hz)
odo_freq = 100; 
% odometry noise standard deviation
odo_noise_std = [0.01; % speed (v/m)
                 0.01; % speed (v/m)
                 1/180*pi]; % angular speed (rad/s)
% GPS frequency (Hz)
gps_freq = 1;
% GPS noise standard deviation (m)
gps_noise_std = 1;

% total number of timestamps
N = T*odo_freq;
% time between succesive timestamps (s)
dt = 1/odo_freq;
% radius of the circle trajectory (m)
radius = 5;

%% Filter Design
% Additionally to the three UKFs, we compare them to an Extended Kalman
% FIlter (EKF) and an Invariant EKF (IEKF). The EKF has the same
% uncertainty representation as the UKF with the retraction on $SO(2)
% \times R^2$, whereas the IEKF has the same uncertainty
% representation as the UKF with the left retraction on $SE(2)$.

% propagation noise matrix
Q = diag(odo_noise_std.^2);

% measurement noise matrix
R = gps_noise_std^2 * eye(2);

% initial error matrix 
P0 = zeros(3, 3);
init_heading_error_std = 45/180*pi;
% we take into account initial heading error
P0(1, 1) = (init_heading_error_std)^2;

% sigma point parameter
ukf_alpha = [1e-3 1e-3 1e-3];

% define the UKF functions
ukf_f = @localization_f;
ukf_h = @localization_h;

ukf_phi = @localization_phi;
ukf_left_phi = @localization_left_phi;
ukf_right_phi = @localization_right_phi;

ukf_phi_inv = @localization_phi_inv;
ukf_left_phi_inv = @localization_left_phi_inv;
ukf_right_phi_inv = @localization_right_phi_inv;

ukf_weights = ukf_set_weight(length(P0), length(Q), ukf_alpha);
ukf_cholQ = chol(Q);

%%
% We set error variables before launching Monte-Carlo simulations. As we
% have five similar methods, the code is redundant.

ukf_err = zeros(2, N_mc);
ukf_left_err = zeros(size(ukf_err));
ukf_right_err = zeros(size(ukf_err));
iekf_err = zeros(size(ukf_err));
ekf_err = zeros(size(ukf_err));

%%
% We record Normalized Estimation Error Squared (NEES) for consistency
% evaluation (see Results).

ukf_nees = zeros(N_mc, N, 2);
left_ukf_nees = zeros(size(ukf_nees));
right_ukf_nees = zeros(size(ukf_nees));
iekf_nees = zeros(size(ukf_nees));
ekf_nees = zeros(size(ukf_nees));

%% Monte-Carlo runs
% We run the Monte-Carlo through a for loop.

for n_mc = 1:N_mc
    disp("Monte-Carlo iteration(s): " + num2str(n_mc) + "/" + num2str(N_mc));
    % simulation true trajectory
    [states, omegas] = localization_simu_f(T, odo_freq, ...
        odo_noise_std, radius);
    % simulate measurement
    [ys, one_hot_ys] = localization_simu_h(states, T, odo_freq, ...
    gps_freq, gps_noise_std);

    % initialize filter with innacurate state
    ukf_state = states(1);
    % we sample an initial heading error from the true distribution. This
    % is the correct manner to compare the filters but requires more
    % Monte-Carlo samples than a static values
    init_heading_error = init_heading_error_std*randn(1); 
    ukf_state.Rot = states(1).Rot * so2_exp(init_heading_error);
    ukf_left_state = ukf_state(1);
    ukf_right_state = ukf_state(1);
    ekf_state = ukf_state(1);
    iekf_state = ukf_state(1);
    
    ukf_P = P0;
    ukf_left_P = P0;
    ukf_right_P = P0;
    ekf_P = P0;
    iekf_P = P0;
    % this is correct for your situation, see [BB17]
    
    % variables for recording estimates of the Monte-Carlo run
    ukf_states = ukf_state;
    ukf_left_states = ukf_left_state;
    ukf_right_states = ukf_right_state;
    iekf_states = iekf_state;
    ekf_states = ekf_state;
    
    ukf_Ps = zeros(N, 3, 3);
    ukf_left_Ps = zeros(size(ukf_Ps));
    ukf_right_Ps = zeros(size(ukf_Ps));
    ekf_Ps = zeros(size(ukf_Ps));
    iekf_Ps = zeros(size(ukf_Ps));
    
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
        [ukf_state, ukf_P] = ukf_propagation(ukf_state, ukf_P, omegas(n-1), ...
            ukf_f, dt, ukf_phi, ukf_phi_inv, ukf_cholQ, ukf_weights);
        [ukf_left_state, ukf_left_P] = ukf_propagation(ukf_left_state, ...
            ukf_left_P, omegas(n-1), ukf_f, dt, ukf_left_phi, ...
            ukf_left_phi_inv, ukf_cholQ, ukf_weights);
        [ukf_right_state, ukf_right_P] = ukf_propagation(...
            ukf_right_state, ukf_right_P, omegas(n-1), ...
            ukf_f, dt, ukf_right_phi, ukf_right_phi_inv, ukf_cholQ, ...
            ukf_weights);
        [ekf_state, ekf_P] = localization_ekf_propagation(ekf_state, ...
            ekf_P, omegas(n-1), dt, Q);
        [iekf_state, iekf_P] = localization_iekf_propagation(...
            iekf_state, iekf_P, omegas(n-1), dt, Q);
   
        % update only if a measurement is received
        if one_hot_ys(n) == 1
           [ukf_state, ukf_P] = ukf_update(ukf_state, ukf_P, ...
               ys(:, k), ukf_h, ukf_phi, R, ukf_weights);
           [ukf_left_state, ukf_left_P] = ukf_update(ukf_left_state, ...
               ukf_left_P, ys(:, k), ukf_h, ukf_left_phi, R, ukf_weights);           
           [ukf_right_state, ukf_right_P] = ukf_update(ukf_right_state, ...
               ukf_right_P, ys(:, k), ukf_h, ukf_right_phi, ...
               R, ukf_weights);
           [ekf_state, ekf_P] = localization_ekf_update(ekf_state, ...
               ekf_P, ys(:, k), R);
           [iekf_state, iekf_P] = localization_iekf_update(iekf_state, ...
               iekf_P, ys(:, k), R);
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
    ukf_left_err(:, n_mc) = localization_error(states, ukf_left_states);
    ukf_right_err(:, n_mc) = localization_error(states, ukf_right_states);
    ukf_err(:, n_mc) = localization_error(states, ukf_states);
    iekf_err(:, n_mc) = localization_error(states, iekf_states);
    ekf_err(:, n_mc) = localization_error(states, ekf_states);
end

%% Results
% Simulations are ended, we can know compare the algorithms. Let us
% visualize the results for the last run. As simulations have random
% process, they just give us an indication but not a proof of filter
% performances.
%
% Very interesting is to compute results averaged over all the Monte-Carlo.
% Let us compute the Root Mean Squared Error (RMSE)for each method both for
% the orientation and the position.

benchmark_localization_helper;

% A consistency metric is the Normalized Estimation Error Squared (NEES).
% Classical criteria used to evaluate the performance of an estimation
% method, like the RMSE, do not inform about consistency as they do not
% take into account the uncertainty returned by the filter. This point is
% addressed by the NEES, which computes the average squared value of the
% error, normalized by the covariance matrix of the filter. The case NEES>1
% reveals an inconsistency issue: the actual uncertainty is higher than the
% computed uncertainty.

model.nees_print(ukf_nees, left_ukf_nees, right_ukf_nees, ...
    iekf_nees, ekf_nees)

%%
% As the filters are initialized with perfect position and zero covariance
% w.r.t. position, we compute NEES only after 20 s for avoiding numerical
% issues (during the first secondes of the trajectory the covariance matrix
% $\mathbf{P}_n$ is very low so inverting it leads to insignificantly high
% numbers). Results are clear, IEKF and $SE(2)$ are the more consistent.
% For the considered example, it seems that the UKFs are slightly less
% optimistic that their EKF counterparts.

%%
% *Which filter is the best ?* We expected it is the left UKF as it takes
% advantage of both theory of IEKF uncertainty representation and better
% non-linearity noise incorporation of UKF compared to (I)EKF. Depending on
% which source of error (sensor noise, initial condition) is the most
% important, it can lead to different results. Indeed, in this setting,
% *left IEKF*, *right UKF** and *IEKF* filters obtain similar accurate
% results, that clearly outperform $SO(2) \times R^2$ UKF, and one
% could expect encounter situations where the left UKF outperforms the
% right UKF.
%
% _We have set all the filters with the same "true" noise covariance
% parameters. However, both EKF and UKF based algorithms may better deal ,
% with non-linearity  by e.g. inflated propagation noise covariance._
%

%% Conclusion
% This script compares different algorithms for 2D robot localization. Two
% groups of filters emerge: the $SO(2) \times R^2` UKF and the EKF
% represent the first group; and the left $SE(2)$ UKF, the right $SE(2)$
% UKF and the IEKF constitute the second group. For the considered set of
% parameters, it is evident that embedded the state in $SE(2)$ is
% advantageous for state estimation. Choosing then between left UKF, right
% UKF or IEKF has negligible effect (with the considered simulation
% setting).
%
% You can now:
%
% * compare the filters in different scenarios. Indeed, UKF and their
%   (I)EKF counterparts may obtain different results when noise is e.g. 
%   inflated or with different initial conditions or trajectory.
%
% * testing the filters in a slightly different model (e.g. with
%   orientation measurement), which is straightforward for the UKFs.
%
% * address the problem of 3D attitude estimations, see the Examples
%   section.
