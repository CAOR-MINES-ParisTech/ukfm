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
%_We assume the reader is already familiar with the considered problem
%described
% in the tutorial._
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
%    * this UKF on $SE(2)$ corresponds to the Invariant Extended Kalman
%      Filter (IEKF) recommended in [BB17]. We have
%      theoretical reason to choose this retraction.
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
N_mc = 1;

%% Simulation Setting
% We set the simulation as in [BB17], section IV. The robot drives along a
% 10 m diameter circle for 40 seconds with high rate odometer measurements
% (100 Hz) and low rate GPS measurements (1 Hz). The vehicle gets moderate
% angular velocity uncertainty and highly precise linear velocity. The
% initial values of the heading error is very strong, *45° standard
% deviation*, while the initial position is known.

% sequence time (s)
T = 2500; 
% odometry frequency (Hz)
odo_freq = 1; 
% true speed of robot (m/s)
v = 0.25;
% true angular velocity (rad/s)
gyro = 1.5/180*pi;
% odometry noise standard deviation (see [1])
odo_noise_std = [0.05*v/sqrt(2);    % speed (v/m)
                 0.05*v*sqrt(2)*2]; % angular speed (rad/s)
% observation noise standard deviation (m)
obs_noise_std = 0.1;

% total number of timestamps
N = T*odo_freq;
% time between succesive timestamps (s)
dt = 1/odo_freq;

%% Filter Design
% Additionally to the three UKFs, we compare them to an EKF and an IEKF.
% The EKF has the same uncertainty representation as the UKF with $SO(2)
% \times R^{2(1+L)}$ uncertainty representation, whereas the IEKF has the
% same uncertainty representation as the UKF with right $SE_{1+L}(2)$
% retraction.

% propagation noise matrix
Q = diag(odo_noise_std.^2);

% measurement noise matrix
R = obs_noise_std^2 * eye(2);

% initial error matrix 
P0 = zeros(3, 3);

% sigma point parameter
ukf_alpha = [1e-3 1e-3 1e-3];

% define the UKF functions
ukf_f = @slam2d_f;
ukf_h = @slam2d_h;

% reduced error used during propagation
ukf_red_phi = @slam2d_red_phi;
ukf_red_phi_inv = @slam2d_red_phi_inv;
% error used during update
ukf_up_phi = @slam2d_phi;
% we do not need phi_inv fonction for update
% error used for augmenting state
ukf_z = @slam2d_z;
ukf_z_aug = @slam2d_z_aug;
ukf_aug_phi = @slam2d_aug_phi;

ukf_left_red_phi = @slam2d_left_red_phi;
ukf_left_red_phi_inv = @slam2d_left_red_phi_inv;
ukf_left_up_phi = @slam2d_left_phi;
ukf_left_aug_phi = @slam2d_left_aug_phi;

ukf_right_red_phi = @slam2d_right_red_phi;
ukf_right_red_phi_inv = @slam2d_right_red_phi_inv;
ukf_right_up_phi = @slam2d_right_phi;
ukf_right_aug_phi = @slam2d_right_aug_phi;


% reduced weights during propagation
ukf_red_weights = ukf_set_weight(3, 2, ukf_alpha);
ukf_red_idxs = 1:3; % indices corresponding to the robot state in P

% weights during update
ukf_weights = ukf_set_weight(5, 2, ukf_alpha);
ukf_aug_weights = ukf_set_weight(3, 2, ukf_alpha);
ukf_aug_idxs = 1:3; % indices corresponding to the robot state in P
ukf_cholQ = chol(Q);

%%
% We set error variables before launching Monte-Carlo simulations
ukf_err = zeros(2, N_mc);
ukf_left_err = zeros(size(ukf_err));
ukf_right_err = zeros(size(ukf_err));
iekf_err = zeros(size(ukf_err));
ekf_err = zeros(size(ukf_err));

%% Monte-Carlo runs
% We run the Monte-Carlo through a for loop.

for n_mc = 1:N_mc
    disp("Monte-Carlo iteration(s): " + num2str(n_mc) + "/" + ...
        num2str(N_mc));

    % simulate true trajectory and noisy input
    [states, omegas, ldks] = slam2d_simu_f(T, odo_freq, odo_noise_std, ...
        v, gyro);
    % number of landmarks
    N_ldk = size(ldks, 2);
    % simulate landmark measurements
    ys = slam2d_simu_h(states, obs_noise_std, N_ldk, ldks);

    % initialize filter with innacurate state
    ukf_state = states(1);
    ukf_state.p_l = [];
    ukf_lmk = [];
    ekf_lmk = [];
    iekf_lmk = [];
    ukf_left_state = ukf_state(1);
    ukf_right_state = ukf_state(1);
    ekf_state = ukf_state(1);
    iekf_state = ukf_state(1);
    
    ukf_P = P0;
    ukf_left_P = P0;
    ukf_right_P = P0;
    ekf_P = P0;
    iekf_P = P0;
    
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

    % filtering loop
    for n = 2:N
        % propagation
        ukf_red_idxs = 1:3;
        [ukf_state, ukf_P] = ukf_jacobian_propagation(ukf_state, ukf_P, ...
            omegas(n-1), ukf_f, dt, ukf_red_phi, ukf_red_phi_inv, ukf_cholQ, ...
            ukf_red_weights, ukf_red_idxs);
        ukf_red_idxs = 1:length(ukf_left_P);
        [ukf_left_state, ukf_left_P] = ukf_jacobian_propagation(ukf_left_state, ukf_left_P, ...
            omegas(n-1), ukf_f, dt, ukf_left_red_phi, ukf_left_red_phi_inv, ukf_cholQ, ...
            ukf_red_weights, ukf_red_idxs);
        ukf_red_idxs = 1:length(ukf_left_P);
        [ukf_right_state, ukf_right_P] = ukf_jacobian_propagation(ukf_right_state, ukf_right_P, ...
            omegas(n-1), ukf_f, dt, ukf_right_red_phi, ukf_right_red_phi_inv, ukf_cholQ, ...
            ukf_red_weights, ukf_red_idxs);

        
        [iekf_state, iekf_P] = slam2d_iekf_propagation(...
            iekf_state, iekf_P, omegas(n-1), dt, Q);
        [ekf_state, ekf_P] = slam2d_ekf_propagation(ekf_state, ...
            ekf_P, omegas(n-1), dt, Q);
        
        y = ys(:, :, n);
        N_y = length(find(y(3, :) > 0));

        % set observalibity matrice and residual
        H = zeros(0, length(ukf_P));
        res = zeros(0);
        
        H_left = zeros(0, length(ukf_left_P));
        res_left = zeros(0);
        
        H_right = zeros(0, length(ukf_right_P));
        res_right = zeros(0);

        % set ukf state for update
        ukf_up_state.Rot = ukf_state.Rot;
        ukf_up_state.p = ukf_state.p;
        Rot = ukf_state.Rot;
        p = ukf_state.p;
        
        ukf_left_up_state.Rot = ukf_left_state.Rot;
        ukf_left_up_state.p = ukf_left_state.p;
        Rot_left = ukf_left_state.Rot;
        p_left = ukf_left_state.p;
        
        ukf_right_up_state.Rot = ukf_right_state.Rot;
        ukf_right_up_state.p = ukf_right_state.p;
        Rot_right = ukf_right_state.Rot;
        p_right = ukf_right_state.p;
        % update each landmark already in the filter
        for i = 1:N_y
            idx = find(~(ukf_lmk - y(3, i))); % same for all the filters
            if isempty(idx)
                continue
            end
            % indices of the robot and observed landmark in P
            up_idxs = [1:3 2+(2*idx:2*idx+1)];
            ukf_up_state.p_l = ukf_state.p_l(:, idx);
            ukf_left_up_state.p_l = ukf_left_state.p_l(:, idx);
            ukf_right_up_state.p_l = ukf_right_state.p_l(:, idx);
            % compute observability matrices and residual
            [H_i, res_i] = ukf_jacobian_update(ukf_up_state, ukf_P, ...
                y(1:2, i), ukf_h, ukf_up_phi, ukf_weights, up_idxs);
            [H_left_i, res_left_i] = ukf_jacobian_update(ukf_left_up_state, ukf_left_P, ...
                y(1:2, i), ukf_h, ukf_left_up_phi, ukf_weights, up_idxs);
  
            [H_right_i, res_right_i] = ukf_jacobian_update(ukf_right_up_state, ukf_right_P, ...
                y(1:2, i), ukf_h, ukf_right_up_phi, ukf_weights, up_idxs);
%             increase observabily matrix and residual
            H = [H; H_i];
            res = [res; res_i];
            
            H_left = [H_left; H_left_i];
            res_left = [res_left; res_left_i];
            
            H_right = [H_right; H_right_i];
            res_right = [res_right; res_right_i];
        end
        
        % update only if some landmards have been observed
        if size(H_right, 1) > 0
            R_n = obs_noise_std^2 * eye(size(H_right, 1));
            % update state and covariance with Kalman update
            [ukf_state, ukf_P] = kf_update(ukf_state, ukf_P, H, res, ...
                R_n, ukf_up_phi);
            [ukf_left_state, ukf_left_P] = kf_update(ukf_left_state, ukf_left_P, H_left, res_left, ...
                R_n, ukf_left_up_phi);
            [ukf_right_state, ukf_right_P] = kf_update(ukf_right_state, ukf_right_P, H_right, res_right, ...
                R_n, ukf_right_up_phi);
        end
        
        % update 
       [iekf_state, iekf_P] = slam2d_iekf_update(iekf_state, ...
           iekf_P, y, R, iekf_lmk);
       [ekf_state, ekf_P] = slam2d_ekf_update(ekf_state, ...
           ekf_P, y, R, ekf_lmk);
       
%       augment the state with new landmark
        for i = 1:N_y
            idx = find(~(ukf_lmk - y(3, i)));
            if ~isempty(idx)
                continue
            end
            % augment the landmark state
            ukf_lmk = [ukf_lmk; y(3, i)];
            % indices of the new landmark
            idx = find(~(ukf_lmk - y(3, i)));
            up_idxs = [1:3 2+(2*idx:2*idx+1)];

            % new landmark position
            p_l = p + Rot*y(1:2, i);
            ukf_up_state.p_l = p_l;
            ukf_state.p_l = [ukf_state.p_l p_l];
            
            p_left_l = p_left + Rot_left*y(1:2, i);
            ukf_left_up_state.p_l = p_left_l;
            ukf_left_state.p_l = [ukf_left_state.p_l p_left_l];
            
            p_right_l = p_right + Rot_right*y(1:2, i);
            ukf_right_up_state.p_l = p_right_l;
            ukf_right_state.p_l = [ukf_right_state.p_l p_right_l];

            % get Jacobian and then covariance following [2]
            R_n = obs_noise_std^2 * eye(2);
            [~, ukf_P] = ukf_aug(ukf_state, ukf_P, y(1:2, i), ukf_z, ...
                ukf_z_aug, ukf_aug_phi, ukf_aug_weights, ukf_aug_idxs, R_n);
            
            [~, ukf_right_P] = ukf_aug(ukf_right_state, ukf_right_P, y(1:2, i), ukf_z, ...
                ukf_z_aug, ukf_right_aug_phi, ukf_aug_weights, ukf_aug_idxs, R_n);
            
            [~, ukf_left_P] = ukf_aug(ukf_left_state, ukf_left_P, y(1:2, i), ukf_z, ...
                ukf_z_aug, ukf_left_aug_phi, ukf_aug_weights, ukf_aug_idxs, R_n);
            
            
        end
       
        % augment 
        [iekf_state, iekf_P, iekf_lmk] = slam2d_iekf_augment(iekf_state, ...
            iekf_P, y, R, iekf_lmk);
       [ekf_state, ekf_P, ekf_lmk] = slam2d_ekf_augment(ekf_state, ...
           ekf_P, y, R, ekf_lmk);
       
    
        % save estimates
        ukf_states(n) = ukf_state;
        ukf_left_states(n) = ukf_left_state;
        ukf_right_states(n) = ukf_right_state;
        ekf_states(n) = ekf_state;
        iekf_states(n) = iekf_state;
        
        ukf_Ps(n, 1:length(ukf_P), 1:length(ukf_P)) = ukf_P;
        ukf_left_Ps(n, 1:length(ukf_P), 1:length(ukf_P)) = ukf_left_P;
        ukf_right_Ps(n, 1:length(ukf_right_P), 1:length(ukf_right_P)) = ukf_right_P;
        ekf_Ps(n, 1:length(ekf_P), 1:length(ekf_P)) = ekf_P;
        iekf_Ps(n, 1:length(iekf_P), 1:length(iekf_P)) = iekf_P;
        
    end
    % record errors
    ukf_left_err(:, n_mc) = slam2d_error(states, ukf_left_states);
    ukf_right_err(:, n_mc) = slam2d_error(states, ukf_right_states);
    ukf_err(:, n_mc) = slam2d_error(states, ukf_states);
    iekf_err(:, n_mc) = slam2d_error(states, iekf_states);
    ekf_err(:, n_mc) = slam2d_error(states, ekf_states);
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

benchmark_slam2d_helper;

%%
% Which filter is the most accurate ? We expect it is the left UKF as it
% takes advantage of both theory of IEKF uncertainty representation and
% better noise incorporation of UKF compared to EKF. Depending on which
% source of error (sensor noise, initial condition) is the most important,
% it can lead to different results. We also note that we have set all the
% filters with the same "true" noise covariance parameters. Hovewer, both
% EKF and UKF based algorihms may better deal with non-linearity by e.g.
% inflated propagation noise covariance.

%% Conclusion
% This script compares different algorithm on the 2D robot localization
% example. Two groups of filters emerge: the standard UKF and the EKF; and
% the left UKF, right UKF and IEKF. For the considered set of parameters,
% it is evident that embedded the state in $SE(2)$ is advantageous for
% state estimation. Choosing then betwenn left UKF, right UKF or IEKF has
% negligeable effet.
%
% You can now compare the filters in different scenarios. UKF and their
% (I)EKF conterparts may obtain different results when noise is inflated.

