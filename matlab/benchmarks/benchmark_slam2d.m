%% 2D Robot Localization - Benchmark
% 
% Goals of this script:
% 
% * implement different UKFs on the 2D robot localization example.
% * (re)discover computational alternatives for performing UKF:
% * design the Extended Kalman Filter (EKF) and the Invariant Extended
%   Kalman Filter (IEKF) [BB17].
% * compare the different algorithms with Monte-Carlo simulations.
% 
% _We assume the reader is already familiar with the considered problem
% described in the related examples._
% 
% For the given, three different UKFs emerge, defined respectively as:
% 
% # The state is embedded in $SO(2) \times R^2$, where the retraction 
%   $\varphi(.,.)$ is the $SO(2)$ exponential for orientation and the vector 
%   addition for positions. The inverse retraction $\varphi^{-1}_.(.)$ is the 
%   $SO(2)$ logarithm for orientation and the vector subtraction for positions.
% # The state is embedded in $SE(2)$ with left multiplication, i.e. the 
%   retraction $\varphi(.,.)$ is the $SE(2)$ exponential, where the state
%   multiplies on the left the uncertainty $\xi$. The inverse retraction 
%   $\varphi^{-1}_.(.)$ is the $SE(2)$ logarithm. This UKF corresponds to the 
%   Invariant Extended Kalman Filter (IEKF) recommended in [BB17].
% # The state is embedded in $SE(2)$ with right multiplication, i.e. the 
%   retraction $\varphi(.,.)$ is the $SE(2)$ exponential, where the state 
%   multiplies on the right the uncertainty $\xi$. The inverse retraction 
%   $\varphi^{-1}_.(.)$ is the $SE(2)$ logarithm.

%% Initialization
% Start by cleaning the workspace.
clear all;
close all;

%% 
% We compare the filters on a large number of Monte-Carlo runs.

% Monte-Carlo runs
N_mc = 100;

%% Simulation Setting
% The trajectory of the robot consists of turning at constant speed. The map
% will be the same for all the simulation, where landmarks are constantly spaced
% along the robot trajectory.

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
% integration step (s)
dt = 1/odo_freq;

%% Filter Design
% Additionally to the three UKFs, we compare them to an EKF and an IEKF. The EKF
% has the same uncertainty representation as the UKF with $SO(2) \times
% R^{2(1+L)}$ uncertainty representation, whereas the IEKF has the same
% uncertainty representation as the UKF with right $SE_{1+L}(2)$ retraction.
%
% We have five similar methods, but the UKF implementations slightly differs.
% Indeed, using our vanilla UKF works for all choice of retraction but is not
% adapted to the problem from a computationally point of view. And we spare
% computation only when Jacobian is known.

% propagation noise covariance matrix
Q = diag(odo_noise_std.^2);
% measurement noise covariance matrix
R = obs_noise_std^2 * eye(2);
% initial uncertainty matrix 
P0 = zeros(3, 3);
% sigma point parameter
alpha = [1e-3 1e-3 1e-3];
% define the UKF functions
f = @slam2d_f;
h = @slam2d_h;
% reduced retraction used during propagation
red_phi = @slam2d_red_phi;
red_phi_inv = @slam2d_red_phi_inv;
% retraction used during update
up_phi = @slam2d_phi;
% we do not need phi_inv fonction for update
% retraction used for augmenting state
z_aug = @slam2d_z_aug;
ukf_aug_phi = @slam2d_aug_phi;
ukf_aug_phi_inv = @slam2d_aug_phi_inv;

ukf_left_red_phi = @slam2d_left_red_phi;
ukf_left_red_phi_inv = @slam2d_left_red_phi_inv;
ukf_left_up_phi = @slam2d_left_phi;
ukf_left_aug_phi = @slam2d_left_aug_phi;
ukf_left_aug_phi_inv = @slam2d_left_aug_phi_inv;

ukf_right_red_phi = @slam2d_right_red_phi;
ukf_right_red_phi_inv = @slam2d_right_red_phi_inv;
ukf_right_up_phi = @slam2d_right_phi;
ukf_right_aug_phi = @slam2d_right_aug_phi;
ukf_right_aug_phi_inv = @slam2d_right_aug_phi_inv;

% reduced weights during propagation
red_weights = ukf_set_weight(3, 2, alpha);
red_idxs = 1:3; % indices corresponding to the robot state in P
% weights during update
weights = ukf_set_weight(5, 2, alpha);
ukf_aug_weights = ukf_set_weight(3, 2, alpha);
ukf_aug_idxs = 1:3; % indices corresponding to the robot state in P
cholQ = chol(Q);

%%
% We set error variables before launching Monte-Carlo simulations
ukf_errs = zeros(3, N, N_mc);
ukf_left_errs = zeros(size(ukf_errs));
ukf_right_errs = zeros(size(ukf_errs));
iekf_errs = zeros(size(ukf_errs));
ekf_errs = zeros(size(ukf_errs));

ukf_nees = zeros(2, N, N_mc);
left_ukf_nees = zeros(size(ukf_nees));
right_ukf_nees = zeros(size(ukf_nees));
iekf_nees = zeros(size(ukf_nees));
ekf_nees = zeros(size(ukf_nees));

%% Monte-Carlo runs
% We run the Monte-Carlo through a for loop.

for n_mc = 1:N_mc
    disp("Monte-Carlo iteration(s): " + num2str(n_mc) + "/" + num2str(N_mc));

    % simulate true trajectory and noisy input
    [states, omegas, ldks] = slam2d_simu_f(T, odo_freq, odo_noise_std, v, gyro);
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
        red_idxs = 1:3;
        red_weights = ukf_set_weight(length(red_idxs), 2, alpha);
        [ukf_state, ukf_P] = ukf_jacobian_propagation(ukf_state, ukf_P, ...
            omegas(n-1), f, dt, red_phi, red_phi_inv, cholQ, ...
            red_weights, red_idxs);
        red_idxs = 1:length(ukf_left_P);
        red_weights = ukf_set_weight(length(red_idxs), 2, alpha);
        [ukf_left_state, ukf_left_P] = ukf_jacobian_propagation(...
            ukf_left_state, ukf_left_P, omegas(n-1), f, dt, ...
            ukf_left_red_phi, ukf_left_red_phi_inv, cholQ, ...
            red_weights, red_idxs);
        [ukf_right_state, ukf_right_P] = ukf_jacobian_propagation(...
            ukf_right_state, ukf_right_P, omegas(n-1), f, dt, ...
             ukf_right_red_phi, ukf_right_red_phi_inv, cholQ, ...
            red_weights, red_idxs);
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
                y(1:2, i), h, up_phi, weights, up_idxs);
            [H_left_i, res_left_i] = ukf_jacobian_update(ukf_left_up_state, ...
                ukf_left_P, y(1:2, i), h, ukf_left_up_phi, weights, up_idxs);
            [H_right_i, res_right_i] = ukf_jacobian_update(...
                ukf_right_up_state, ukf_right_P, y(1:2, i), h, ...
                ukf_right_up_phi, weights, up_idxs);
            % increase observabily matrix and residual
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
                R_n, up_phi);
            [ukf_left_state, ukf_left_P] = kf_update(ukf_left_state, ...
                ukf_left_P, H_left, res_left, R_n, ukf_left_up_phi);
            [ukf_right_state, ukf_right_P] = kf_update(ukf_right_state, ...
                ukf_right_P, H_right, res_right, R_n, ukf_right_up_phi);
        end
       [iekf_state, iekf_P] = slam2d_iekf_update(iekf_state, ...
           iekf_P, y, R, iekf_lmk);
       [ekf_state, ekf_P] = slam2d_ekf_update(ekf_state, ekf_P, y, R, ekf_lmk);
           
        % augment the state with new landmark
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
            [~, ukf_P] = ukf_aug(ukf_state, ukf_P, y(1:2, i), ...
                z_aug, ukf_aug_phi, ukf_aug_phi_inv, ukf_aug_weights, ...
                ukf_aug_idxs, R_n);
            [~, ukf_left_P] = ukf_aug(ukf_left_state, ukf_left_P, ...
                y(1:2, i), z_aug, ukf_left_aug_phi, ukf_left_aug_phi_inv, ...
                ukf_aug_weights, ukf_aug_idxs, R_n);
            [~, ukf_right_P] = ukf_aug(ukf_right_state, ukf_right_P, ...
                y(1:2, i), z_aug, ukf_right_aug_phi, ukf_right_aug_phi_inv, ...
                ukf_aug_weights, ukf_aug_idxs, R_n);     
        end
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
        ukf_right_Ps(n, 1:length(ukf_right_P), 1:length(ukf_right_P)) = ...
            ukf_right_P;
        ekf_Ps(n, 1:length(ekf_P), 1:length(ekf_P)) = ekf_P;
        iekf_Ps(n, 1:length(iekf_P), 1:length(iekf_P)) = iekf_P;
        
    end
    % get state trajectory
    [Rots, ps] = slam2d_get_states(states);
    [ukf_Rots, ukf_ps] = slam2d_get_states(ukf_states);
    [ukf_left_Rots, ukf_left_ps] = slam2d_get_states(ukf_left_states);
    [ukf_right_Rots, ukf_right_ps] = slam2d_get_states(ukf_right_states);
    [iekf_Rots, iekf_ps] = slam2d_get_states(iekf_states);
    [ekf_Rots, ekf_ps] = slam2d_get_states(ekf_states);
    
    % record errors
    ukf_errs(:, :, n_mc) = slam2d_errors(Rots, ukf_Rots, ps, ukf_ps);
    ukf_left_errs(:, :, n_mc) = slam2d_errors(Rots, ukf_left_Rots, ps, ...
        ukf_left_ps);
    ukf_right_errs(:, :, n_mc) = slam2d_errors(Rots, ukf_right_Rots, ...
        ps, ukf_right_ps);
    iekf_errs(:, :, n_mc) = slam2d_errors(Rots, iekf_Rots, ps, iekf_ps);
    ekf_errs(:, :, n_mc) = slam2d_errors(Rots, ekf_Rots, ps, ekf_ps);
    
    % record NEES
    ukf_nees(:, :, n_mc) = slam2d_nees(ukf_errs(:, :, n_mc), ukf_Ps, ...
        ukf_Rots, ukf_ps, "STD");
    left_ukf_nees(:, :, n_mc) = slam2d_nees(ukf_left_errs(:, :, n_mc), ...
        ukf_left_Ps, ukf_left_Rots, ukf_left_ps, "LEFT");
    right_ukf_nees(:, :, n_mc) = slam2d_nees(...
        ukf_right_errs(:, :, n_mc), ukf_right_Ps, ukf_right_Rots, ...
        ukf_right_ps, "RIGHT");
    iekf_nees(:, :, n_mc) = slam2d_nees(iekf_errs(:, :, n_mc), iekf_Ps, ...
        iekf_Rots, iekf_ps, "LEFT");
    ekf_nees(:, :, n_mc) = slam2d_nees(ekf_errs(:, :, n_mc), ekf_Ps, ...
        ekf_Rots, ekf_ps, "STD");
end

%% Results
% We first visualize the results for the last run, and then plot the orientation
% and position errors averaged over Monte-Carlo.
%
% We then compute the Root Mean Squared Error (RMSE) for each method both for
% the orientation and the position.

benchmark_slam2d_helper;

%%
% Right UKF and IEKF outperform the remaining filters.
%
% We now compare the filters in term of consistency (NEES).

benchmark_slam2d_helper_nees;

%%
% The right UKF and the IEKF obtain similar NEES and are the more consistent
% filters, whereas the remaining filter have their NEES increasing.
% 
% *Which filter is the best ?* The *right UKF* and the *IEKF* are the best both
% in term of accuracy and consistency.

%% Conclusion
% This script compares different algorithms for 2D robot SLAM. The right UKF and
% the IEKF are the more accurate filters. They are also consistent along all the
% trajectory.
% 
% You can now:
%
% * compare the filters in different scenarios. UKF and their (I)EKF
%   counterparts may obtain different results when noise is inflated.
