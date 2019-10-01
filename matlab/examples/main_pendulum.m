%% Pendulum Example
% The set of all points in the Euclidean space $R^{3}$, that lie on the surface
% of the unit ball about the origin belong to the two-sphere manifold,
% 
% $$ S^2 = \left\{ \mathbf{x} \in R^3 \mid \|\mathbf{x}\|_2 = 1 \right\}, $$
% 
% which is a two-dimensional manifold. Many mechanical systems such as a
% spherical pendulum, double pendulum, quadrotor with a cable-suspended load,
% evolve on either $S^2$ or products comprising of $S^2$.
% 
% In this script, we estimate the state of a system living on the sphere but
% where observations are standard vectors. You can have a text description of
% the spherical pendulum dynamics in [SE19], Section 7, and [KS19].

%% Initialization
% Start by cleaning the workspace.

clear all;
close all;

%% Model and Simulation

% sequence time (s)
T = 10;
% model frequency (Hz)
freq = 100;
% model noise standard deviation (noise is isotropic)
model_noise_std = [1/180*pi;  % orientation (rad) 
                  1/180*pi];  % orientation velocity (rad/s)
% simulate true states and noisy inputs
[states, omegas] = pendulum_simu_f(T, freq, model_noise_std);
% total number of timestamps
N = T*freq;
% integration step (s)
dt = 1/freq;

%%
% The state and the input contain the following variables:
%
%   states(n).Rot  % 3d orientation (matrix)
%   states(n).u    % 3d angular velocity
%   omegas(n)      % empty input
%
% The model dynamics is based on the Euler equations of pendulum motion.

%%
% We compute noisy measurements at low frequency based on the true states.

% observation frequency (Hz)
obs_freq = 20;
% observation noise standard deviation (m)
obs_std = 0.02;
% simulate measurements
[ys, one_hot_ys] = pendulum_simu_h(states, T, freq, obs_freq, obs_std);

%%
% We assume observing only the position of the state only in the $yz$ plan.

%% Filter Design and Initialization
% We embed the state in $SO(3) \times R^3$ with left multiplication, such that:
%
% * the retraction $\varphi(.,.)$ is the $SO(3)$ exponential for orientation
%   where the state multiplies the uncertainty on the left, and the vector
%   addition for the  velocity.
% * the inverse retraction $\varphi^{-1}_.(.)$ is the $SO(3)$ logarithm for
%   orientation and the vector subtraction for the velocity.
%
% Remaining parameter setting is standard.

% propagation noise covariance matrix
Q = blkdiag(model_noise_std(1)^2*eye(3), model_noise_std(2)^2*eye(3));
% measurement noise covariance matrix
R = obs_std^2*eye(2);
% initial uncertainty matrix
P0 = blkdiag((45/180*pi)^2*eye(3), (10/180*pi)^2*eye(3));
% sigma point parameters
alpha = [1e-3; 1e-3; 1e-3];
% define the UKF propagation and measurement functions
f = @pendulum_f;
h = @pendulum_h;
phi = @pendulum_phi;
phi_inv = @pendulum_phi_inv;
% get UKF weight parameters
weights = ukf_set_weight(6, 6, alpha);
% compute Cholewski decomposition of Q only once
cholQ = chol(Q);

ukf_state = states(1);
ukf_state.Rot = eye(3);
ukf_state.u = zeros(3, 1);
ukf_P = P0;

% set variables for recording estimates along the full trajectory
ukf_states = ukf_state;
ukf_Ps = zeros(N, length(ukf_P), length(ukf_P));
ukf_Ps(1, :, :) = ukf_P;

%% Filtering
% The UKF proceeds as a standard Kalman filter with a for loop.

k = 2;
for n = 2:N
    % propagation
    [ukf_state, ukf_P] = ukf_propagation(ukf_state, ukf_P, omegas(n-1), ...
        f, dt, phi, phi_inv, cholQ, weights);
    % update only if a measurement is received
    if one_hot_ys(n) == 1
        [ukf_state, ukf_P] = ukf_update(ukf_state, ukf_P, ys(:, k), ...
            h, phi, R, weights);
        k = k + 1;
    end
    % save estimates
    ukf_states(n) = ukf_state;
    ukf_Ps(n, :, :) = ukf_P;
end

%% Results
% We plot the position of the pendulum as function of time, the position in the
% $xy$ plan and the position in the $yz$ plan (we are more interested in the
% position of the pendulum than its orientation). We compute the $3\sigma$
% interval confidence by leveraging the _covariance retrieval_ proposed in
% [BBB19], Section V-B.

pendulum_results_plot(ukf_states, ukf_Ps, states, dt);

%%
% On the first plot, we observe that even if the state if unaccurately
% initialized, the filter estimates the depth position ($x$ axis) of the
% pendulum whereas only the $yz$ position of the pendulum is observed.
%
% The second and third plots show how the filter converges to the true state.
% Finally, the last plot reveals the consistency of the filter, where the
% interval confidence encompasses the error.

%% Conclusion
% This script shows how well works the UKF on parallelizable manifolds for
% estimating the position of a spherical pendulum where only two components of
% the pendulum are measured. The filter is accurate, robust to strong initial
% errors, and obtains consistent covariance estimates with the method proposed
% in [BBB19].
%
% You can now:
%
% * address the same problem with another retraction, e.g. with right
%   multiplication.
% * modify the measurement with 3D position.
% * consider the mass of the system as unknown and estimate it.
