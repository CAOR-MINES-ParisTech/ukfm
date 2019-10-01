function [w] = ukf_set_weight(d, q, alpha)
%UKF_SET_WEIGHT Compute weight parameters
%
% Syntax: [w] = ukf_set_weight(d, q, alpha)
%
% Inputs:
%    d - uncertainty state dimension
%    q - noise dimension
%    alpha - parameters
%
% Outputs:
%    w  - weights

% parameters for state propagation w.r.t. state uncertainty
lambda = (alpha(1)^2 -1) * d;
w_d.lambda = lambda;
w_d.sqrt_d_lambda = sqrt(d + lambda);
w_d.wj = 1/(2*(d + lambda));
w_d.wm0 = lambda/(lambda + d);
w_d.wc0 = lambda/(lambda + d) + 3 - alpha(1)^2;

% parameters for state propagation w.r.t. noise
lambda = (alpha(1)^2 -1) * q;
w_q.lambda = lambda;
w_q.sqrt_q_lambda = sqrt(q + lambda);
w_q.wj = 1/(2*(q + lambda));
w_q.wm0 = lambda/(lambda + q);
w_q.wc0 = lambda/(lambda + q) + 3 - alpha(2)^2;

% parameters for state update
lambda = (alpha(3)^2 -1) * d;
w_u.lambda = lambda;
w_u.sqrt_d_lambda = sqrt(d + lambda);
w_u.wj = 1/(2*(d + lambda));
w_u.wm0 = lambda/(lambda + d);
w_u.wc0 = lambda/(lambda + d) + 3 - alpha(3)^2;

w.d = w_d;
w.q = w_q;
w.u = w_u;
end
