function [ys] = slam2d_simu_h(states, obs_noise_std, N_ldk, ldks)
%SLAM2D_SIMU_H Perform measurement simulation
%
% Syntax: [ys] = slam2d_simu_h(states, obs_noise_std, N_ldk, ldks)
%
% Inputs.
%    states - states
%    obs_noise_std - observation noise standard deviation
%    N_ldk - number of landmarks
%    ldks - landmarks
%
% Outputs:
%    ys - noisy measurement for the sequence

% total number of timestamps
N = length(states);

max_range = 5;
min_range = 1;
ys = zeros(3, N_ldk, N);
for n = 1:N
    Rot = states(n).Rot;
    p = states(n).p;
    n_iter = 0;
    for i = 1:N_ldk
        p_l = ldks(:, i);
        r = norm(p_l-p);
        if r < max_range && r > min_range
            n_iter = n_iter + 1;
            ys(1:2, n_iter, n) = Rot'*(p_l-p) + obs_noise_std * ...
                randn(2, 1);
            ys(3, n_iter, n) = i;
        end
    end
end
end
