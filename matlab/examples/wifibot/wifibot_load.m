function [states, omegas, ys, one_hot_ys, t] = wifibot_load(...,
    n_sequence, gps_freq, gps_noise_std)

f_name = pwd + "/matlab/examples/data/wifibot" + ...
    num2str(n_sequence) + ".mat";
data = load(f_name);
states = data.state;
omegas = data.omega;
t = data.t;
N = data.N;

% simulate measurement
% vector to know where GPS measurement happen
one_hot_ys = zeros(N, 1);
t_gps = t(1) + (1:10000)/gps_freq;
k = 1;
ys = zeros(2, length(t_gps));
for n = 1:N
    if t_gps(k) <= t(n)
        ys(:, k) = states(n).p + gps_noise_std*randn(2, 1);
        one_hot_ys(n) = 1;
        k = k + 1;
    end
end
ys = ys(:, 1:k);
end

