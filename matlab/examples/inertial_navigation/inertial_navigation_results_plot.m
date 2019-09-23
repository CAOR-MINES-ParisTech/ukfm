function [] = inertial_navigation_results_plot(ukf_states, ukf_Ps, ...
    states, dt)
%INERTIAL_NAVIGATION_RESULTS_PLOT plot
%
% Syntax: [] = inertial_navigation_results_plot(ukf_states, ukf_Ps, ...
%       states, dt)

set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');

% landmarks
ldk = [[0; 2; 2], [-2; -2; -2], [2; -2; -2]];

% get true state
state_table = struct2table(states);
Rots = state_table.Rot;
ps = cell2mat(state_table.p')';
N = length(ps);
t = linspace(0, dt*N, N);

% get estimated state
ukf_state_table = struct2table(ukf_states);
ukf_Rots = ukf_state_table.Rot;
ukf_ps = cell2mat(ukf_state_table.p')';

fig = figure;
hold on;
grid on;
title('Position')
xlabel('$x$ (m)')
ylabel('$y$ (m)')
zlabel('$z$ (m)')
scatter3(ldk(1, :), ldk(2, :), ldk(3, :), 'r');
plot3(ps(:, 1), ps(:, 2), ps(:, 3), 'k');
plot3(ukf_ps(:, 1), ukf_ps(:, 2), ukf_ps(:, 3), 'b');
legend('features', 'true trajectory', 'UKF');
print(fig, 'matlab/examples/html/main_inertial_navigation_01', ...
    '-dpng', '-r600')

err_orientation = zeros(N, 1);
for n = 1:N
    err_orientation(n) = norm(so3_log(Rots{n}' * ukf_Rots{n}));
end
three_sigma_orientation = 3*sqrt(ukf_Ps(:, 1, 1).^2 + ...
    ukf_Ps(:, 2, 2).^2 + ukf_Ps(:, 3, 3).^2);
fig = figure;
hold on;
grid on;
title('Attitude error (deg)')
xlabel('$t$ (s)')
ylabel('error (deg)')
plot(t, 180/pi*err_orientation, 'b');
plot(t, 180/pi*three_sigma_orientation, '--b');
plot(t, -180/pi*three_sigma_orientation, '--b');
legend('UKF', '$3\sigma$ UKF');
print(fig, 'matlab/examples/html/main_inertial_navigation_02', ...
    '-dpng', '-r600')

err_p = sqrt(sum((ps-ukf_ps).^2, 2));
three_sigma_p = 3*sqrt(ukf_Ps(:, 7, 7).^2 + ukf_Ps(:, 8, 8).^2 ...
    + ukf_Ps(:, 9, 9).^2);
fig = figure;
hold on;
grid on;
title('Position error (m)')
xlabel('$t$ (s)')
ylabel('error (m)')
plot(t, err_p, 'b');
plot(t, three_sigma_p, '--b');
plot(t, -three_sigma_p, '--b');
legend('UKF', '$3\sigma$ UKF');
print(fig, 'matlab/examples/html/main_inertial_navigation_03', ...
    '-dpng', '-r600')
end

