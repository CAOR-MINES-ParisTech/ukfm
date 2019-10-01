function [] = inertial_navigation_results_plot(ukf_states, ukf_Ps, ...
    states, dt)
%INERTIAL_NAVIGATION_RESULTS_PLOT
%
% Syntax: [] = inertial_navigation_results_plot(ukf_states, ukf_Ps, ...
%       states, dt)

set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');
cur_folder = pwd;
cur_folder = cur_folder(end-5:end);
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

fig1 = figure;
hold on;
grid on;
title('Horizontal vehicle position')
xlabel('$x$ (m)')
ylabel('$y$ (m)')
scatter(ldk(1, :), ldk(2, :), 'r');
plot(ps(:, 1), ps(:, 2), 'k', 'LineWidth', 1.5);
plot(ukf_ps(:, 1), ukf_ps(:, 2), 'b');
legend('landmarks', 'true trajectory', 'UKF');

err_orientation = zeros(N, 1);
for n = 1:N
    err_orientation(n) = norm(so3_log(Rots{n}' * ukf_Rots{n}));
end
three_sigma_orientation = 3*sqrt(ukf_Ps(:, 1, 1).^2 + ...
    ukf_Ps(:, 2, 2).^2 + ukf_Ps(:, 3, 3).^2);
fig2 = figure;
hold on;
grid on;
title('Attitude error (deg)')
xlabel('$t$ (s)')
ylabel('error (deg)')
plot(t, 180/pi*err_orientation, 'b', 'LineWidth', 1.5);
plot(t, 180/pi*three_sigma_orientation, '--b');
legend('UKF', '$3\sigma$ UKF');

err_p = sqrt(sum((ps-ukf_ps).^2, 2));
three_sigma_p = 3*sqrt(ukf_Ps(:, 7, 7).^2 + ukf_Ps(:, 8, 8).^2 ...
    + ukf_Ps(:, 9, 9).^2);
fig3 = figure;
hold on;
grid on;
title('Position error (m)')
xlabel('$t$ (s)')
ylabel('error (m)')
plot(t, err_p, 'b', 'LineWidth', 1.5);
plot(t, three_sigma_p, '--b');
legend('UKF', '$3\sigma$ UKF');

if cur_folder == "matlab"
    print(fig1, 'examples/html/figures/main_inertial_navigation_01', ...
        '-dpng', '-r600')
    print(fig2, 'examples/html/figures/main_inertial_navigation_02', ...
        '-dpng', '-r600')
    print(fig3, 'examples/html/figures/main_inertial_navigation_03', ...
        '-dpng', '-r600')
end
end

