function [] = sphere_results_plot(ukf_state, ukf_P, state, dt, y)
%LOCALIZATION_RESULTS_PLOT plot
%
% Syntax: [] = sphere_results_plot(ukf_state, ukf_P, state, dt, y)

set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');

% get true state
state_table = struct2table(state);
Rot = state_table.Rot;
p = cell2mat(state_table.p')';

% get estimated state
ukf_state_table = struct2table(ukf_state);
ukf_Rot = ukf_state_table.Rot;
ukf_p = cell2mat(ukf_state_table.p')';

% get orientation error and confidence interval
N = length(p);
t = linspace(0, dt*N, N);
ukf_err = zeros(N, 1);
for n = 1:N
    ukf_err(n) = so2_log(ukf_Rot{n}' * Rot{n});
end

fig = figure;
title('Position in $xy$-plan')
xlabel('$x$ (m)')
ylabel('$y$ (m)')
hold on;
grid on;
plot(p(:, 1), p(:, 2), 'k');
scatter(y(1, :), y(2, :), 'r');
plot(ukf_p(:, 1), ukf_p(:, 2), 'b');
legend('true position', 'GPS measurements', 'UKF');
axis equal;
% save to a correct resolution
print(fig, 'matlab/examples/html/main_localization_01', '-dpng', '-r600')

ukf_three_sigma = 3*sqrt(ukf_P(:, 1, 1));
fig = figure;
title('Attitude error (deg)')
xlabel('$t$ (s)')
ylabel('orientation error (deg)')
hold on;
grid on;
plot(t, 180/pi*ukf_err, 'b');
plot(t, 180/pi*ukf_three_sigma,'b--');
plot(t, 180/pi*(- ukf_three_sigma),'b--');
legend('UKF', '$3\sigma$ UKF');
print(fig, 'matlab/examples/html/main_localization_02', '-dpng', '-r600')

ukf_three_sigma = 3*sqrt(ukf_P(:, 2, 2) + ukf_P(:, 3, 3));
p_err = sqrt(sum((p-ukf_p).^2, 2));
fig = figure;
title('Position error (m)')
xlabel('$t$ (s)')
ylabel('position error (m)')
hold on;
grid on;
plot(t, p_err, 'b');
plot(t, ukf_three_sigma,'b--');
plot(t, -ukf_three_sigma,'b--');
legend('UKF', '$3\sigma$ UKF');
print(fig, 'matlab/examples/html/main_localization_03', '-dpng', '-r600')
end

