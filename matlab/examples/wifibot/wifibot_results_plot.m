function [] = wifibot_results_plot(ukf_states, ukf_Ps, states, dt, ys)
%WIFIBOT_RESULTS_PLOT
%
% Syntax: [] = wifibot_results_plot(ukf_states, ukf_Ps, states, dt, ys)

set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');
cur_folder = pwd;
cur_folder = cur_folder(end-6:end);
% get true state
state_table = struct2table(states);
Rots = state_table.Rot;
ps = cell2mat(state_table.p')';

% get estimated state
ukf_state_table = struct2table(ukf_states);
ukf_Rots = ukf_state_table.Rot;
ukf_ps = cell2mat(ukf_state_table.p')';

% get orientation error and confidence interval
N = length(ps);
t = linspace(0, dt*N, N);
ukf_err = zeros(N, 1);
for n = 1:N
    ukf_err(n) = so2_log(ukf_Rots{n}' * Rots{n});
end

fig1 = figure;
title('Position in $xy$-plan')
xlabel('$x$ (m)')
ylabel('$y$ (m)')
hold on;
grid on;
plot(ps(:, 1), ps(:, 2), 'k', 'LineWidth', 1.5);
scatter(ys(1, :), ys(2, :), 'r');
plot(ukf_ps(:, 1), ukf_ps(:, 2), 'b');
legend('true position', 'GPS measurements', 'UKF');
axis equal;

ukf3sigma = 3*sqrt(ukf_Ps(:, 1, 1));
fig2 = figure;
title('Robot attitude error (deg)')
xlabel('$t$ (s)')
ylabel('orientation error (deg)')
hold on;
grid on;
plot(t, 180/pi*ukf_err, 'b', 'LineWidth', 1.5);
plot(t, 180/pi*ukf3sigma,'b--');
plot(t, 180/pi*(- ukf3sigma),'b--');
legend('UKF', '$3\sigma$ UKF');

ukf3sigma = 3*sqrt(ukf_Ps(:, 2, 2) + ukf_Ps(:, 3, 3));
p_err = sqrt(sum((ps-ukf_ps).^2, 2));
fig3 = figure;
title('Robot position error (m)')
xlabel('$t$ (s)')
ylabel('error (m)')
hold on;
grid on;
plot(t, p_err, 'b', 'LineWidth', 1.5);
plot(t, ukf3sigma,'b--');
legend('UKF', '$3\sigma$ UKF');

% save to a correct resolution
if cur_folder ~= "matlab"
    print(fig1, 'examples/html/figures/main_wifibot_01', '-dpng', '-r600')
    print(fig2, 'examples/html/figures/main_wifibot_02', '-dpng', '-r600')
    print(fig3, 'examples/html/figures/main_wifibot_03', '-dpng', '-r600')
end
end

