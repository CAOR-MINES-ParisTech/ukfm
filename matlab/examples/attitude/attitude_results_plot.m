function [] = attitude_results_plot(ukf_states, ukf_Ps, states, omegas, dt)
%ATTITUDE_RESULTS_PLOT
%
% Syntax: [] = attitude_results_plot(ukf_states, ukf_Ps, states, ...
%               omegas, dt)

set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');
cur_folder = pwd;
cur_folder = cur_folder(end-5:end);
Rots = attitude_get_states(states);

N = length(Rots);
t = linspace(0, dt*N, N);

% get estimation state
ukf_state_table = struct2table(ukf_states);
ukf_Rots = ukf_state_table.Rot;

% get roll, pitch and yaw
rpys = zeros(3, N);
ukf_err = zeros(3, N);
for n = 1:N
    rpys(:, n) = so3_to_rpy(Rots{n});
    ukf_err(:, n) = so3_log(ukf_Rots{n}'*Rots{n});
end
ukf3sigma = 3*[sqrt(ukf_Ps(:, 1, 1))';
    sqrt(ukf_Ps(:, 2, 2))';
    sqrt(ukf_Ps(:, 3, 3))'];

fig1 = figure;
hold on;
title('Orientation')
xlabel('$t$ (s)')
ylabel('orientation (deg)')
hold on;
grid on;
plot(t, 180/pi*rpys(1, :), 'r');
plot(t, 180/pi*rpys(2, :), 'y');
plot(t, 180/pi*rpys(3, :), 'k');
legend('roll', 'pitch', 'yaw');

fig2 = figure;
hold on;
title('Roll error')
xlabel('$t$ (s)')
ylabel('Roll error (deg)')
hold on;
grid on;
plot(t, 180/pi*ukf_err(1, :), 'b', 'LineWidth', 1.5);
plot(t, 180/pi*ukf3sigma(1, :), 'b--');
plot(t, -180/pi*ukf3sigma(1, :), 'b--');
legend('UKF', '$3\sigma$ UKF');

fig3 = figure;
hold on;
title('Pitch error')
xlabel('$t$ (s)')
ylabel('Pitch error (deg)')
hold on;
grid on;
plot(t, 180/pi*ukf_err(2, :), 'b', 'LineWidth', 1.5);
plot(t, 180/pi*ukf3sigma(2, :), 'b--');
plot(t, -180/pi*ukf3sigma(2, :), 'b--');
legend('UKF', '$3\sigma$ UKF');

fig4 = figure;
hold on;
title('Yaw error')
xlabel('$t$ (s)')
ylabel('Yaw error (deg)')
hold on;
grid on;
plot(t, 180/pi*ukf_err(3, :), 'b', 'LineWidth', 1.5);
plot(t, 180/pi*ukf3sigma(3, :), 'b--');
plot(t, -180/pi*ukf3sigma(3, :), 'b--');
legend('UKF', '$3\sigma$ UKF');

if cur_folder == "matlab"
    print(fig1, 'examples/html/figures/main_attitude_01', '-dpng', '-r600')
    print(fig2, 'examples/html/figures/main_attitude_02', '-dpng', '-r600')
    print(fig3, 'examples/html/figures/main_attitude_03', '-dpng', '-r600')
    print(fig4, 'examples/html/figures/main_attitude_04', '-dpng', '-r600')
end
end

