function [] = slam2d_results_plot(ukf_states, ukf_Ps, states, dt, ldks)
%SLAM2D_RESULTS_PLOT plot
%
% Syntax: [] = slam2d_results_plot(ukf_states, ukf_Ps, states, dt, ldks)
set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');

% get true state
state_table = struct2table(states);
Rots = state_table.Rot;
ps = cell2mat(state_table.p')';

% get estimation state
ukf_state_table = struct2table(ukf_states);
ukf_Rots = ukf_state_table.Rot;
ukf_ps = cell2mat(ukf_state_table.p')';

N = length(ps);
t = linspace(0, dt*N, N);

% get orientation error and confidence interval
N = length(ps);
t = linspace(0, dt*N, N);
ukf_err = zeros(N, 1);
for n = 1:N
    ukf_err(n) = so2_log(ukf_Rots{n}' * Rots{n});
end

figure;
title('Position in $xy$-plan')
xlabel('$x$ (m)')
ylabel('$y$ (m)')
hold on;
grid on;
plot(ps(:, 1), ps(:, 2));
plot(ukf_ps(:, 1), ukf_ps(:, 2));
scatter(ldks(1, :), ldks(2, :));
axis equal;
legend('true position', 'UKF', 'landmarks');
print(fig, 'matlab/examples/html/main_slam2d_01', '-dpng', '-r600')

ukf_three_sigma = 3*sqrt(ukf_Ps(:, 1, 1));
fig = figure;
title('Orientation error (deg)')
xlabel('$t$ (s)')
ylabel('orror (deg)')
hold on;
grid on;
plot(t, 180/pi*ukf_err, 'b');
plot(t, 180/pi*ukf_three_sigma,'b--');
plot(t, 180/pi*(- ukf_three_sigma),'b--');
legend('UKF', '$3\sigma$ UKF');
print(fig, 'matlab/examples/html/main_slam2d_02', '-dpng', '-r600')

p_err = sqrt(sum((ps-ukf_ps).^2, 2));
ukf_three_sigma = 3*sqrt(ukf_Ps(:, 2, 2) + ukf_Ps(:, 3, 3));
fig = figure;
title('Robot position error (m)')
xlabel('$t$ (s)')
ylabel('error (m)')
hold on;
grid on;
plot(t, p_err, 'b');
plot(t, ukf_three_sigma,'b--');
plot(t, -ukf_three_sigma,'b--');
legend('UKF', '$3\sigma$ UKF');
print(fig, 'matlab/examples/html/main_slam2d_03', '-dpng', '-r600')
end