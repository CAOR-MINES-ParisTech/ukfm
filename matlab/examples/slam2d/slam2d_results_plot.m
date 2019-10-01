function [] = slam2d_results_plot(ukf_states, ukf_Ps, states, dt, ldks)
%SLAM2D_RESULTS_PLOT
%
% Syntax: [] = slam2d_results_plot(ukf_states, ukf_Ps, states, dt, ldks)
set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');
cur_folder = pwd;
cur_folder = cur_folder(end-5:end);

[Rots, ps] = slam2d_get_states(states);
[ukf_Rots, ukf_ps] = slam2d_get_states(ukf_states);

N = length(ps);
t = linspace(0, dt*N, N);

% get orientation error and confidence interval
N = length(ps);
ukf_errs = slam2d_errors(Rots, ukf_Rots, ps, ukf_ps);

fig1 = figure;
title('Position')
xlabel('$x$ (m)')
ylabel('$y$ (m)')
hold on;
grid on;
plot(ps(:, 1), ps(:, 2), 'LineWidth', 1.5);
plot(ukf_ps(:, 1), ukf_ps(:, 2));
scatter(ldks(1, :), ldks(2, :));
axis equal;
legend('true position', 'UKF', 'landmarks');

ukf3sigma = 3*sqrt(ukf_Ps(:, 1, 1));
fig2 = figure;
title('Orientation error (deg)')
xlabel('$t$ (s)')
ylabel('orror (deg)')
hold on;
grid on;
plot(t, 180/pi*ukf_errs(1, :), 'b', 'LineWidth', 1.5);
plot(t, 180/pi*ukf3sigma,'b--');
plot(t, 180/pi*(- ukf3sigma),'b--');
legend('UKF', '$3\sigma$ UKF');

p_err = sqrt(ukf_errs(3, :).^2 + ukf_errs(2, :).^2);
ukf3sigma = 3*sqrt(ukf_Ps(:, 2, 2) + ukf_Ps(:, 3, 3));
fig3 = figure;
title('Robot position error (m)')
xlabel('$t$ (s)')
ylabel('error (m)')
hold on;
grid on;
plot(t, p_err, 'b', 'LineWidth', 1.5);
plot(t, ukf3sigma,'b--');
legend('UKF', '$3\sigma$ UKF');

if cur_folder == "matlab"
    print(fig1, 'examples/html/figures/main_slam2d_01', '-dpng', '-r600')
    print(fig2, 'examples/html/figures/main_slam2d_02', '-dpng', '-r600')
    print(fig3, 'examples/html/figures/main_slam2d_03', '-dpng', '-r600')
end
end