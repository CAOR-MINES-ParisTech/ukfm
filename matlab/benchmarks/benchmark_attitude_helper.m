set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');
cur_folder = pwd;
cur_folder = cur_folder(end-5:end);

t = linspace(0, dt*N, N);

f = @(x) squeeze(sqrt(mean(mean(x.^2, 3), 1)));
ukf_left_errs = f(ukf_left_errs);
ukf_right_errs = f(ukf_right_errs);
ekf_errs = f(ekf_errs);

fig1 = figure;
title('Orientation error (deg)')
xlabel('$t$ (s)')
ylabel('orientation error (deg)')
hold on;
grid on;
plot(t, 180/pi*ukf_left_errs, 'g');
plot(t, 180/pi*ukf_right_errs, 'c');
plot(t, 180/pi*ekf_errs, 'r');
legend('\textbf{left UKF}', '\textbf{right UKF}', 'EKF');

if cur_folder == "matlab"
    print(fig1, 'benchmarks/html/figures/benchmark_attitude_01', ...
        '-dpng', '-r600')
end

f = @(x) sprintf('%0.2f', 180/pi*sqrt(mean(x.^2)));
ukf_left_err = f(ukf_left_errs);
ukf_right_err = f(ukf_right_errs);
ekf_err = f(ekf_errs);

disp(' ')
disp('Root Mean Square Error w.r.t. orientation (deg)');
disp("    -left UKF    : " + ukf_left_err);
disp("    -right UKF   : " + ukf_right_err);
disp("    -EKF         : " + ekf_err);
