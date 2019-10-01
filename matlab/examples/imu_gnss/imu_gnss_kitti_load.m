function [omegas, ys, one_hot_ys, t] = imu_gnss_kitti_load(gps_freq)

cur_folder = pwd;
cur_folder = cur_folder(end-5:end);

f_name = "data/KittiGps_converted.txt";
gps_data = load(f_name);
f_name = "data/KittiEquivBiasedImu.txt";
imu_data = load(f_name);
t = imu_data(:, 1);
t0 = t(1);
t = t - t0;
N = length(t);

omegaX = imu_data(:, 6);
omegaY = imu_data(:, 7);
omegaZ = imu_data(:, 8);
accelX = imu_data(:, 3);
accelY = imu_data(:, 4);
accelZ = imu_data(:, 5);

omegas(N) = struct;
for n = 1:N
    omegas(n).gyro = [omegaX(n); omegaY(n); omegaZ(n)]; 
    omegas(n).acc = [accelX(n); accelY(n); accelZ(n)]; 
end

% total number of timestamps
N = length(t);
t_gps = gps_data(:, 1) - t0;
N_gps = length(t_gps);

% vector to know where GPS measurement happen
one_hot_ys = zeros(N, 1);
n_y = 1;
ys = zeros(3, N_gps);
for n = 1:N
    if t_gps(n_y) <= t(n)
        ys(:, n_y) = gps_data(n_y, 2:4);
        one_hot_ys(n) = 1;
        n_y = n_y + 1;
    end
    if n_y > length(t_gps)
        break;
    end
end
end