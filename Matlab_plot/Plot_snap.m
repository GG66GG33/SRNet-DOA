clc;
clear all;
close all;

file_path = 'Snap20to200_SNR0_-10.49_10.62.csv';

data = readtable(file_path);

Snap_values = data.Var1;
model_rmse = data.Model; 
CV = data.CV;
deepcnn_rmse = data.DeepCNN; 
music_rmse = data.Music;
cbf_rmse = data.CBF; 
mvdr_rmse = data.MVDR;
pm_rmse = data.PM; 
ss_music_rmse = data.SS_Music; 
toeplitz_music_rmse = data.Toeplitz_Music; 

figure;

plot(Snap_values, model_rmse, '-o', 'DisplayName', 'Proposed', 'LineWidth', 2, 'Color', 'r', 'Marker', 'o');
hold on;
plot(Snap_values, deepcnn_rmse, '-^', 'DisplayName', 'E2E-CNN', 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410], 'Marker', '^'); 
plot(Snap_values, music_rmse, '-s', 'DisplayName', 'MUSIC', 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980], 'Marker', 's'); 
plot(Snap_values, cbf_rmse, '-d', 'DisplayName', 'CBF', 'LineWidth', 2, 'Color', [0.9290, 0.6940, 0.1250], 'Marker', 'd'); 
plot(Snap_values, mvdr_rmse, '-p', 'DisplayName', 'MVDR', 'LineWidth', 2, 'Color', [0.4940, 0.1840, 0.5560], 'Marker', 'p'); 
plot(Snap_values, pm_rmse, '-h', 'DisplayName', 'PM', 'LineWidth', 2, 'Color', [0.4660, 0.6740, 0.1880], 'Marker', 'h'); 
plot(Snap_values, ss_music_rmse, '--*', 'DisplayName', 'SS-MUSIC', 'LineWidth', 2, 'Color', [0.3010, 0.7450, 0.9330], 'Marker', '*'); 
plot(Snap_values, toeplitz_music_rmse, '--+', 'DisplayName', 'Toeplitz-MUSIC', 'LineWidth', 2, 'Color', [0.6350, 0.0780, 0.1840], 'Marker', '+'); 
plot(Snap_values, CV, '-x', 'DisplayName', 'E2E-CV-CNN', 'LineWidth', 2, 'Color', [1, 0.4, 0.6], 'Marker', 'x'); 
legend show;
legend('Location', 'best', 'NumColumns', 3);
set(gca, 'YScale', 'log');

xlabel('Number of snapshots', 'FontSize', 12);
ylabel('RMSE (°)', 'FontSize', 12);

set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'FontSize', 12);
box('on');
xticks([10, 20:30:200]);
xlim([10,200]);
ylim([0.05,100]);
grid on;
