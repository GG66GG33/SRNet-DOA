clc;
clear all;
close all;

file_path = 'SNR-9to9_Snap100_-10.49_10.62_CV.csv';
data = readtable(file_path);

SNR_values = data.Var1;
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

plot(SNR_values, model_rmse, '-o', 'DisplayName', 'Proposed', 'LineWidth', 2, 'Color', 'r', 'Marker', 'o');
hold on;
plot(SNR_values, deepcnn_rmse, '-^', 'DisplayName', 'E2E-CNN', 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410], 'Marker', '^');
plot(SNR_values, music_rmse, '-s', 'DisplayName', 'MUSIC', 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980], 'Marker', 's');
plot(SNR_values, cbf_rmse, '-d', 'DisplayName', 'CBF', 'LineWidth', 2, 'Color', [0.9290, 0.6940, 0.1250], 'Marker', 'd');
plot(SNR_values, mvdr_rmse, '-p', 'DisplayName', 'MVDR', 'LineWidth', 2, 'Color', [0.4940, 0.1840, 0.5560], 'Marker', 'p');
plot(SNR_values, pm_rmse, '-h', 'DisplayName', 'PM', 'LineWidth', 2, 'Color', [0.4660, 0.6740, 0.1880], 'Marker', 'h');
plot(SNR_values, ss_music_rmse, '--*', 'DisplayName', 'SS-MUSIC', 'LineWidth', 2, 'Color', [0.3010, 0.7450, 0.9330], 'Marker', '*');
plot(SNR_values, toeplitz_music_rmse, '--+', 'DisplayName', 'Toeplitz-MUSIC', 'LineWidth', 2, 'Color', [0.6350, 0.0780, 0.1840], 'Marker', '+');
plot(SNR_values, CV, '-x', 'DisplayName', 'E2E-CV-CNN', 'LineWidth', 2, 'Color', [1, 0.4, 0.6], 'Marker', 'x');

legend show;
legend('Location', 'best', 'NumColumns', 3);
set(gca, 'YScale', 'log');

xlabel('SNR (dB)', 'FontSize', 12);
ylabel('RMSE (°)', 'FontSize', 12);

set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'FontSize', 12);
box('on');
xticks(-9:2:9);
xlim([-9,9]);
ylim([0.06,100]);

grid on;
