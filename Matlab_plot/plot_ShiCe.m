clc;
clear all;
close all;
addpath("E:\Python_code\ULA_DOA_WTG\Matlab_code\ShiCe_Data\Model_estimation_Data")

line_width = 1.6;
alpha_value = 0.6;

data = load('09_06_2024_17_09_43_aoadata_IQ_CV_subarray8.mat');

Noise_Subspace_MODEL = data.Noise_Subspace_MODEL;
DeepCNN_MODEL = data.DeepCNN_MODEL;
CV_MODEL = data.CV_MODEL;
MUSIC = data.MUSIC;
CBF = data.CBF;
MVDR = data.MVDR;
SS_Music = data.SS_Music;
Toeplitz_Music = data.Toeplitz_Music;
PM = data.PM;
true_spectrum = data.true_spectrum;

figure;
hold on;

angles = linspace(-90, 90, length(Noise_Subspace_MODEL));

h1 = plot(angles(Noise_Subspace_MODEL ~= 0), Noise_Subspace_MODEL(Noise_Subspace_MODEL ~= 0), 'r-', 'LineWidth', line_width+1, 'DisplayName', 'Noise Subspace MODEL');
h2 = plot(angles(MUSIC ~= 0), MUSIC(MUSIC ~= 0), 'b:', 'LineWidth', line_width, 'DisplayName', 'MUSIC');
h3 = plot(angles(CBF ~= 0), CBF(CBF ~= 0), 'm-.', 'LineWidth', line_width, 'DisplayName', 'CBF');
h4 = plot(angles(MVDR ~= 0), MVDR(MVDR ~= 0), 'c-', 'LineWidth', line_width, 'DisplayName', 'MVDR');
h5 = plot(angles(SS_Music ~= 0), SS_Music(SS_Music ~= 0), 'k--', 'LineWidth', line_width, 'DisplayName', 'SS-MUSIC');
h6 = plot(angles(Toeplitz_Music ~= 0), Toeplitz_Music(Toeplitz_Music ~= 0), 'b-', 'LineWidth', line_width, 'DisplayName', 'Toeplitz-MUSIC');
h7 = plot(angles(PM ~= 0), PM(PM ~= 0), 'g-', 'LineWidth', line_width, 'DisplayName', 'PM');
h8 = plot(linspace(-90, 90-1, length(DeepCNN_MODEL)), DeepCNN_MODEL, 'Color', [1, 0.65, 0], 'LineWidth', line_width, 'DisplayName', 'DeepCNN');
h8_1 = plot(linspace(-90, 90-1, length(CV_MODEL)), CV_MODEL, 'Color', [0.58, 0, 0.83], 'LineWidth', line_width, 'DisplayName', 'CV');

y_limits = ylim;
for i = 1:length(true_spectrum)
    if true_spectrum(i) == 1
        plot([angles(i), angles(i)], [y_limits(1), y_limits(2)], 'k--', 'LineWidth', 1);
        plot(angles(i), y_limits(2), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y');
    end
end

[pks, locs] = findpeaks(Noise_Subspace_MODEL);

for i = 1:length(pks)
    if pks(i) > 0.1
        plot(angles(locs(i)), pks(i), 'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'Peaks of Noise Subspace MODEL');
    end
end

h9 = plot(nan, nan, 'k--', 'LineWidth', 2);
h10 = plot(nan, nan, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y');

legend([h1, h2, h3, h4, h5, h6, h7, h8, h8_1, h9], {'Proposed', 'MUSIC', 'CBF', 'MVDR', 'SS-MUSIC', 'Toeplitz-MUSIC', 'PM', 'E2E-CNN', 'E2E-CV-CNN', 'True-Spectrum'});

h1.Color(4) = alpha_value + 0.4;
h2.Color(4) = alpha_value;
h3.Color(4) = alpha_value;
h4.Color(4) = alpha_value;
h5.Color(4) = alpha_value;
h6.Color(4) = alpha_value;
h7.Color(4) = alpha_value;
h8.Color(4) = alpha_value;
h8_1.Color(4) = alpha_value;

set(gca, 'LooseInset', [0,0,0,0]);
box('on');
xlim([-90, 90]);
ylim([0, 1]);
xticks(-90:30:90);
xlabel('Angle (°)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
set(gca, 'FontSize', 12);

hold off;
