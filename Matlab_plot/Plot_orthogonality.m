file_path = 'SNR-9to9_Snap100_-10.49_10.62_CV_orthogonal.csv';
data = readtable(file_path);

SNR_values = data.Var1;
model = data.Model;
music = data.Music;
toeplitz_music = data.Toeplitz_Music;

figure;

plot(SNR_values, model, '-o', 'DisplayName', 'Proposed', 'LineWidth', 2, 'Color', '#D62728', 'Marker', 'o'); 
hold on;

plot(SNR_values, music, '-s', 'DisplayName', 'MUSIC', 'LineWidth', 2, 'Color', [0.9290, 0.6940, 0.1250], 'Marker', 's'); 
plot(SNR_values, toeplitz_music, '--+', 'DisplayName', 'Toeplitz-MUSIC', 'LineWidth', 2, 'Color', '#2CA02C', 'Marker', '+');

legend show;
legend('Location', 'best');
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Orthogonality Metric', 'FontSize', 12);

set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'FontSize', 12);
box('on');
xticks(-9:2:9); 
xlim([-9, 9]); 
ylim([0, 1]); 

set(gca, 'Box', 'on', 'LineWidth', 1, 'XColor', 'k', 'YColor', 'k'); 
grid on;
