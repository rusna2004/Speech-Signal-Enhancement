% Load the noisy speech signal
x_noisy = load('Shaikh Rusna Aiyubbhai - speech_noisy.txt');

% Plot original spectrogram
fs = 8000; % Sampling rate (8 kHz)
windowLength = 160; % Window length
overlap = windowLength / 2; % 50% overlap
nfft = windowLength;

% Compute STFT
[S, f, t] = spectrogram(x_noisy, hamming(windowLength), overlap, nfft, fs);

% Noise estimation using MSNE
N_frames = size(S, 2);
N_freq_bins = size(S, 1);
Rxx = zeros(N_freq_bins, N_frames);
alpha = 0.85;

% Initialize the first frame of Rxx
Rxx(:, 1) = (1 - alpha) * abs(S(:, 1)).^2;

for n = 2:N_frames
    Rxx(:, n) = alpha * Rxx(:, n - 1) + (1 - alpha) * abs(S(:, n)).^2;
end

S_squared = abs(S).^2;

% Define r as a fraction of N_frames
r_fraction = 0.2; % 20% of total frames
r = round(r_fraction * N_frames);

beta = 1.5;
N_est_squared = zeros(N_freq_bins, N_frames);

for n = 1:N_frames
    start_frame = max(1, n - r);
    Rxx_min = min(Rxx(:, start_frame:n), [], 2);
    N_est_squared(:, n) = min(beta * Rxx_min, S_squared(:, n));
end

% Ephraim-Malah Weighting
a = 0.98; % a priori factor
b = 1 - a; % a posteriori factor
priori_SNR = a * S_squared ./ N_est_squared;
EM_weight = (a * priori_SNR) ./ (1 + b * priori_SNR);

% Apply Ephraim-Malah weighting
weighted_S = S .* sqrt(EM_weight);

% Inverse STFT
filtered_x = istft(weighted_S, hamming(windowLength), overlap, nfft, fs);

% Plot original and filtered signals
figure;

% Original signal
subplot(2, 1, 1);
plot(x_noisy);
title('Original Speech Signal');
xlabel('Sample');
ylabel('Amplitude');

% Filtered signal
subplot(2, 1, 2);
plot(filtered_x);
title('Filtered Speech Signal');
xlabel('Sample');
ylabel('Amplitude');

soundsc(filtered_x, fs);

function x = istft(S, window, overlap, nfft, fs)
    xlen = (size(S,2)-1)*overlap + nfft;
    x = zeros(xlen, 1);
    for n = 1:size(S,2)
        start_idx = (n-1)*overlap + 1;
        end_idx = start_idx + nfft - 1;
        x_seg = real(ifft(S(:,n), nfft));
        x(start_idx:end_idx) = x(start_idx:end_idx) + x_seg .* window;
    end
    x = x(1:xlen-nfft+1) / sum(window.^2); % Scaling
end