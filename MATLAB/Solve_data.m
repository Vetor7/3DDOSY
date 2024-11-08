clc;clear;close all;

load QGC_3D.mat

DOSYData = NmrData.SPECTRA;% DOSYData    

g=100*NmrData.Gzlvl; % gradient values
BD=NmrData.DELTAOriginal; % diffusion time
LD=NmrData.deltaOriginal; % diffusion encoding time
cs=NmrData.Specscale;     % chemical shift
gamma = 4257.7;
g2 = (2*pi*gamma*g*LD).^2*(BD-LD/3)*1e4;
b = g2*1e-10;
b_dim = length(b);

DOSYData = reshape(DOSYData', b_dim, 256, []);
DOSYData = fft(DOSYData, [], 2);
DOSYData = abs(DOSYData);
DOSYData = DOSYData / max(DOSYData(:));

DOSYData2 = reshape(DOSYData, b_dim, []);
idx_peaks = find(DOSYData2(1, :) < 0.03);
DOSYData2(:, idx_peaks)=0;

S_r = DOSYData2 (:, find(DOSYData2(1, :) >= 0.03));

DOSYData2 = reshape(DOSYData2, b_dim, 256, []);

DOSYData1 = squeeze(DOSYData2(1, :, :));

figure(1)
DOSYsum = squeeze(sum(DOSYData2, 2));
% DOSYsum = DOSYsum / max(DOSYsum(:));
idx_peaks1 = find(DOSYsum(1, :) >= 0.03);

plot(DOSYsum(1,:))

figure(2)
DOSYsum2 = squeeze(sum(DOSYData2, 3));
% DOSYsum2 = DOSYsum2 / max(DOSYsum2(:));
idx_peaks2 = find(DOSYsum2(1, :) >= 0.03);
plot(DOSYsum2(1,:))

figure(3)

contour(DOSYData1, 40);
S = DOSYData(:, idx_peaks2, idx_peaks1);
% S(2,:,:) = flipud(squeeze(S(2,:,:)));
% S(4,:,:) = flipud(squeeze(S(4,:,:)));
% S(6,:,:) = flipud(squeeze(S(6,:,:)));

ppm = cs;
whole_spec = zeros([b_dim, 256, length(ppm)]);
whole_spec(:, idx_peaks2, idx_peaks1) = S;

figure(4)
contour(squeeze(whole_spec(1,:,:)), 40);

figure(5)
contour(squeeze(S(1,:,:)), 40);

% figure(6)
% S_new = reshape(S, b_dim, []);
% S_new = S_new ./ S_new(1, :);
% plot(b, S_new)

figure(6)
S_r = S_r ./ S_r(1, :);
plot(b, S_r)

HNMR = S(1,:,:);
save('QGC_net_input.mat', 'S', 'ppm', 'b', 'idx_peaks', 'HNMR', '-mat');



