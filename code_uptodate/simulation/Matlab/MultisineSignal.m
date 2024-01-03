function [ u ] = MultisineSignal(freLimit, t_stamp, f_stamp, amp, N, mode)

if strcmp(mode, 'average')
    U_amp = AmpAverage(N, f_stamp, freLimit, amp);
elseif strcmp(mode, 'exp')
    U_amp = AmpExp(N, f_stamp, freLimit, amp);
end

U_phase = PhaseGeneration(N);  % randomly generate the phase spectrum
  
U = U_amp .* exp(1i * U_phase);
u = real(ifft( U ));  % use only the real part

u_x = ifft( U );
U_x_amp = abs(fft(u));  % for double check

% figure(1);
% plot(f_stamp, U_amp, 'LineWidth', 1.5);
% hold on
% plot(f_stamp, U_x_amp, 'LineWidth', 1.5);
% ylim([0, max(U_amp)*1.5])
% xlim([f_stamp(1), f_stamp(end)/2])
% xlabel('Frequency $f$ in Hz')
% ylabel('Amplitude')
% set(gca,'LineWidth',1.5);
% set(gca,'FontSize',14);
% 
% figure(2)
% plot(t_stamp, u, 'color', [0 0.4470 0.7410], 'LineWidth', 1.5);
% xlabel('Time $t$ in s')
% ylabel('Amplitude')
% set(gca,'LineWidth',1.5);
% set(gca,'FontSize',14);
end

function U_amp = AmpAverage(N, f_stamp, freLimit, amp)
U_amp = zeros(N, 1);  % specify signal's amplitude spectrum

idx_1 = find(abs(f_stamp - freLimit(1)) < 0.00001);
idx_2 = find(abs(f_stamp - freLimit(2)) < 0.00001);

idx_vector = idx_1:idx_2;

U_amp(idx_vector) = amp;

U_amp(1) = 0;  % set the DC component
end

function U_amp = AmpExp(N, f_stamp, freLimit, amp)
U_amp = zeros(N, 1);  % specify signal's amplitude spectrum

idx_1 = find(abs(f_stamp - freLimit(1)) < 0.00001);
idx_2 = find(abs(f_stamp - freLimit(2)) < 0.00001);

idx_vector = idx_1:idx_2;

factor = 10.^(-4 * idx_vector/100);
U_amp(idx_vector) = amp  * factor;

U_amp(1) = 0;  % set the DC component
end


function U_phase = PhaseGeneration( N )
U_phase = rand(N, 1) * 2*pi;
end
