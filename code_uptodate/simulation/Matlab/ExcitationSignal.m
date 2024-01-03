%% This script is used to generate signals for identification.
clear;
clc;
%% 
% Import the latex environment.
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
%% 
% We generate the multisine signal in frequency domain, and then convert it 
% into the time domian. 
% We specify the band of the frequency, the increment and the sampling frequency.
freLimit = [0 4];
fs = 100;
f_delta = 0.1;
% Specify the number of periods and the number of signals.
% We have $m$ different signals, and each signal will we repeated $p$ periods. 
p = 10;
m = 10;
% We can get the number of points accroding to $N=\frac{f_s }{\bigtriangleup 
% f}$.
N = fs / f_delta;
% The identification duration for one signal will be $T_0 =\frac{N\cdot p-1}{f_s 
% }$.
% We can also get the stamp in time domain and frequecny domain.
t_stamp = (0 : N-1) / fs ;
f_stamp = (0 : N-1) / N * fs;
% Specify the desired amplitude for frequency and time signals, respectively.
amp_f = 4000;
amp_t = 1000;
% Specify the difference limit to avoid the transient effects between periods 
% and different signals.
difference_limit = 10;
% average: average amplitude in frequency domain
% exp:     -40 dB/dec in frequency domain
mode = 'exp';
%%
u = zeros(m, N);
for i = 1 : m
    while 1 
        u_temp = MultisineSignal(freLimit, t_stamp, f_stamp, amp_f, N, mode);
        k = max(abs(u_temp)) / amp_t;
        u_temp = u_temp / k;  % normalization
        if abs(u_temp(1) - u_temp(end)) < difference_limit
            figure(1)
            plot(t_stamp, u_temp, 'color', [0 0.4470 0.7410], 'LineWidth', 1.5);
            xlabel('Time $t$ in s')
            ylabel('Amplitude')
            set(gca,'LineWidth',1.5);
            set(gca,'FontSize',14);
            
            U_temp = abs(fft(u_temp));
            figure(2)
            plot(f_stamp, 20*log10(U_temp), 'color', [0 0.4470 0.7410], 'LineWidth', 1.5);
            xlabel('Frequency $f$ in Hz')
            ylabel('Amplitude in dB')
            xlim([0, fs/2]);
            set(gca,'LineWidth',1.5);
            set(gca,'FontSize',14);
            break;
        end  
    end
u(i, :) = u_temp;
end
%%
fid = strcat('/home/hao/Desktop/MPI/Pamy_simulation/data/excitation_signal/excitation_', mat2str(freLimit(2)), 'hz_exp.csv');
writematrix(u, fid);
