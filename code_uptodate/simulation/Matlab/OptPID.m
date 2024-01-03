function [k_new, k_ini] = OptPID( data, fs )
% A * k <= b
A = [1 0 0;
     0 1 0;
     0 0 1;
     -1 0 0;
     0 -1 0;
     0 0 -1];
 
b = [100000000;
     1000000000;
     100000000;
     100000000;
     1000000000;
     100000000];

closed_loop = @(k) Fopt(k, data, fs);
% "P"
% "PI"
% "PD"
% "PID"
% "PIDso"
% "PIDno"
% "PIR"
k_ini = Ziegler(data , fs, "PID", 'yes');
[k_new fval] = fmincon(closed_loop, k_ini/1000000); %, A, b);

% for kp = -50000:1000:0 % -80000:1000:-20000
%     for ki = -30000:1000:0
%         for kd = -2000:100:0
%             k = [kp, ki, kd]
%             temp = closed_loop(k);
%             if temp < temp_min
%                 temp_min = temp;
%                 k_min = k;
%             end
%         end
%     end
% end

% for kp = -300000:1000:0
%     for ki = -5000
%         for kd = 0
%             k = [kp, ki, kd]
%             temp = closed_loop(k);
%             sen = [sen temp];
%             index = [index kp];
%         end
%     end
% end
% %% 
% 
%  figure( 1 )
%  plot(index, 20 * log10(sen), '-mo','color', [0 0.4470 0.7410], 'MarkerEdgeColor','k', 'MarkerSize', 2, 'LineWidth',1.5 )
%  xlabel('Kp')
%  ylabel('Sensitivity $s$ in dB')
%  legend("Sensitivity",  "Location", "best")
%  set(gca,'LineWidth',1.5);
%  set(gca,'FontSize',14);
 end

function [k_ini] = Ziegler(data, fs, mode_name, if_plot)

syms w
Td = 1/fs;
s = tf('s');

num = data.num;
den = data.den;
ndelay = data.ndelay;

% the continuous transfer function with delay
G = tf(num, den) * exp( -s * Td * ndelay );
G_dis = c2d(G, 1/fs, 'matched');

w = 0.001:0.01:500;
[mag_temp, pha_temp, w] = bode(G , w);
mag = [];
phase = [];
Np = length( mag_temp(1,1,:) );

for m = 1 : Np
    mag = [mag mag_temp(1,1,m)];
    phase = [phase pha_temp(1,1,m)];
end

[val, index] = min(abs(phase+180));

phase(index)
wcg = w(index);
Gm = 10^((0 - 20 * log10(mag(index)))/20);
Gm_ini = 10^((0 - 20 * log10(mag(1)))/20);
Gm = Gm - Gm_ini;


% plot
if strcmp(if_plot, 'yes')
    figure(10)
    subplot(211)
    semilogx( w, 20 * log10( mag ) ,  'LineWidth', 1.5)
    line([wcg,wcg],[-150,-40],'Color','red','LineStyle','--','LineWidth', 1.5)   
    xlim([w(1) w(end)])

    xlabel('Frequency $\omega$ in rad/s')
    ylabel('Amplitude $A$ in dB')
    set(gca,'LineWidth',1.5);
    set(gca,'FontSize',14);
    grid on

    subplot(212)
    semilogx( w, phase , 'LineWidth', 1.5)
    hold on
    line([w(1),w(end)],[-180,-180],'Color','red','LineStyle','--','LineWidth', 1.5)
    line([wcg,wcg],[min(phase), 200],'Color','red','LineStyle','--','LineWidth', 1.5)

    xlim([w(1) w(end)])
    set(gca,'LineWidth',1.5);
    set(gca,'FontSize',14);
    xlabel('Frequency $\omega$ in rad/s')
    ylabel('Phase $\phi$ in degree')
    grid on
end

kc = -abs( Gm );
Tc = 2 * pi / abs( wcg ); 
k_ini = Ziegler_Nichols(kc, Tc, mode_name);
end


function [fopt] = Fopt(k, data, fs)
Td = 1/fs;
s = tf('s');
R = k(1) + k(2)/s + k(3)*s;
W = makeweight(10^-8, 10, 10^(1/10));

num = data.num;
den = data.den;
ndelay = data.ndelay;
   
appx = 4;
G_without_delay = tf(num, den) * (1 / (0.001*s + 1))^2;
G = G_without_delay * exp(- ndelay * s * Td);

z = tf('z', 1/fs);
R_dis = k(1) + k(2) / (fs * (1 - z^(-1))) + k(3) * (1 - z^(-1)) * fs;
p_dis = c2d(G_without_delay, 1/fs, 'matched') * z^(-ndelay);
W_dis = c2d(W, 1/fs);

G_con = 1 /( 1 + R *  pade(G, appx)) * (1/W);
G_dis = 1 / (1 + R_dis * p_dis) * (1 / W_dis);
fopt = hinfnorm(G_dis);   
end

