function [] = PlotSensForAllController(num, den, ndelay, fs, Ns, freLimit)

% value obtained by Ziegler-Nichols
dof = 3


pid(1, :, :) = [1.0e+06 * -0.034022819367718, 1.0e+06 * -3.379980505280133 / 10, 1.0e+06 * -0.000226032214005;
         1.0e+05 * 0.811860197990571, 1.0e+05 * 8.967282514517638 / 10 , 1.0e+05 * 0.048511598336439;
         1.0e+06 * -0.020222520550019, 1.0e+06 * -1.354350736357323 / 10, 1.0e+06 * -0.000199289013869];

% H optimization
pid(2, :, :) = [-3.505924158687806e+04, -3.484022215671791e+05 / 5, -5.665386729745434e+02;
         8.228984656729296e+04, 1.304087541343074e+04, 4.841489121599795e+02;
         -36752.24956301624, -246064.5612272051/ 10, -531.2866756516057];
     
% 1. grid
pid(3, :, :) = [-5000, -6000, 0;
         31000, 2000, 200;
         -2000, -5000, 0];
% 2. grid     
pid(4, :, :) = [-14000, -15000, -300;
         64000, 30000, 1900;
         -4000, -13000, -100];
% 3. grid
pid(5, :, :) = [-31000, -16000, -300;
         88000, 40000, 1700;
         -6000, -18000, -100];
     
   
Td = 1 / fs;

freq_stamp = ( 0 : Ns - 1) * fs / Ns;

s = tf('s');

G = tf(num, den) * exp(- s * ndelay * Td) * (1 / (0.005*s + 1))^2;

w = 0.1:0.1:100

figure(1)

for i = 1:5
    R = pid(i, dof, 1) + pid(i, dof, 2)/s + pid(i, dof, 3) * s;
    sens = 1 / (1 + R * G);
    
    [mag, pha, w] = bode(sens, w);

    mag_est = [];
    phase_est = [];

    Np = length( mag(1,1,:) );

     for m = 1 : Np
         mag_est = [mag_est mag(1,1,m)];
         phase_est = [phase_est pha(1,1,m)];
     end

     semilogx( w, 20 * log10( mag_est ), 'LineWidth', 1.5 )
     hold on
     
end
legend( "Z-N method", "Heuristic","1. Grid",  "2. Grid", "3. Grid", "Location", "best")
xlim([w(1) w(end)])
xlabel('Frequency $\omega$ in rad/s')
ylabel('Sensitivity $S$ in dB')

set(gca,'LineWidth',1.5);
set(gca,'FontSize',14);
    
grid on

end

