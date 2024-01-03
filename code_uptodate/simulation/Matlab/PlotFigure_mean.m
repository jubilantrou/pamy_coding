function [ ] = PlotFigure_mean( G_BLA, G_mean, data_mean, fs, Ns, freLimit)
    
Td = 1 / fs;

freq_stamp = ( 0 : Ns - 1) * fs / Ns;

[row col] = size(G_BLA);

for i = 1 : row
    phase_G_BLA(i, :) = phase(G_BLA(i,:)) * 360 / (2*pi);
end

number = length(data_mean);

dof = data_mean(1).dof; 

for i = 1:number
    
    num = data_mean(i).num;
    den = data_mean(i).den;
    ndelay = data_mean(i).ndelay;
    
    path = strcat(pwd, "/figures/", ...
              "Bode_mean_dof_", mat2str(dof), "_", mat2str(i), ".fig");
              
    s = tf('s');
    
    G_est = tf(num, den) * exp(- s * ndelay * Td);

    [mag, pha, w] = bode(G_est, {1e-4, freLimit(2) * 2 * pi});
    
    mag_est = [];
    phase_est = [];
    Np = length( mag(1,1,:) );

    for j = 1 : Np
        mag_est = [mag_est mag(1,1,j)];
        phase_est = [phase_est pha(1,1,j)];
    end
    
    
    
    fig = figure(i+1);
    
    subplot(2,1,1);
    semilogx(freq_stamp * 2 * pi, 20 * log10( abs( G_BLA ) ), 'x' )
    % semilogx(freq_stamp * 2 * pi, 20 * log10( abs( G_BLA((i-1)*6+1 : i*6 ,:) ) ), 'x' )
    hold on
    semilogx(w, 20 * log10( mag_est) )
    xlim([w(1) w(end)])
    
    xlabel('Frequency $\omega$ in rad/s')
    ylabel('Amplitude in dB')
    
    title(strcat("dof_",mat2str(dof)));
    grid on

    subplot(2,1,2);
    semilogx(freq_stamp * 2 * pi,  phase_G_BLA, 'x' )
    % semilogx(freq_stamp * 2 * pi,  phase_G_BLA((i-1)*6+1 : i*6 ,:), 'x' )
    hold on
    semilogx(w, phase_est)
    xlim([w(1) w(end)])
    xlabel('Frequency $\omega$ in rad/s')
    ylabel('Phase in degree')
    
    grid on
    
    savefig(fig, path);

    close(fig);
    
end

end

