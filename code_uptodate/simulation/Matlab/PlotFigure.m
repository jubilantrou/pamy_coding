function [ ] = PlotFigure(G_BLA, u_abs_obs, u_abs_des, p_abs, delta_nl, delta_noise, data, fs, Ns, freLimit)
anchor_list = [17500, 18500, 16000, 15000];
dof         = data(1).dof; 
anchor      = anchor_list(dof+1);
Td          = 1 / fs;
freq_stamp  = (0 : Ns-1) * fs / Ns;
[number, l] = size(u_abs_obs);
t_stamp     = (0:l-1)*Td;



for i = 1:number
    num    = data(i).num;
    den    = data(i).den;
    ndelay = data(i).ndelay;
    amp    = data(i).amp;
          
    s             = tf('s');
    G_est         = tf(num, den) * exp(- s * ndelay * Td);
    [mag, pha, w] = bode(G_est, {0.0, freLimit(2) * 2 * pi});
    
    mag_est = [];
    phase_est = [];
    Np = length( mag(1,1,:) );

    for j = 1 : Np
        mag_est = [mag_est mag(1,1,j)];
        phase_est = [phase_est pha(1,1,j)];
    end
    fig = figure(1);
    subplot(4, number, i);
    hold on;
    plot(freq_stamp * 2 * pi, 20 * log10( abs( G_BLA(i,:) ) ), 'x' )
    plot(freq_stamp * 2 * pi, 20 * log10( sqrt (delta_nl(i,:) ) ), 'o' )
    plot(freq_stamp * 2 * pi, 20 * log10( sqrt( delta_noise(i,:) )  ), '^' )
    plot(w, 20 * log10( mag_est) )
    xlim([0.0 w(end)])
    set(gca,'xscale','log')
    xlabel('Frequency $\omega$ in rad/s')
    ylabel('Amplitude in dB')
    legend('$G_{BLA}$','$\delta_{nonlinear}^{2}$','$\delta_{noise}^{2}$','$G_{est}$','Location','best')
    
    title(strcat("dof_",mat2str(dof),"_amp_", mat2str(amp)));
    grid on
    hold off;

    subplot(4, number, i+number);
    hold on;
    plot(freq_stamp * 2 * pi, phase(G_BLA(i,:)) * 360 / (2*pi), 'x' )
    plot(w, phase_est)
    xlim([0.0 w(end)])
    set(gca,'xscale','log')
    xlabel('Frequency $\omega$ in rad/s')
    ylabel('Phase in degree')
    legend('measured data','estimated data','Location','best')
    grid on
    hold off;
    
    subplot(4, number, i+2*number);
    hold on
    plot(t_stamp, u_abs_obs(i, :));
    plot(t_stamp, u_abs_des(i, :));
    line([0, 10],[anchor-amp*1000,anchor-amp*1000]);
    line([0, 10],[anchor+amp*1000,anchor+amp*1000]);
    xlim([t_stamp(1) t_stamp(end)]);
    
    xlabel('Time $t$ in s')
    ylabel('Normalized pressure')
    legend('observed','desired','Location','best')
    grid on
    hold off;
    
    subplot(4, number, i+3*number);
    hold on
    plot(t_stamp, p_abs(i, :)*180/pi);
    line([0, 10],[90, 90]);
    line([0, 10],[-90, -90]);
    xlim([t_stamp(1) t_stamp(end)]);
    xlabel('Time $t$ in s')
    ylabel('Angle $\theta$ in degree')
    grid on
    hold off;

end
end
