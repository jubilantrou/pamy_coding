function [ ] = PlotSensitivity(controller, dof, data, fs, Ns, freLimit, mode_name )
    
Td = 1/fs;
f_stamp = (0 : Ns-1) * fs/Ns;
number = length(controller);

s = tf('s');
fig = figure;
path_1 = strcat(pwd, "/data/ResponseSignal/", ...
               "Sensitivity_Pos_", mat2str(dof),"_", mode_name , ".fig");

path_2 = strcat(pwd, "/data/ResponseSignal/", ...
              "Bode_Pos_", mat2str(dof),"_", mode_name , ".fig");
          
for i = 1:number
    kp = controller(i).kp;
    ki = controller(i).ki;
    kd = controller(i).kd;
    
    R = kp + ki/s + kd * s;
               
    num = data(i).num;
    den = data(i).den;
    ndelay = data(i).ndelay;
        
    G = tf(num, den) * exp(- s * ndelay * Td) * (1 / (0.005*s + 1))^2;
        
    sens = 1 / (1 + R * G) ;
    
    path_2 = strcat(pwd, "/data/ResponseSignal/", ...
              "Nyquist_Pos_", mat2str(dof),"_", mat2str(i),"_",mode_name , ".fig");
          
    fig2 = figure
    
    nyquist(R * G)
    xlim([-2 2])
    ylim([-2 2])
    
    savefig(fig2, path_2);

    close(fig2);
    
    [mag, pha, w] = bode(sens, {1, 20 * 2 * pi});
    
     mag_est = [];
     phase_est = [];
        
     Np = length( mag(1,1,:) );

     for m = 1 : Np
         mag_est = [mag_est mag(1,1,m)];
         phase_est = [phase_est pha(1,1,m)];
     end
        
     semilogx( w, 20 * log10( mag_est ) )
     hold on
        
     xlim([w(1) w(end)])
    
     xlabel('Frequency $\omega$ in rad/s')
     ylabel('Sensitivity in dB')
   
     grid on
     
     legend("Posture_1", "Posture_2", "Posture_3", "Location", "best")
    
end
    savefig(fig, path_1);
    close(fig);
end

