function [] = DoubleCheck( data, num_rcd, den_rcd, fs, Ns );
global G G_delay G_inv des_input simIn


T_1 = 10;

t_y_des = 0 : 1/fs : T_1;

y_des_2 = 0.25 * sin(  6 * t_y_des - 0.5 * pi)  ;

y_des = y_des_2;

% x_ini = [0; 0; 0];
% 
% x_final = [y_des_2(1); 2; 0];


T_2 = 0;
% 
% [ y_des_1, y_dot_des ] = TrajGenerate(T_2, fs, x_ini, x_final);
% 
% y_des = [y_des_1(1:end-1) y_des_2];
% 
tt = 0: 1/fs : (T_1 + T_2);
% 
% tt = 0: 1/fs : (T_1 + T_2 +T_3);
% for i = 1:100
%     y_des = [y_des y_des(end)];
%     y_dot_des = [y_dot_des y_dot_des(end)];
% end
% 
% T = T + 1/fs * 100;
%% 
% T = 3;
% 
% t_y_des = 0 : 1/fs : T;
% 
% y_des = sin(  2 * t_y_des);

des_input = [t_y_des.' y_des.'];
%% 

freLimit = [0 10]; % frequency in Hz

fs = 100; % Hz

fre = (0 : length(y_des)- 1 ) / length(y_des) * fs ;

y_des = y_des .* hann(length(y_des)).';

y_fft = fft(y_des);

plot(fre, (abs(y_fft)))

xlim([0 10])
%% 
T = T_1 + T_2;
% fid = fopen("SmoothSignal.txt",'w');
% 
% fprintf(fid, '%.0f\n', T);
% fprintf(fid, '%.0f\n', length(y_des));
% 
% for i = 1:length(y_des)
%     fprintf(fid,'%.10f\n', y_des(i));
% end
% 
% fclose(fid);

%% 
delay_comp = 0;

Td = 1 / fs;

s = tf('s');

section = [20414, 19914, 19414, 18914, 18414];

l_y_des = length( y_des );

model_name = 'DoubleCheck_simulink';
Tchr = int2str( T );
set_param(model_name, 'StopTime', Tchr);
%% 

for i = 1 : length(data)
    
    pre_ago = section(i);
    pre_ant = 2 * section(1) - pre_ago;
    
    num = data(i).num;
    den = data(i).den;
    ndelay = data(i).ndelay;
    
    num_inv = num_rcd(i, :);
    den_inv = den_rcd(i, :);
    
    G_inv = tf(num_inv, den_inv);
    G = tf(num, den);
    G_delay = G * exp( - s * Td * ndelay );
    
    ff = [];
    
 
    for step = 1 : l_y_des
        
        a0 = step - 0 + ndelay + delay_comp;
        a1 = step - 1 + ndelay + delay_comp; 
        a2 = step - 2 + ndelay + delay_comp; 
        a3 = step - 3 + ndelay + delay_comp;
        a4 = step - 4 + ndelay + delay_comp;
        
        if a0 > l_y_des
            a0 = l_y_des;    
        end
        if a0 > 0
            tm0 = num_inv(1) * y_des(a0);
        else
            tm0 = 0;
        end
        
        if a1 > l_y_des
            a1 = l_y_des;    
        end
        if a1 > 0
            tm1 = num_inv(2) * y_des(a1);
        else
            tm1 = 0;
        end
            
        
        if a2 > l_y_des
            a2 = l_y_des;    
        end
        if a2 > 0
            tm2 = num_inv(3) * y_des(a2);
        else
            tm2 = 0;
        end
        
%         if a3 > l_y_des
%             a3 = l_y_des;    
%         end
%         if a3 > 0
%             tm3 = num_inv(4) * y_des(a3);
%         else
%             tm3 = 0;
%         end
            
%         if a4 > l_y_des
%             a4 = l_y_des;    
%         end
%         if a4 > 0
%             tm4 = num_inv(5) * y_des(a4);
%         else
%             tm4 = 0;
%         end
        
        if step >= 3
            term_1 = den_inv(2) * ff(step - 1);
            term_2 = den_inv(3) * ff(step - 2);
%           term_3 = den_inv(4) * ff(step - 3);
%           term_4 = den_inv(5) * ff(step - 4);
        elseif step == 2
            term_1 = den_inv(2) * ff(step - 1);
            term_2 = den_inv(3) * 0;
%             term_3 = 0; %den_inv(4) * ff(step - 3);
%            term_4 = 0;
%         elseif step == 1
%             term_1 = den_inv(2) * 0;
%             term_2 = 0; %den_inv(3) * ff(step - 2);
% %             term_3 = 0;
% %           term_4 = 0;
        else
            term_1 = 0; %den_inv(2) * ff(step - 1);
            term_2 = 0;
%             term_3 = 0;
%            term_4 = 0;
%         else
%             term_1 = 0;
%             term_2 = 0;
%             term_3 = 0;
%             term_4 = 0;
            
        end
        
        feedforward = tm0 + tm1 + tm2  - term_1 - term_2 ;
                
        ff = [ff feedforward];
        
    end
    
    simIn = [t_y_des.' ff.'];
    
    simOut = sim(model_name, 'ReturnWorkspaceOutputs', 'on');

    y_no_delay = simOut.y_1;

    y_with_delay = simOut.y_2;
    
    y_unchanged = simOut.y_3;
    %% 
    
    figure(1)
    plot(t_y_des, y_no_delay,'--','Linewidth',1)
    hold on
    plot(t_y_des, y_with_delay)
    hold on
    plot(t_y_des, y_des)
    xlim([0 T])
    legend('y without delay','y with delay','desired y','Location','best')
end

end

