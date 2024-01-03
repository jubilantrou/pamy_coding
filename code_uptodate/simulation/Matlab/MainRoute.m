%% This script is for simulation.
clc;
clear all;
%% Import the latex environment.
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
%% set up some constants
anchor_list = [17500, 20500, 16000, 15000];
freLimit = [0 10]; % frequency in Hz
fs       = 100; % Hz
dof      = 0;
amp_list = [3];
if_plot  = 'no';
%% Get the linear transfer function.
idx = 0;
for amp = amp_list 
    anchor  = anchor_list(dof+1);
    idx     = idx + 1;
    [u_des, u_obs, position, p, M, N] = ReadData(dof, amp, if_plot, freLimit(end));   
    
    for iM = 1 : M
        [u_des_iM, u_obs_iM, position_iM] = SeparateData(u_des, u_obs, position, p, iM, N); 
        [Y_mean(iM, :), U_mean(iM, :), G(iM, :), delta_G_sqr(iM, :) ]...
        = DeltaYU(position_iM(3:end,:), u_des_iM(3:end,:));                            
    end
    
    u_abs_obs(idx, :) = u_obs_iM(5, :);
    u_abs_des(idx, :) = u_des_iM(5, :);
    p_abs(idx, :)     = position_iM(5, :);
    
    [G_BLA(idx, :), delta_nl(idx, :), delta_noise(idx, :), data(idx)]...
    = ParameterIdentification(G, delta_G_sqr, M, freLimit, fs, N, dof, amp);
    
    format long
    % get the num and den for the discrete forward and inverse model,
    % respectively
    [num_inv_dis, den_inv_dis, num_dis, den_dis] = FeedForward(data(idx), fs);
    G_inv = tf(num_inv_dis, den_inv_dis);
    G = tf(num_dis, den_dis);
    pole(G_inv)
    pole(G)
%     DoubleCheck(data, num_rcd, den_rcd, fs, Ns);
%     [k_new, k_ini] = OptPID(data(idx), fs);  
%     controller(idx).k_new = k_new;
%     controller(idx).k_ini = k_ini;
end

% for i = 1:length(amp_list)
%     fprintf('amp = %f\n', i)
%     fprintf('kp = %f\n', controller(i).k_new(1));
%     fprintf('ki = %f\n', controller(i).k_new(2));
%     fprintf('kd = %f\n', controller(i).k_new(3));
% end

% PlotSensitivity(controller, dof, data, fs, Ns, freLimit, mode_name);
PlotFigure(G_BLA, u_abs_obs, u_abs_des, p_abs, delta_nl, delta_noise, data, fs, N, freLimit);
%% 
% TransFunc = [TransFunc data];      
%% 
% path = strcat(pwd, "/figures/", ...
%               "TransFunc", ".mat");    
% save(path,'TransFunc')


