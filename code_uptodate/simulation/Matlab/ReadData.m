function [u_des, u_obs, position, p, M, N] = ReadData(dof, amp, if_plot, freq)
anchor_list = [17500, 20500, 16000, 15000];
anchor      = anchor_list(dof+1);

path = strcat('/home/hao/Desktop/MPI/Pamy_simulation/data/response_signal/', 're_response_dof_', mat2str(dof), "_fb", ".txt");
data = importdata(path);

idx = 4;

strategy = data(1); 
p        = data(2);  % number of periods
M        = data(3);  % number of signals
N        = data(4);  % number of points of each period
Ns       = N * M * p; 

t_stamp          = data(idx+1 : Ns+idx);
u                = data(Ns+idx+1 : 2*Ns+idx);
diff             = data(2*Ns+idx+1 : 3*Ns+idx);
des_pressure_ago = data(3*Ns+idx+1 : 4*Ns+idx);
des_pressure_ant = data(4*Ns+idx+1 : 5*Ns+idx);
obs_pressure_ago = data(5*Ns+idx+1 : 6*Ns+idx);
obs_pressure_ant = data(6*Ns+idx+1 : 7*Ns+idx);
position         = data(7*Ns+idx+1 : 8*Ns+idx);

if strategy == 1
    u_des = des_pressure_ago - anchor;
    u_obs = obs_pressure_ago - anchor;
elseif strategy == 2
    u_des = des_pressure_ago - des_pressure_ant;
    u_obs = obs_pressure_ago - obs_pressure_ant;
end

if strcmp(if_plot, 'yes')
    figure(1);
    hold on;
    plot(t_stamp(1:N), diff(1:N), '--','LineWidth', 2);
    plot(t_stamp(1:N), u_des(1:N));
    plot(t_stamp(1:N), u_obs(1:N));
    legend('input', 'desired', 'observed', 'location', 'best');
end
end
