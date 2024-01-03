function [num_inv_dis, den_inv_dis, num_dis, den_dis] = FeedForward(data, fs)
%% This function is used to get the linear feedforward model
% Roll-off
Roff = tf([1], [0.01 1]); % with different time constant
% get the num, den and delay of the continuous forward system from system
% identification
num_con = data.num;
den_con = data.den;
ndelay_con = data.ndelay;
order_diff = data.den_order - data.num_order;

G_con = tf(num_con, den_con); % the continuous transfer function without delay
G_dis = c2d(G_con, 1/fs ,'matched'); % the discrete function without delay

% the num and den of the discrete forward transfer function
num_dis(1:length(cell2mat(G_dis.num))) = cell2mat(G_dis.num);
den_dis(1:length(cell2mat(G_dis.den))) = cell2mat(G_dis.den);

% calculate the inverse transfer function (incorporate the roll-off)
G_inv_con = tf(den_con, num_con) * Roff^order_diff;
num_inv_con = cell2mat(G_inv_con.num);
den_inv_con = cell2mat(G_inv_con.den);

% convert transfer function into state space
G_inv_dis = c2d(G_inv_con, 1/fs ,'matched');
num_inv_dis(1:length(cell2mat(G_inv_dis.num))) = cell2mat(G_inv_dis.num);
den_inv_dis(1:length(cell2mat(G_inv_dis.den))) = cell2mat(G_inv_dis.den);
end

