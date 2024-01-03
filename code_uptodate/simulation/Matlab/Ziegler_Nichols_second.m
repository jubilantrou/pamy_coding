function [kp, ki, kd] = Ziegler_Nichols_second(data);

Ks = data.K;
T = data.Ts;
Tt = data.Td;

kp = 1.2 / Ks * T/Tt;
Tn = 2 * Tt;


end


