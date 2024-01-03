function [u_des_iM, u_obs_iM, position_iM] = SeparateData(u_des, u_obs, position, p, iM, N)

u_des_temp    = u_des((iM-1)*p*N+1 : iM*p*N);
u_obs_temp    = u_obs((iM-1)*p*N+1 : iM*p*N);
position_temp = position((iM-1)*p*N+1 : iM*p*N);

u_des_iM = reshape(u_des_temp, [], p).';
u_obs_iM = reshape(u_obs_temp, [], p).';
position_iM = reshape(position_temp, [], p).';

end

