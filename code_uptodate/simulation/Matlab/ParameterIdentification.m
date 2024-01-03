function [G_BLA, delta_nl, delta_noise, data] = ParameterIdentification( G, delta_G_sqr, M,...
                                             freLimit, fs, N, dof, amp)
                                         
G_BLA       = mean(G);
G_BLA(1)    = G_BLA(2);
delta_nl    = mean((abs(G - G_BLA)).^2) / (M-1);
delta_noise = mean(delta_G_sqr) / M;

Td = 1 / fs;

freq_stamp = (0 : N-1) * fs / N;
idx        = find(freq_stamp == freLimit(2));
w          = freq_stamp(1: idx) * 2 * pi;

rcd_idx = 0;

for ndelay = 0 : 5
    for den_order = 1 : 3
        for num_order = 0 : den_order

        G_ident    = G_BLA(1: idx) .* exp(Td * ndelay * w * 1i);  % compensate the delay
        [num, den] = ParamIdent(G_ident, w, num_order, den_order);
           
        s     = tf('s');
        G_est = tf(num, den) * exp(-s * Td * ndelay);
        [var] = CalVariance(G_BLA(1: idx), G_est, w);
        
        rcd_idx                   = rcd_idx + 1;
        nm_rcd(rcd_idx).num       = num;
        nm_rcd(rcd_idx).den       = den;
        nm_rcd(rcd_idx).num_order = num_order;
        nm_rcd(rcd_idx).den_order = den_order;
        nm_rcd(rcd_idx).ndelay    = ndelay;
        nm_rcd(rcd_idx).var       = var;
        var_rcd(rcd_idx)          = var;         
        end
    end
end

[var, index] = min(var_rcd);

data         = nm_rcd(index);
data.dof     = dof;
data.amp     = amp;
end

