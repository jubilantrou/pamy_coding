function [ var ] = CalVariance(G, G_est, w)
[mag, pha, omega] = bode(G_est, w);

mag_est = [];
phase_est = [];
Np = length( mag(1,1,:) );

for i = 1:Np
    mag_est   = [mag_est mag(1,1,i)];
    phase_est = [phase_est pha(1,1,i)];
end

mag_diff   = 20 * abs( log10( abs( G ) ) - log10( mag_est ) ) ;
pha_diff   = abs( phase(G) * 360 / (2 * pi)- phase_est);
% pha_diff = abs( phase(G) - phase_est * 2 * pi / 360 );
var        = sqrt( sum(mag_diff.^2) / ( length(mag_diff)) + sum(pha_diff.^2) / length(pha_diff) );
% var      = log10( max(abs(mag_diff)) * max(abs(pha_diff)) );
end

