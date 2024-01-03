function [Y_hat, U_hat, G_hat, delta_G_sqr ] = DeltaYU(y, u)

[row_y, col_y] = size( y );
[row_u, col_u] = size( u );

p = row_y; % or row_u

for i = 1:p % or row_u = p
    Y(i,:) = fft( y(i,:) );
    U(i,:) = fft( u(i,:) );
end

Y_hat = mean( Y );
U_hat = mean( U );

for i = 1:p
    Y_temp(i,:) = ( Y(i,:) - Y_hat ) .* conj( Y(i,:) - Y_hat );
    U_temp(i,:) = ( U(i,:) - U_hat ) .* conj( U(i,:) - U_hat );
    YU_temp(i,:) = (Y(i,:) - Y_hat ) .* conj( U(i,:) - U_hat );
end

delta_y_hat_sqr = mean(Y_temp) * p / (p-1);
delta_u_hat_sqr = mean(U_temp) * p / (p-1);
delta_yu_hat_sqr = mean(YU_temp) * p / (p-1);

G_hat = Y_hat ./ U_hat;

for i = 1:col_y
    delta_G_sqr(i) = ( abs(G_hat(i)) )^2 / p * ( delta_y_hat_sqr(i) / (abs( Y_hat(i) ))^2 ...
                   +delta_u_hat_sqr(i) / (abs(U_hat(i)))^2 ...
                   - 2 * real( delta_yu_hat_sqr(i) / (Y_hat(i) * conj(U_hat(i)))));
end

end

