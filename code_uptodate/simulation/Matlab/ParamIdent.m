function [ num, den ] = ParamIdent(G, w, n_o, d_o)
s_vec     = (1i * w).'; % the corresponding radian frequency vector
G         = G.';
len       = length(s_vec);
const_vec = - G .* ( s_vec.^d_o);

V = [];
for i = n_o:-1:0
    V = [V s_vec.^i];
end

for i = (d_o-1):-1:0
    V = [ V -G.*(s_vec.^i) ];
end

const_vec = [real(const_vec); imag(const_vec)];
V         = [real(V); imag(V)];
param     = pinv(V) * (-const_vec);
num       = param(1 : n_o+1).';
den       = [1 param(n_o+2:end).'];
%%  [a0 a1 b0 b1 b2 b3]

% s_vec = 1i * w; % the corresponding radian frequency vector
% len=length(s_vec);
% const_vec = ( - G .* ( s_vec.^4) ).';
% 
% V = [ones(len,1) s_vec.' -G.' ( -G .* s_vec).'  ( -G .* (s_vec.^2)).'  ( -G .* (s_vec.^3)).'];
% 
% const_vec = [real(const_vec); imag(const_vec)];
% V=[real(V); imag(V)];
% 
% param = pinv(V) * ( -const_vec );
%% [a0 a1 b0 b1 b2]

% s_vec = 1i * w; % the corresponding radian frequency vector
% len=length(s_vec);
% const_vec = ( - G .* ( s_vec.^3) ).';
% 
% V = [ones(len,1) s_vec.' -G.' ( -G .* s_vec).'  ( -G .* (s_vec.^2)).'  ];
% 
% const_vec = [real(const_vec); imag(const_vec)];
% V=[real(V); imag(V)];
% 
% param = pinv(V) * ( -const_vec );
%%  [a0 b0 b1 b2]

% s_vec = 1i * w; % the corresponding radian frequency vector
% len=length(s_vec);
% const_vec = ( - G .* ( s_vec.^3) ).';
% 
% V = [ones(len,1)  -G.' ( -G .* s_vec).'  ( -G .* (s_vec.^2)).'  ];
% 
% const_vec = [real(const_vec); imag(const_vec)];
% V=[real(V); imag(V)];
% 
% param = pinv(V) * ( -const_vec );
%% [a0 b0 b1]
% 
% s_vec = 1i * w; % the corresponding radian frequency vector
% len=length(s_vec);
% const_vec = ( - G .* ( s_vec.^2) ).';
% 
% V = [ones(len,1)  -G.' ( -G .* s_vec).'];
% 
% const_vec = [real(const_vec); imag(const_vec)];
% V=[real(V); imag(V)];
% 
% param = pinv(V) * ( -const_vec );
end

