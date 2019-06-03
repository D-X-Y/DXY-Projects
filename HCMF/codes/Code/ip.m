function i = ip(x,h1,h2)
%% IP	computes the inner product of h1,h2 which are tangents at x
%
%	i = IP(x,h1,h2)
%	
%   h1 and h2 must be tangent on x (not checked)
%
%%

%i = trace( h1.M*h2.M' ) + trace( h1.Up'*h2.Up ) + trace( h1.Vp'*h2.Vp );

i = h1.M(:)'*h2.M(:) + h1.Up(:)'*h2.Up(:) + h1.Vp(:)'*h2.Vp(:);
