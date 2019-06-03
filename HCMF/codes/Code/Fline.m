function f = Fline(prob,t,Y,H)
% FLINE	the objective function, F, along the geodesic passing through Y
%	in the direction H a distance t.
%
%	f= FLINE(t,Y,H)
% basic window dressing for the fmin function
f = zeros(size(t));
for i = 1:length(t)
    f(i) = F(prob, moveEIG(prob, Y,H,t(i)) );
end