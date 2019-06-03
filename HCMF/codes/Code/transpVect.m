function ht = transpVect(prob, x,h,d, type)
%TRANSPVECT  transport the vector (x,h) along the direction of d
% type == 0: transpVect(x,h,d)
%    we transport by projecting h orth. on R_x(d)
% type == 1: transpVect(x,h,z)
%    we transport by projecting h orth. on z
% returns ht, the transported h

%% z is the foot of ht
if nargin == 4
    type = 0;
end
if type == 0    
    z = moveEIG(prob,x,d,1);
else % type =1
    z = d;
end
%% the given vector is (x,h) and is transported onto d

ip_old = ip(x,h,h);

%% h1 is x.U*h.M*x.V'
M1 = (z.U'*x.U)*h.M*(x.V'*z.V);
%Up1 = x.U*(h.M *(x.V'*z.V));
%Vp1 = x.V*(h.M'*(x.U'*z.U));

%% h2 is h.Up*x.V'
%M2 = (z.U'*h.Up)*(x.V'*z.V);
Up2 = h.Up*(x.V'*z.V);
%Vp2 = x.V*(h.Up'*z.U);

%% h2 is x.U*h.Vp'
%M3 = (z.U'*x.U)*(h.Vp'*z.V);
%Up3 = x.U*(h.Vp'*z.V);
Vp3 = h.Vp*(x.U'*z.U);

%% result
% ht.M = M1 + M2 + M3;
% ht.Up = Up1 + Up2 + Up3;
% ht.Up = ht.Up - z.U*(z.U'*ht.Up);
% %ht.Up = ht.Up - z.U*(z.U'*ht.Up);
% ht.Vp = Vp1 + Vp2 + Vp3;
% ht.Vp = ht.Vp - z.V*(z.V'*ht.Vp);
% %ht.Vp = ht.Vp - z.V*(z.V'*ht.Vp);

ht.M = M1;
ht.Up = Up2;
ht.Up = ht.Up - z.U*(z.U'*ht.Up);
ht.Vp = Vp3;
ht.Vp = ht.Vp - z.V*(z.V'*ht.Vp);

ip_new = ip(z,ht,ht);
ht = scaleTxM(ht, ip_old/ip_new);
