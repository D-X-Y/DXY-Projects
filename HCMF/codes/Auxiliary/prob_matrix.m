function M = prob_matrix(n_1,n_2,r,m,varargin)

% creates problem matrix in different formats (struct)
% M = prob_matrix(n_of_rows,n_of_cols,rank,n_of_entries,parameters)
% paramters:
% type: '2factors'/'decaying' (string)
% noiseratio: relative noise added (Gaussian) (double)
% slope: -slope of decaying sing. values (double)
% diag: sets the diagonal directly (vector)
% scaling: scales the factors or the diagonal (double)
% spnoiseratio: percentage of entries corrupted by sparse noise (double)
% spnoiseratio: maximum value of the sparse noise (double)


options.type = '2factors';
options.noiseratio = 0;
options.scaling = 1;
options.spnoiseratio = 0;
options.spnoisevalue = 1;



if nargin > 4
    if isfield(varargin{1},'type'); options.type = varargin{1}.type; end
    if isfield(varargin{1},'noiseratio'); options.noiseratio = varargin{1}.noiseratio; end
    if isfield(varargin{1},'slope'); options.slope = varargin{1}.slope; options.diag = diag(10.^(0:-options.slope:-options.slope*(r-1))); end 
    if isfield(varargin{1},'diag'); options.diag = varargin{1}.diag; r = length(options.diag); end
    if isfield(varargin{1},'scaling'); options.scaling = varargin{1}.scaling; end
    if isfield(varargin{1},'spnoiseratio'); options.spnoiseratio = varargin{1}.spnoiseratio; end
    if isfield(varargin{1},'spnoisevalue'); options.spnoisevalue = varargin{1}.spnoisevalue; end
    if isfield(varargin{1},'decay_matrix'); options.decay_matrix = varargin{1}.decay_matrix; end
end

% Random sampling set
[M.ind_Omega,m] = randsampling(n_1,n_2,m); M.ind_Omega = M.ind_Omega';
M.ind_Omega = M.ind_Omega(1:m); 
M.ind_Omega = sort(M.ind_Omega); %SVT,cvx
[M.row,M.col] = ind2sub([n_1,n_2],M.ind_Omega); %GROUSE

M.size1 = n_1; %APGL
M.size2 = n_2; %APGL



%%% BARTVDE: determninistic matrices with decaying singular values

if strcmpi(options.type,'correlation')
    if n_1~=n_2; error('n_1 should equal n_2'); end
    sigma = options.decay_matrix.sigma;
    xx = linspace(0,1,n_1);
    M.fun = @(x,y)(1./(sigma+(x-y).^2));
    M.values_Omega = M.fun(xx(M.row),xx(M.col));
end


if strcmpi(options.type,'2factors') %random from two factor
    M.left = options.scaling*randn(n_1,r);
    M.right = options.scaling*randn(n_2,r);
    M.values_Omega = partXY(M.left',M.right',M.row,M.col,m); 
    %M.values_Omega = XonOmega(M.left,M.right,M.ind_Omega); %a bit slower
end


if strcmpi(options.type,'decaying') %random with decaying sing. values
    M.left = orth(randn(n_1,r))*options.scaling*options.diag;
    M.right = orth(randn(n_2,r));
    M.values_Omega = partXY(M.left',M.right',M.row,M.col,m); 
end





% Noise

if options.noiseratio > 0 %add Gaussian noise
    noise = randn(1,m);
    s = options.noiseratio*norm(M.values_Omega)/norm(noise);       
    M.values_Omega = M.values_Omega + s*noise; %SVT,cvx,GROUSE
end

if options.spnoiseratio > 0 %add sparse noise
    num = round(options.spnoiseratio*m);
    noisee = randsample(m,num);
    noisev = -options.spnoisevalue + 2*options.spnoisevalue*rand(1,num);
    M.values_Omega(noisee) = M.values_Omega(noisee)+noisev; 
end

M.proj  = @(X) X(M.ind_Omega)';
M.projT = @(y) sparse(M.row,M.col,y,M.size1,M.size2);

end

function [omega,m] = randsampling(n_1,n_2,m)
if m > n_1*n_2
  warning('More samples than elements')
  m = n_1*n_2;
  
  omega = 1:m;
else
  omega = ceil(rand(m, 1) * n_1 * n_2);
  omega = unique(omega);
  while length(omega) < m    
      omega = [omega; ceil(rand(m-length(omega), 1)*n_1*n_2);];
      omega = unique(omega);
  end
end
end