function [abs,rel] = real_error(prob,x)


fast = true;

if fast
    AA = prob.M_exact_A'*prob.M_exact_A;
    BB = prob.M_exact_B'*prob.M_exact_B;
%     VB = x.V'*prob.M_exact_B;
%     UA = x.U'*prob.M_exact_A;
%     E = diag(x.sigma)*diag(x.sigma) - 2*diag(x.sigma)*VB*UA' + BB*AA;
%     trace( E )
%     abs = sqrt(trace( E ));
%     rel = abs / sqrt(trace( AA*BB ));
    
    [Q1,R1] = qr([x.U*diag(x.sigma) -prob.M_exact_A], 0);
    [Q2,R2] = qr([x.V prob.M_exact_B], 0);
    abs = norm( R1*R2', 'fro' );
    rel = abs / sqrt(trace( AA*BB ));
else
    abs = norm( x.U*diag(x.sigma)*x.V' - prob.M_exact_A*prob.M_exact_B', 'fro' );
    rel = abs / norm( prob.M_exact_A*prob.M_exact_B', 'fro' );
end