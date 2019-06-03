function do_updateSval(prob,Omega,d,n)

if prob.use_blas
  updateSval_blas(Omega,d,n);
else
  updateSval(Omega,d,n);
end