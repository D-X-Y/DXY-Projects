function d = do_partXY(prob,X,Y,i,j,m)

if prob.use_blas
  d = partXY_blas(X,Y,i,j,m);
else
  d = partXY(X,Y,i,j,m);
end