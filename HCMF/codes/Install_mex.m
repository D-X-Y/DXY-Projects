cd Auxiliary

% Extracting a subset of entries of a matrix 
mex -largeArrayDims partXY.c

% Update a sparse matrix 
mex -largeArrayDims updateSval.c
    
% BLAS versions if needed...    
mex -largeArrayDims -lmwlapack -lmwblas partXY_blas.c
mex -v -largeArrayDims -lmwlapack -lmwblas updateSval_blas.c

cd ..