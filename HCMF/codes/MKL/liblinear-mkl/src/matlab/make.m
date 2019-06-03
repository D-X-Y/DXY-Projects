% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
function make()
cd ..
mex -largeArrayDims -c linear.cpp
mex -largeArrayDims -c tron.cpp
cd blas
mex -largeArrayDims -c *.c
cd ../matlab
mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmread.c
mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmwrite.c
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims train.c linear_model_matlab.c ../linear.o ../tron.o ../blas/*.o
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims predict.c linear_model_matlab.c ../linear.o ../tron.o ../blas/*.o
%{
try
	% This part is for MATLAB
	% Add -largeArrayDims on 64-bit machines of MATLAB
    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmread.c
    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmwrite.c
    mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims train.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
    mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims predict.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
catch err
	fprintf('Error: %s failed (line %d)\n', err.stack(1).file, err.stack(1).line);
	disp(err.message);
	fprintf('=> Please check README for detailed instructions.\n');
end
%}
