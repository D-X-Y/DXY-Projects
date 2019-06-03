function ss_build()
% Compile Selective Search. Code modified from the Selective Search IJCV release.
%
% Compile anisotropic gaussian filter
  if ~exist('anigauss')
    fprintf('Compiling the anisotropic gauss filtering of:\n');
    fprintf('   J. Geusebroek, A. Smeulders, and J. van de Weijer\n');
    fprintf('   Fast anisotropic gauss filtering\n');
    fprintf('   IEEE Transactions on Image Processing, 2003\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://staff.science.uva.nl/~mark/downloads.html#anigauss\n\n');
    mex -outdir bin ...
        selective_search/SelectiveSearchCodeIJCV/Dependencies/anigaussm/anigauss_mex.c ...
        selective_search/SelectiveSearchCodeIJCV/Dependencies/anigaussm/anigauss.c ...
        -output anigauss
  end

  if ~exist('mexCountWordsIndex')
    mex -outdir bin ...
      selective_search/SelectiveSearchCodeIJCV/Dependencies/mexCountWordsIndex.cpp ...
      -output mexCountWordsIndex
  end

  % Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
  if ~exist('mexFelzenSegmentIndex')
    fprintf('Compiling the segmentation algorithm of:\n');
    fprintf('   P. Felzenszwalb and D. Huttenlocher\n');
    fprintf('   Efficient Graph-Based Image Segmentation\n');
    fprintf('   International Journal of Computer Vision, 2004\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://www.cs.brown.edu/~pff/segment/\n');
    fprintf('Note: A small Matlab wrapper was made.\n');
    mex -outdir bin ...
        selective_search/SelectiveSearchCodeIJCV/Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp ...
        -output mexFelzenSegmentIndex;
  end

end
