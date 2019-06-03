function o = iou(boxA, boxB)
  assert (size(boxA, 2) == 4);
  assert (size(boxB, 2) == 4);
  areaA = (boxA(:,3)-boxA(:,1)+1) .* (boxA(:,4)-boxA(:,2)+1);
  areaB = (boxB(:,3)-boxB(:,1)+1) .* (boxB(:,4)-boxB(:,2)+1);
  xx1 = max(boxA(:,1), boxB(:,1));
  yy1 = max(boxA(:,2), boxB(:,2));
  xx2 = min(boxA(:,3), boxB(:,3));
  yy2 = min(boxA(:,4), boxB(:,4));
  w   = max(0.0, xx2-xx1+1);
  h   = max(0.0, yy2-yy1+1);
  inter = w .* h;
  o   = inter ./ (areaA + areaB - inter);
end
