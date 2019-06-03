function mAP = CalculateMAP(confidence, gt)
    assert(numel(confidence) == numel(gt));
    [so,si]=sort(-confidence);
    tp=gt(si)>0;
    fp=gt(si)<0;

    fp=cumsum(fp);
    tp=cumsum(tp);
    rec=tp/sum(gt>0);
    prec=tp./(fp+tp);

    mAP=VOCap(rec,prec);
end

function ap = VOCap(rec,prec)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));

end
