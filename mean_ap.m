function [ meanap ] = mean_ap( opts, score, testlabels )
%MEAN_AP Summary of this function goes here
%   Detailed explanation goes here
    
for cls_index=1:opts.nclasses
    
    gt=(testlabels==cls_index);   % images in test set which belong to class cls_index

    % compute precision/recall
    [so,si]=sort(-score);
    tp=gt(si)>0;
    fp=gt(si)==0;

    fp=cumsum(fp);
    tp=cumsum(tp);
    rec=tp/sum(gt>0);
    prec=tp./(fp+tp);

    % compute average precision
    ap=0;
    for t=0:0.1:1
        p=max(prec(rec>=t));
        if isempty(p)
            p=0;
        end
        ap=ap+p/11;
    end
    meanap(cls_index) = ap;
end

meanap = mean(meanap);