function [x, y, model] = feat_model_format(allHists, allLabels, wall, aall)
% [INPUT]
% wall: KxMxC
% aall: RxKxC
% [OUTPUT]
% w = {w1,...,wC}, wy = {By,Ay}; By = is MxK; Ay is a KxR prior matrix

Hists = [allHists{:}]';
trainset = logical(allLabels(:,2));
trainset = ~trainset; % NOTE: allLabels(:,2) is for testing

trainHists = Hists(trainset,:);
trainHists = trainHists ./ norm(trainHists, 1); % normalize features
labels = allLabels(:,1);
trainlabels = labels(trainset);
numTrainImg = length(trainlabels);

[K, M, C] = size(wall);
[R, ~, ~] = size(aall);
w = cell(C,1);

for i = 1:C
    w{i} = cell(2,1);
    w{i}{1} = wall(:,:,i)';
    w{i}{2} = aall(:,:,i)';
end

x = cell(numTrainImg, 1);
y = cell(numTrainImg, 1);
for i = 1:numTrainImg
    x{i} = reshape(trainHists(i,:), M, R);
    y{i} = trainlabels(i);
end

model.w = w;
model.M = M;
model.K = K;
model.R = R;
model.C = C;

end