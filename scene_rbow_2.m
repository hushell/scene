%%-------------------------------------------------------------------------
% 15 scene baseline: spatial pyramid + K-Means vocabulary + RBOW-gen
%%-------------------------------------------------------------------------
addpath funcs/
setup

if ~exist( 'allHists', 'var')
    %% make annotations
    classes = make_anno('data/myImages');

    %% create vocabulary
    voc = computeVocabularyFromImageList(classes);

    %% compute histograms of kth words seen in regions
    allHists = compute_all_hists(voc, classes);
end

%%
% select training images, this step should be randomly do 5 times
trainset = [];
trainset = logical(trainset);
labels = [];
numTrain = 100;
for i = 1:numel(classes)
    %numTrain = round(length(classes{i}.images) * trainPortion);    
    index = randperm(length(classes{i}.images));
    trainidx = index(1:numTrain);
    temp = ismember(1:length(classes{i}.images), trainidx);
    trainset = cat(2, trainset, temp);        
    labels = cat(1, labels, i.*ones(length(classes{i}.images),1));
end   
allLabels = zeros(length(trainset),2);
allLabels(:,1) = labels;
allLabels(:,2) = ~trainset; % note: the output for testing

Hists = [allHists{:}]';
testset = logical(allLabels(:,2));
trainset = ~testset; % NOTE: allLabels(:,2) is for testing

trainHists = Hists(trainset,:);
%trainHists = trainHists ./ norm(trainHists, 1); % normalize features
labels = allLabels(:,1);
trainlabels = labels(trainset);
numTrainImg = length(trainlabels);

K = 16;
R = 16;
M = 1000;
C = 15;

x = cell(numTrainImg, 1);
y = cell(numTrainImg, 1);
for i = 1:numTrainImg
    x{i} = reshape(trainHists(i,:), M, R);
    y{i} = trainlabels(i);
end

%% learning
wall_rbow = zeros(K,M,C);
aall_rbow = zeros(R,K,C);

for c = 1:C
    fprintf('***learing in category %d\n', c);
    [prio,wall_rbow(:,:,c),dump] = learn_rbow(x((c-1)*100+c:c*100), K);
    aall_rbow(:,:,c) = prio';
end


%% inference without latent variables
[probCls_rbow, aps_rbow] = infer_gen(allHists, allLabels_gen, wall_rbow, aall_rbow, classes, 100);

%% evaluation
%[recs,precs,aps] = multiclass_eval(numel(classes),probCls_gen,allLabels_gen,0);
multiclass_eval(probCls_rbow,allLabels_gen)