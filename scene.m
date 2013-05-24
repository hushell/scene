%%-------------------------------------------------------------------------
% 15 scene baseline: spatial pyramid + K-Means vocabulary
%%-------------------------------------------------------------------------
addpath funcs/
setup
run('vlfeat/toolbox/vl_setup.m');

%% make annotations
classes = make_anno('data/myImages');

%% create vocabulary
voc = computeVocabularyFromImageList(classes);

%% compute histograms of kth words seen in regions
allHists = compute_all_hists(voc, classes);

%% one-vs-all
[models, allScores, allLabels, aps] = svm_ova_traininng_testing(allHists, classes, 0.2, 1);

%% evaluation
%[recs,precs,aps] = multiclass_eval(numel(classes),allScores,allLabels,0);