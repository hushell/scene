%%-------------------------------------------------------------------------
% 15 scene baseline: spatial pyramid + K-Means vocabulary + RBOW-gen
%%-------------------------------------------------------------------------
addpath funcs/
setup

%% make annotations
classes = make_anno('data/myImages');

%% create vocabulary
voc = computeVocabularyFromImageList(classes);

%% compute histograms of kth words seen in regions
allHists = compute_all_hists(voc, classes);

%% model from EM and MLE
% TODO allLabels has to be generated inside
[wall_rbow, aall_rbow] = rbow_gen_learning_par(allHists, allLabels_gen, 2000, 100, 0.0000001, 1000, 15, 4, 4);

%% inference without latent variables
[probCls_rbow, aps_rbow] = infer_gen(allHists, allLabels_gen, wall_rbow, aall_rbow, classes, 100);

%% evaluation
%[recs,precs,aps] = multiclass_eval(numel(classes),probCls_gen,allLabels_gen,0);
multiclass_eval(probCls_rbow,allLabels_gen)