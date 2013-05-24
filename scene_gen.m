%%-------------------------------------------------------------------------
% 15 scene baseline: spatial pyramid + K-Means vocabulary + generative
%%-------------------------------------------------------------------------
addpath funcs/
setup

%% make annotations
classes = make_anno('data/myImages');

%% create vocabulary
voc = computeVocabularyFromImageList(classes);

%% compute histograms of kth words seen in regions
allHists = compute_all_hists(voc, classes);

%% model from frequency
[wall_gen, allLabels_gen] = freq_gen(allHists, classes, 100, 1000, 15, 4);

%% inference without latent variables
[probCls_gen] = infer_gen_nolatent(allHists, allLabels_gen, wall_gen, 1000);

%% evaluation
%[recs,precs,aps] = multiclass_eval(numel(classes),probCls_gen,allLabels_gen,0);
multiclass_eval(probCls_gen,allLabels_gen)