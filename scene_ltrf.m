%%-------------------------------------------------------------------------
% LTRF feature extraction
%%-------------------------------------------------------------------------
dbstop if error

if ~exist('PATH_OK')
    addpath funcs/
    setup
    PATH_OK = 1;
end

dataset = 2; % 1: 15scene, 2:mitindoor

% modify computeHistogram.m
if ~exist( 'allHists', 'var')
    if dataset == 1
        classes = make_anno('data/myImages');
        %classes.averageSize = [-1,-1];
        % classes = pre_screen(classes);
        voc = computeVocabularyFromImageList(classes,'data');
        allHists = compute_all_code_vectors(voc, classes, 'data/myImages/');
    else
        % make annotations
        classes = make_anno('/scratch/working/git-dir/scene/indoor_data/myImages');
        %% create vocabulary or load from ./data/cache/global
        voc = computeVocabularyFromImageList(classes,'indoor_data');
        % compute histograms of kth words seen in regions
        % control partition in computeHistogram
        %allHists = compute_all_hists(voc, classes, 'indoor_data/myImages/', 'indoor_data/cache/global/');
        allHists = compute_all_code_vectors(voc, classes, '/scratch/working/git-dir/scene/indoor_data/myImages/', ...
            '/scratch/working/git-dir/scene/indoor_data/cache/global/');
    end
end