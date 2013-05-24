function vocabulary = computeVocabularyFromImageList(classes, rootDir, force)
% COMPUTEVOCABULARYFROMIMAGELIST Compute a visual word vocabulary
%   VOCABULARY = COMPUTEVOCABULARYFROMIMAGELIST(CLASSES) computes a
%   visual word vocabulary from all classes
%
%   VOCABULARY is a structure with fields:
%
%   WORDS:: 128 x K matrix of visual word centers.
%
%   KDTREE:: KD-tree indexing the visual word for fast quantization.

% Author: Andrea Vedaldi, Xu Hu
if nargin < 3
    force = 0;
end

if exist([rootDir,'/cache/global/voc.mat'],'file') && ~force
    fprintf('loading vocabulary from %s/cache/global/voc.mat...\n', rootDir);
    vocabulary = getfield(load([rootDir,'/cache/global/voc.mat']),'vocabulary');
    return;
end

numWords = 200 ;
numFeatures = numWords * 500 ;
% numWords = 1000 ;
% numFeatures = numWords * 100 ;

% This extracts a number of visual descriptors from the specified
% images. Only NUMFEATURES overall descriptors are retrieved as more
% do not really improve the estimation of the visual dictionary but
% slow down computation.

numImages = 0;
for k = 1:numel(classes)
   numImages = numImages + numel(classes{k}.images); 
end

if matlabpool('size') == 0
    matlabpool 4;
end

%descriptors = cell(1,numImages) ;
descriptors = {};

for k = 1:numel(classes)
    names = classes{k}.images;
    cdescriptors = cell(1,numel(names));
    
    parfor i = 1:numel(names) % ***CAUTION: no load and save 
    %for i = 1:numel(names)
       d = [];
       if exist([rootDir,'/cache/',classes{k}.name,'/',names{i},'_d','.mat'],'file')
          disp(['load features from file', classes{k}.name,'/',names{i}]);
          %d = getfield(load([rootDir,'/cache/',classes{k}.name,'/',names{i},'_d','.mat']),'d');
       else
          if exist(names{i}, 'file')
             fullPath = names{i} ;
          else
             fullPath = fullfile(rootDir,'myImages',[classes{k}.name '/'],names{i}) ;
          end
          fprintf('Extracting features from %s\n', fullPath) ;
          im = imread(fullPath) ;
          [f, d] = computeFeatures(im) ; 
          % save features
          %save([rootDir,'/cache/',classes{k}.name,'/',names{i},'_d','.mat'], 'd');
          %save([rootDir,'/cache/',classes{k}.name,'/',names{i},'_f','.mat'], 'f');
       end
       cdescriptors{i} = vl_colsubset(d, round(numFeatures / numel(names)), 'uniform') ;
    end
    
    descriptors = cat(2, descriptors, cdescriptors);
end

if matlabpool('size') > 0
    matlabpool close;
end

% This clusters the descriptors into NUMWORDS visual words by using
% KMEANS. It then compute a KDTREE to index them. The use of a KDTREE
% is optional, but speeds-up quantization significantly.

fprintf('Computing visual words and kdtree\n') ;
tic;
descriptors = single([descriptors{:}]) ;
vocabulary.words = vl_kmeans(descriptors, numWords, 'verbose', 'algorithm', 'elkan') ;
vocabulary.kdtree = vl_kdtreebuild(vocabulary.words) ;
fprintf('TIME in building vocabulary is %f', toc);

% save voc
save([rootDir,'/cache/global/voc.mat'], 'vocabulary');
