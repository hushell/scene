function codevecs = computeCodeVectorFromImageList(vocabulary, names, cache, path)
% computeCodeVectorFromImageList: Compute code vector for multiple images
%   codevectors = computeCodeVectorFromImageList(VOCABULARY, NAMES)
%   each element of codevectors: sizedict x num visual words
%   computes the code vector of visual words for the list of image
%   paths NAMES by applying iteratively
%   computeCodeVectorFromImage().
%
%   computeCodeVectorFromImageList(VOCABULARY, NAMES, CACHE) caches
%   the results to the CACHE directory.

% Author: Shell Hu

if nargin < 4
    path = 'data/myImages';
end

if matlabpool('size') == 0
    matlabpool 4;
end

start = tic ;
codevecs = cell(1,numel(names)) ;
parfor i = 1:length(names)
%for i = 1:length(names)
  if exist(names{i}, 'file')
    fullPath = names{i} ;
  else
    %fullPath = fullfile('data',path,names{i}) ;
    fullPath = fullfile(path,names{i}) ;
  end
  if nargin > 1
    % try to retrieve from cache
    codevecs{i} = getFromCache(fullPath, cache) ;
    if ~isempty(codevecs{i}), continue ; end
  end
  fprintf('Extracting code vectors from %s (time remaining %.2fs)\n', fullPath, ...
          (length(names)-i) * toc(start)/i) ;
  codevecs{i} = computeCodeVectorFromImage(vocabulary, fullPath) ;
  if nargin > 1
    % save to cache
    if exist(cache, 'dir')
        storeToCache(fullPath, cache, codevecs{i}) ;
    end
  end
end
%codevecs = [codevecs{:}] ;

if matlabpool('size') > 0
    matlabpool close;
end

function histogram = getFromCache(fullPath, cache)
[drop, name] = fileparts(fullPath) ;
cachePath = fullfile(cache, [name '.mat']) ;
if exist(cachePath, 'file')
  data = load(cachePath) ;
  histogram = data.histogram ;
else
  histogram = [] ;
end

function storeToCache(fullPath, cache, histogram)
[drop, name] = fileparts(fullPath) ;
cachePath = fullfile(cache, [name '.mat']) ;
vl_xmkdir(cache) ;
data.histogram = histogram ;
save(cachePath, '-STRUCT', 'data') ;
