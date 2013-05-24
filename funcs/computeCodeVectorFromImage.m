function codevecs = computeCodeVectorFromImage(vocabulary,im)
% computeCodeVectorFromImage: Compute code vectors of visual words for an image
%   codevecs = computeCodeVectorFromImage(VOCABULARY,IM) compute the
%   codevecs of visual words for image IM given the visual word
%   vocaublary VOCABULARY. To do so the function calls in sequence
%   COMPUTEFEATURES(), LLC_coding(), and COMPUTECODEVECS().
%
%   See also: COMPUTEVOCABULARYFROMIMAGELIST().

% Author: Shell Hu

if isstr(im)
  if exist(im, 'file')
    fullPath = im ;
  else
    fullPath = fullfile('data','images',[im '.jpg']) ;
  end
  im = imread(im) ;
end

%width = size(im,2) ;
%height= size(im,1) ;
[keypoints, descriptors] = computeFeatures(im) ;
%words = quantizeDescriptors(vocabulary, descriptors) ;
%codevecs = computeHistogram(width, height, keypoints, words) ;

coeff = LLC_coding_appr(vocabulary.words', descriptors', 10);
coeff = coeff';
h = sum(keypoints(1,:) == keypoints(1,1));
w = size(keypoints,2) / h;

codevecs = zeros(h,w,size(coeff,1));
[hind, wind] = meshgrid(1:h, 1:w);
ind = sub2ind([h,w],hind,wind)';

for j = 1:w
    for i = 1:h
        codevecs(i,j,:) = coeff(:,ind(i,j));
    end
end
% TODO: get codevecs by reshaping
%codevecs = reshape(coeff, [h,w,size(coeff,1)])

