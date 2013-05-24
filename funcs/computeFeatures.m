function [keypoints,descriptors] = computeFeatures(im, binsize, step, averageSize)
% COMPUTEFEATURES Compute keypoints and descriptors for an image
%   [KEYPOINTS, DESCRIPTORS] = COMPUTEFEAUTRES(IM) computes the
%   keypoints and descriptors from the image IM. KEYPOINTS is a 4 x K
%   matrix with one column for keypoint, specifying the X,Y location,
%   the SCALE, and the CONTRAST of the keypoint.
%
%   DESCRIPTORS is a 128 x K matrix of SIFT descriptors of the
%   keypoints.

% Author: Andrea Vedaldi

if ~exist('binsize', 'var') || isempty(binsize),
    binsize = 8; % size of a SIFT bin
end
if ~exist('step', 'var') || isempty(step),
    step = 8;
end
if ~exist('averageSize', 'var') || isempty(averageSize),
    averageSize = [-1,-1];
end

im = standardizeImage(im, averageSize(1), averageSize(2)) ;
[keypoints, descriptors] = vl_phow(im, 'step', step, 'sizes', [binsize], 'floatdescriptors', true) ;
