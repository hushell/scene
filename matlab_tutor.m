% This is a demo for vectorization from Bhaskar
function [r, c] = inhibitCornerFast(cim, theta, blockH, blockW, alpha)

% This function implements the surround suppression mask as described in
% Grigorescu et al. "Contour and boundary detection improved by surround suppresion of texture edges",
% Input:
% cim: Corner strength.
% theta: Gradient angles.
% blockH, blockW: Suppression mask size
% alpha: Pruning coefficient.
%
% Output:
% [r, c]: Coordinate of the pruned corner points. 

% Thinning mask size
maskS = 5;
% creation of X corodinate of the mask
% primary mask with centre = (0, 0)

indH = (blockH - 1) / 2; 
indW = (blockW - 1) / 2;

[maskX maskY] = ndgrid(-indH:indH,-indW:indW);

% maskX = [-1 0 1 -1 0 1 -1 0 1]'
% maskY = [-1 -1 -1 0 0 0 1 1 1]'

maskX = maskX(:);
maskY = maskY(:);

[row col] = size(cim);

% Image pixel coordinate in X and Y axes
% imX = [1 1 1 ... col, 2 2 2 ... col, ..., row row row ... col]'
% imY = [1 2 3 ... col, 1 2 3 ... col, ..., 1 2 3 ...col]
temp = repmat([1 : row]', 1, col);
imX = temp';
imX = imX(:);

temp = repmat([1 : col]', 1, row);
imY = temp(:);
clear temp;

% Repeat each element of imX, length(maskX) times

temp = repmat(imX, 1, length(maskX))';
imXFinal = temp(:);

% Repeat each element of imY, length(maskY) times
temp = repmat(imY, 1, length(maskY))';
imYFinal = temp(:);
clear temp;

% Actual coordinate

imXFinal = imXFinal + repmat(maskX, length(imX), 1) + indH;
imYFinal = imYFinal + repmat(maskY, length(imY), 1) + indW;

clear imX;
clear imY;

% Zero padding

startIndexH = 1 + (blockH - 1)/2;
startIndexW = 1 + (blockW - 1)/2;

newCIm = zeros(size(cim, 1) + (blockH -1), size(cim, 2) + (blockW - 1));

endIndexH = size(newCIm, 1) - (blockH - 1)/2;
endIndexW = size(newCIm, 2) - (blockW - 1)/2;

newCIm(startIndexH : endIndexH, startIndexW : endIndexW) = cim;

newTheta = zeros(size(theta, 1) + (blockH -1), size(theta, 2) + (blockW - 1));
newTheta(startIndexH : endIndexH, startIndexW : endIndexW) = theta;

[indexImData] = sub2ind(size(newCIm), imXFinal, imYFinal);

% Preparing data
theta = theta';
theta = repmat(theta(:), 1, length(maskX))';
theta = theta(:);

angleDiff = abs(newTheta(indexImData) - theta);

clear theta;
clear newTheta;

cosTerm = abs(cos(angleDiff));

weightCosSum = sum(reshape(cosTerm, length(maskX), []));

inhibitFactor = newCIm(indexImData) .* cosTerm;

clear newCIm;

inhibitFactor = sum(reshape(inhibitFactor, length(maskX), []))./weightCosSum;

inhibitFactor = reshape(inhibitFactor, col, [])';

% compute the supression value
inhibitFactor = (cim - (alpha*inhibitFactor));
