function [shape_initial] = translateshape(meanshape, shape_union)
%RESETSHAPE Summary of this function goes here
%   Function: reset the initial shape according to the groundtruth shape and union shape for all faces
%   Detailed explanation goes here
%   Input: 
%       bbox: bbounding box of groundtruth shape
%       shape_union: uniionshape
%   Output:
%       shape_initial: reset initial shape
%       bbox: bounding box of face image

% get the bounding box according to the ground truth shape
width_union = (max(shape_union(:, 1)) - min(shape_union(:, 1)));
height_union = (max(shape_union(:, 2)) - min(shape_union(:, 2)));

width_meanshape = (max(meanshape(:, 1)) - min(meanshape(:, 1)));
height_meanshape = (max(meanshape(:, 2)) - min(meanshape(:, 2)));

shape_union = bsxfun(@minus, (shape_union), (min(shape_union)));
% get the center point of union shape
shape_center = mean(shape_union)./[width_union, height_union];

shape_initial = bsxfun(@plus, meanshape, [width_meanshape*(shape_center(1)-0.5) height_meanshape*(shape_center(2)-0.5)]);

end

