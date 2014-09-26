function [shape_initial] = resetshape(bbox, shape_union)
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

shape_union = bsxfun(@minus, (shape_union), (min(shape_union)));

shape_initial = bsxfun(@times, shape_union, [(bbox(3)/width_union) (bbox(4)/height_union)]);
shape_initial = bsxfun(@plus, shape_initial, double([bbox(1) bbox(2)]));

end

