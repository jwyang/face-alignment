function shape_scaled = scaleshape(shape, scale)
%SCALESHAPE Summary of this function goes here
%   Function: scale input shape using a scale ratio
%   Detailed explanation goes here
%   Input:
%        bbox: the bbox of current sample
%        shape: input shape
%        scale: scale ratio
%   Output:
%        shape_scaled: scaled shape

shape_scaled = bsxfun(@plus, scale*(bsxfun(@minus, shape, mean(shape))), mean(shape));


end

