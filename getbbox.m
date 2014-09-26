function bbox = getbbox(shape)
%GETBBOX Summary of this function goes here
%   Function: get the bounding box of given shape
%   Detailed explanation goes here
%   Input:
%      shape: the shape of face
%   Output:
%      bbox: the bounding box of given face shape


bbox = zeros(1, 4);

left_x   = min(shape(:, 1));
right_x  = max(shape(:, 1));
top_y    = min(shape(:, 2));
bottom_y = max(shape(:, 2));

bbox(1)  = left_x;
bbox(2)  = top_y;
bbox(3)  = right_x - left_x + 1;
bbox(4)  = bottom_y - top_y + 1;

end

