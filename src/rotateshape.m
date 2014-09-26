function shape_rotated = rotateshape(shape)
%ROTATESHAPE Summary of this function goes here
%   Rotate input shape randomly
%   Detailed explanation goes here
%   Input:
%       shape: original shape
%   Output:
%       shape_rotated: rotated shape

anti_clockwise_angle = 120*rand(1);

shape_rotated = rotatepoints(shape,mean(shape),anti_clockwise_angle - 60, 1);

end

