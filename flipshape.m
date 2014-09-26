function shape_flipped = flipshape(shape)
%FLIPSHAPE Summary of this function goes here
%   Function: flip input shape horizonically
%   Detailed explanation goes here

if size(shape, 1) == 68
    shape_flipped = shape;
    % flip check
    shape_flipped(1:17, :) = shape(17:-1:1, :);
    % flip eyebows
    shape_flipped(18:27, :) = shape(27:-1:18, :);
    % flip eyes
    shape_flipped(32:36, :) = shape(36:-1:32, :);
    % flip eyes
    shape_flipped(37:40, :) = shape(46:-1:43, :);
    shape_flipped(41:42, :) = shape(48:-1:47, :);
    shape_flipped(43:46, :) = shape(40:-1:37, :);
    shape_flipped(47:48, :) = shape(42:-1:41, :);
    
    % flip mouth
    shape_flipped(49:55, :) = shape(55:-1:49, :);
    shape_flipped(56:60, :) = shape(60:-1:56, :);   
    
    shape_flipped(61:65, :) = shape(65:-1:61, :);
    shape_flipped(66:68, :) = shape(68:-1:66, :);   
    
else
    disp('The Flip Funtion is Error!')
end

end

