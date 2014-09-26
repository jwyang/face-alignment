function drawshapes(img, shapes)
%DRAWSHAPE Summary of this function goes here
%   Function: draw face landmarks on img
%   Detailed explanation goes here
%   Input:
%       img: input image
%       shapes: given face shapes

imshow(img);
hold on;

colors = {'r.' 'g.' 'b.' 'c.' 'm.' 'y.' 'k.'};

for s = 1:size(shapes, 2)/2
for is = 1:size(shapes, 1)
    plot(shapes(is, 2*(s-1) + 1), shapes(is, 2*s), colors{s}, 'LineWidth', 1);
end
end

end

