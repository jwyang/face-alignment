function rotated_coords = rotatepoints(input_XY,center,anti_clockwise_angle, scale, varargin)
degree = 1; %Radians : degree = 0; Default is calculations in degrees

% Process the inputs
if length(varargin) ~= 0
    for n = 1:1:length(varargin)
        if strcmp(varargin{n},'degree') 
            degree = 1;
        elseif strcmp(varargin{n},'radians')
            degree = 0;
        end
    end
    clear n;
end
[r,c] = size(input_XY);
if c ~= 2
    error('Not enough columns in coordinates XY ');
end
[r,c] = size(center);
if (r~=1 & c==2) | (r==1 & c~=2)
    error('Error in the size of the "center" matrix');
end

% Format the coordinate of the center of rotation
center_coord = input_XY;
center_coord(:,1) = center(1);
center_coord(:,2) = center(2);

% Turns the angles given to be such that the +ve is anti-clockwise and -ve is clockwise
anti_clockwise_angle = -1*anti_clockwise_angle;
% if in degrees, convert to radians because that's what the built-in functions use. 
if degree == 1 
    anti_clockwise_angle = deg2rad(anti_clockwise_angle);
end

%Produce the roation matrix
rotation_matrix = [cos(anti_clockwise_angle),-1*sin(anti_clockwise_angle);...
                   sin(anti_clockwise_angle),cos(anti_clockwise_angle)];
%Calculate the final coordinates
rotated_coords = scale*((input_XY-center_coord) * rotation_matrix) + center_coord;

end

