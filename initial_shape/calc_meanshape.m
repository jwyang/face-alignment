function mean_shape = calc_meanshape(shapepathlistfile)

fid = fopen(shapepathlistfile);
shapepathlist = textscan(fid, '%s', 'delimiter', '\n');

if isempty(shapepathlist)
    error('no shape file found');
    mean_shape = [];
    return;
end

shape_header = loadshape(shapepathlist{1}{1});

if isempty(shape_header)
    error('invalid shape file');
    mean_shape = [];
    return;
end

mean_shape = zeros(size(shape_header));

num_shapes = 0;
for i = 1:length(shapepathlist{1})
    shape_i = double(loadshape(shapepathlist{1}{i}));
    if isempty(shape_i)
        continue;
    end
    shape_min = min(shape_i, [], 1);
    shape_max = max(shape_i, [], 1);
    
    % translate to origin point
    shape_i = bsxfun(@minus, shape_i, shape_min);
    
    % resize shape
    shape_i = bsxfun(@rdivide, shape_i, shape_max - shape_min);
    
    mean_shape = mean_shape + shape_i;
    num_shapes = num_shapes + 1;
end

mean_shape = mean_shape ./ num_shapes;


img = 255 * ones(500, 500, 3);

drawshapes(img, 50 + 400 * mean_shape);

end

function shape = loadshape(path)
% function: load shape from pts file
file = fopen(path);
if file == -1
    shape = [];
    fclose(file);
    return;
end
shape = textscan(file, '%d16 %d16', 'HeaderLines', 3, 'CollectOutput', 2);
fclose(file);
shape = shape{1};
end