function Te_Data = globalprediction_nn(binaryfeatures, W, Te_Data, params, stage)
%GLOBALREGRESSION Summary of this function goes here
%   Function: implement global regression given binary features and
%   groundtruth shape
%   Detailed explanation goes here
%   Input:
%        binaryfeatures: extracted binary features from all samples (N X d  )
%        Te_Data: test data
%        params: parameters for model
%   Output:
%        W: regression matrix

% organize the groundtruth shape
dbsize      = length(Te_Data);
gtshapes = zeros(dbsize*(params.augnumber + 1), 2*size(params.meanshape, 1));  % concatenate 2-D coordinates into a vector (N X (2*L))
dist_pupils = zeros(dbsize*(params.augnumber + 1), 1);

for i = 1:dbsize*(params.augnumber + 1)
    k = floor((i-1)/(params.augnumber + 1)) + 1;
    s = mod(i-1, (params.augnumber + 1)) + 1;    
    shape_gt = Te_Data{k}.shape_gt;
    gtshapes(i, :) = shape_gt(:)';
    
    % left eye: 37-42
    % right eye: 43-48
    dist_pupils(i) = norm((mean(shape_gt(37:42, :)) - mean(shape_gt(43:48, :))));
    
end

% Predict the location of lanmarks using current regression matrix
% deltashapes_bar = binaryfeatures*W;

% Predict the location of lanmarks using neural network
[deltashapes_bar, ~] = nntest_reg(W, binaryfeatures);


predshapes = zeros(dbsize*(params.augnumber + 1), 2*size(params.meanshape, 1));  % concatenate 2-D coordinates into a vector (N X (2*L))

for i = 1:dbsize*(params.augnumber + 1)
    k = floor((i-1)/(params.augnumber + 1)) + 1;
    s = mod(i-1, (params.augnumber + 1)) + 1;
    
    shapes_stage = Te_Data{k}.intermediate_shapes{stage};
    shape_stage = shapes_stage(:, :, s);
    
    deltashapes_bar_xy = reshape(deltashapes_bar(i, :), [uint8(size(deltashapes_bar, 2)/2) 2]);

    % transform above delta shape into the coordinate of current intermmediate shape
    delta_shape_intermmed_coord = tforminv(Te_Data{k}.tf2meanshape{s}, deltashapes_bar_xy(:, 1), deltashapes_bar_xy(:, 2));
    
    delta_shape_meanshape_coord = bsxfun(@times, delta_shape_intermmed_coord, [Te_Data{k}.bbox.width Te_Data{k}.bbox.height]);
    
        
    shape_newstage = shape_stage + delta_shape_meanshape_coord;        
    
    predshapes(i, :) = shape_newstage(:)';
    
    Te_Data{k}.intermediate_shapes{stage+1}(:, :, s) = shape_newstage;
    
    % update transformation of current intermediate shape to meanshape
    meanshape_resize = resetshape(Te_Data{k}.bbox, params.meanshape);        
    Te_Data{k}.tf2meanshape{s} = cp2tform(bsxfun(@minus, Te_Data{k}.intermediate_shapes{stage+1}(:,:, s), mean(Te_Data{k}.intermediate_shapes{stage+1}(:,:, s))), ...
        bsxfun(@minus, meanshape_resize, mean(meanshape_resize)), 'nonreflective similarity');
    
    
    if stage > 10
        % [Te_Data{k}.shape_gt Te_Data{k}.intermediate_shapes{1}(:,:, s) shape_newstage]
        drawshapes(Te_Data{k}.img_gray, [Te_Data{k}.shape_gt shape_newstage]);        
        close;
    end
    
end

Delta = gtshapes-predshapes;
Delta_x = Delta(:, 1:int8(size(deltashapes_bar, 2)/2));
Delta_y = Delta(:, uint8(size(deltashapes_bar, 2)/2)+1:end);

MRSE = 100*mean(mean(sqrt(Delta_x.^2+Delta_y.^2), 2)./dist_pupils);
MRSE_display = sprintf('Mean Root Square Error for %d Test Samples: %f', (dbsize*(params.augnumber + 1)), MRSE);
disp(MRSE_display);

end

