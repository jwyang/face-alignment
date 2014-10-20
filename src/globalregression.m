function [W, Tr_Data] = globalregression(binaryfeatures, Tr_Data, params, stage)
%GLOBALREGRESSION Summary of this function goes here
%   Function: implement global regression given binary features and
%   groundtruth shape
%   Detailed explanation goes here
%   Input:
%        binaryfeatures: extracted binary features from all samples (N X d  )
%        Tr_Data: training data
%        params: parameters for model
%   Output:
%        W: regression matrix

% organize the groundtruth shape
dbsize      = length(Tr_Data);
deltashapes = zeros(dbsize*(params.augnumber), 2*size(params.meanshape, 1));  % concatenate 2-D coordinates into a vector (N X (2*L))
dist_pupils = zeros(dbsize*(params.augnumber), 1);

for i = 1:dbsize*(params.augnumber)
    k = floor((i-1)/(params.augnumber)) + 1;
    s = mod(i-1, (params.augnumber)) + 1;
    
    shape_gt = Tr_Data{k}.shape_gt;    
    if size(shape_gt, 1) == 68
        dist_pupils(i) = norm((mean(shape_gt(37:42, :)) - mean(shape_gt(43:48, :))));
    elseif size(shape_gt, 1) == 51
        dist_pupils(i) = norm((mean(shape_gt(20, :)) - mean(shape_gt(29, :))));
    elseif size(shape_gt, 1) == 29
        dist_pupils(i) = norm((mean(shape_gt(9:2:17, :)) - mean(shape_gt(10:2:18, :))));        
    end
    
    delta_shape = Tr_Data{k}.shapes_residual(:, :, s);
    deltashapes(i, :) = delta_shape(:)';
end

% conduct regression using libliear
% X : binaryfeatures
% Y:  gtshapes

param = sprintf('-s 12 -p 0 -c %f -q heart_scale', 1/(size(binaryfeatures, 1)));
W_liblinear = zeros(size(binaryfeatures, 2), size(deltashapes, 2));
tic;
parfor o = 1:size(deltashapes, 2)
    model = train(deltashapes(:, o), sparse(binaryfeatures), param);
    W_liblinear(:, o) = model.w';
end
toc;
W = W_liblinear;

% Predict the location of lanmarks using current regression matrix

deltashapes_bar = binaryfeatures*W;
deltashapes_bar_x = deltashapes_bar(:, 1:int8(size(deltashapes_bar, 2)/2));
deltashapes_bar_y = deltashapes_bar(:, uint8(size(deltashapes_bar, 2)/2)+1:end);

Delta = deltashapes_bar-deltashapes;


Delta_x = Delta(:, 1:int8(size(deltashapes_bar, 2)/2));
Delta_y = Delta(:, uint8(size(deltashapes_bar, 2)/2)+1:end);

Delta_x_imgcoord = zeros(size(Delta_x));
Delta_y_imgcoord = zeros(size(Delta_y));
for i = 1:dbsize*(params.augnumber)
    k = floor((i-1)/(params.augnumber)) + 1;
    s = mod(i-1, (params.augnumber)) + 1;
    Delta_x_imgcoord(i,:) = Delta_x(i, :)*Tr_Data{k}.intermediate_bboxes{stage}(s, 3);
    Delta_y_imgcoord(i,:) = Delta_y(i, :)*Tr_Data{k}.intermediate_bboxes{stage}(s, 4);
end

MRSE = 100*mean(mean(sqrt(Delta_x_imgcoord.^2+Delta_y_imgcoord.^2), 2)./dist_pupils);

MRSE_display = sprintf('Mean Root Square Error for %d Training Samples: %f', (dbsize*(params.augnumber)), MRSE);
disp(MRSE_display);

for i = 1:dbsize*(params.augnumber)
    k = floor((i-1)/(params.augnumber)) + 1;
    s = mod(i-1, (params.augnumber)) + 1;
    
    shapes_stage = Tr_Data{k}.intermediate_shapes{stage};
    shape_stage = shapes_stage(:, :, s);     
    
    % transform above delta shape into the coordinate of current intermmediate shape
    delta_shape_interm_coord = [deltashapes_bar_x(i, :)', deltashapes_bar_y(i, :)'];   
    % [delta_shape_interm_coord] = tforminv(Tr_Data{k}.tf2meanshape{s}, deltashapes_bar_x(i, :)', deltashapes_bar_y(i, :)');
    
    % derive the delta shape in the coordinate system of meanshape
    delta_shape_interm_coord = bsxfun(@times, delta_shape_interm_coord, Tr_Data{k}.intermediate_bboxes{stage}(s, 3:4));        
    
    shape_newstage = shape_stage + delta_shape_interm_coord;  
    
    Tr_Data{k}.intermediate_shapes{stage+1}(:, :, s) = shape_newstage;
    
    % update transformation of current intermediate shape to meanshape
    Tr_Data{k}.intermediate_bboxes{stage+1}(s, :) = getbbox(Tr_Data{k}.intermediate_shapes{stage+1}(:, :, s));
    
    meanshape_resize = resetshape(Tr_Data{k}.intermediate_bboxes{stage+1}(s, :), params.meanshape);    
    
    Tr_Data{k}.tf2meanshape{s} = cp2tform(bsxfun(@minus, Tr_Data{k}.intermediate_shapes{stage+1}(1:end,:, s), mean(Tr_Data{k}.intermediate_shapes{stage+1}(1:end,:, s))), ...
        bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :))), 'nonreflective similarity');
    
    Tr_Data{k}.meanshape2tf{s} = cp2tform(bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :))), ...
        bsxfun(@minus, Tr_Data{k}.intermediate_shapes{stage+1}(1:end,:, s), mean(Tr_Data{k}.intermediate_shapes{stage+1}(1:end,:, s))), 'nonreflective similarity');
    
    shape_residual = bsxfun(@rdivide, Tr_Data{k}.shape_gt - shape_newstage, Tr_Data{k}.intermediate_bboxes{stage+1}(s, 3:4));    
    
    Tr_Data{k}.shapes_residual(:, :, s) = tformfwd(Tr_Data{k}.tf2meanshape{s}, shape_residual(:, 1), shape_residual(:, 2));   
    
    %{
    drawshapes(Tr_Data{k}.img_gray, [Tr_Data{k}.shape_gt shape_stage shape_newstage]);   
    hold off;
    drawnow;
    w = waitforbuttonpress;
    %}
end

end

