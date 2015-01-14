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
gtshapes = zeros([size(params.meanshape) dbsize*(params.augnumber)]);  % concatenate 2-D coordinates into a vector (N X (2*L))

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
    else
        dist_pupils(i) = norm((mean(shape_gt(1:2, :)) - mean(shape_gt(end - 1:end, :))));
    end
    gtshapes(:, :, i) = shape_gt;
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
predshapes = zeros([size(params.meanshape) size(binaryfeatures, 1)]);  % concatenate 2-D coordinates into a vector (N X (2*L))

for i = 1:dbsize*(params.augnumber)
    k = floor((i-1)/(params.augnumber)) + 1;
    s = mod(i-1, (params.augnumber)) + 1;
    
    shapes_stage = Tr_Data{k}.intermediate_shapes{stage};
    shape_stage = shapes_stage(:, :, s);     
    
    deltashapes_bar_xy = reshape(deltashapes_bar(i, :), [uint8(size(deltashapes_bar, 2)/2) 2]);
    
    % transform above delta shape into the coordinate of current intermmediate shape
    % delta_shape_interm_coord = [deltashapes_bar_x(i, :)', deltashapes_bar_y(i, :)'];   
    [u, v] = transformPointsForward(Tr_Data{k}.meanshape2tf{s}, deltashapes_bar_xy(:, 1)', deltashapes_bar_xy(:, 2)');
    delta_shape_interm_coord = [u', v'];
    
    % derive the delta shape in the coordinate system of meanshape
    delta_shape_interm_coord = bsxfun(@times, delta_shape_interm_coord, Tr_Data{k}.intermediate_bboxes{stage}(s, 3:4));        
    
    shape_newstage = shape_stage + delta_shape_interm_coord;  
    predshapes(:, :, i) = shape_newstage;
    
    Tr_Data{k}.intermediate_shapes{stage+1}(:, :, s) = shape_newstage;
    
    % update transformation of current intermediate shape to meanshape
    Tr_Data{k}.intermediate_bboxes{stage+1}(s, :) = getbbox(Tr_Data{k}.intermediate_shapes{stage+1}(:, :, s));
    
    meanshape_resize = (resetshape(Tr_Data{k}.intermediate_bboxes{stage+1}(s, :), params.meanshape));    
    
    shape_residual = bsxfun(@rdivide, Tr_Data{k}.shape_gt - shape_newstage, Tr_Data{k}.intermediate_bboxes{stage+1}(s, 3:4));    
        
    Tr_Data{k}.tf2meanshape{s} = fitgeotrans(bsxfun(@minus, Tr_Data{k}.intermediate_shapes{stage+1}(1:end,:, s), mean(Tr_Data{k}.intermediate_shapes{stage+1}(1:end,:, s))), ...
        bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :))), 'NonreflectiveSimilarity');
    
    Tr_Data{k}.meanshape2tf{s} = fitgeotrans(bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :))), ...
        bsxfun(@minus, Tr_Data{k}.intermediate_shapes{stage+1}(1:end,:, s), mean(Tr_Data{k}.intermediate_shapes{stage+1}(1:end,:, s))), 'NonreflectiveSimilarity');
        
    [u, v] = transformPointsForward(Tr_Data{k}.tf2meanshape{s}, shape_residual(:, 1), shape_residual(:, 2));   
    Tr_Data{k}.shapes_residual(:, 1, s) = u';
    Tr_Data{k}.shapes_residual(:, 2, s) = v';
    %{
    drawshapes(Tr_Data{k}.img_gray, [Tr_Data{k}.shape_gt shape_stage shape_newstage]);   
    hold off;
    drawnow;
    w = waitforbuttonpress;
    %}
end

error_per_image = compute_error(gtshapes, predshapes);

MRSE = 100*mean(error_per_image);
MRSE_display = sprintf('Mean Root Square Error for %d Test Samples: %f', length(error_per_image), MRSE);
disp(MRSE_display);

end

function [ error_per_image ] = compute_error( ground_truth_all, detected_points_all )
%compute_error
%   compute the average point-to-point Euclidean error normalized by the
%   inter-ocular distance (measured as the Euclidean distance between the
%   outer corners of the eyes)
%
%   Inputs:
%          grounth_truth_all, size: num_of_points x 2 x num_of_images
%          detected_points_all, size: num_of_points x 2 x num_of_images
%   Output:
%          error_per_image, size: num_of_images x 1


num_of_images = size(ground_truth_all,3);
num_of_points = size(ground_truth_all,1);

error_per_image = zeros(num_of_images,1);

for i =1:num_of_images
    detected_points      = detected_points_all(:,:,i);
    ground_truth_points  = ground_truth_all(:,:,i);
    if num_of_points == 68
        interocular_distance = norm(mean(ground_truth_points(37:42,:))-mean(ground_truth_points(43:48,:)));  % norm((mean(shape_gt(37:42, :)) - mean(shape_gt(43:48, :))));
    elseif num_of_points == 51
        interocular_distance = norm(ground_truth_points(20,:) - ground_truth_points(29,:));
    elseif num_of_points == 29
        interocular_distance = norm(mean(ground_truth_points(9:2:17,:))-mean(ground_truth_points(10:2:18,:)));        
    else
        interocular_distance = norm(mean(ground_truth_points(1:2,:))-mean(ground_truth_points(end - 1:end,:)));        
    end
    
    sum=0;
    for j=1:num_of_points
        sum = sum+norm(detected_points(j,:)-ground_truth_points(j,:));
    end
    error_per_image(i) = sum/(num_of_points*interocular_distance);
end

end