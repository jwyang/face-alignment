function [W, Tr_Data] = globalregression_nn(binaryfeatures, Tr_Data, params, stage)
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
deltashapes = zeros(dbsize*(params.augnumber + 1), 2*size(params.meanshape, 1));  % concatenate 2-D coordinates into a vector (N X (2*L))
dist_pupils = zeros(dbsize*(params.augnumber + 1), 1);
bboxes = zeros(dbsize*(params.augnumber + 1), 2);

for i = 1:dbsize*(params.augnumber + 1)
    k = floor((i-1)/(params.augnumber + 1)) + 1;
    s = mod(i-1, (params.augnumber + 1)) + 1;
    
    shape_gt = Tr_Data{k}.shape_gt;    
    dist_pupils(i) = norm((mean(shape_gt(37:42, :)) - mean(shape_gt(43:48, :))));
    bboxes(i, :) = [Tr_Data{k}.bbox.width Tr_Data{k}.bbox.height];
    
    delta_shape = Tr_Data{k}.shapes_residual(:, :, s);
    deltashapes(i, :) = delta_shape(:)';
end

% conduct regression using libliear
% X : binaryfeatures
% Y:  gtshapes

% Method 1: closed-form solution by matrix inverse
% % X^T*X
% A = binaryfeatures'*binaryfeatures;
% 
% % Y^TX
% B = gtshapes'*binaryfeatures;
% 
% % W = B*(A+\lambdaI)^{-1}
% W = B*inv(A+0.001*ones(size(A)));

% Method 2: iterative optimization using gradient descend 
% the range of c should be in [0.0005, 0.005]

param = '-s 12 -p 0 -c 0.0005 -q heart_scale';
W = zeros(size(binaryfeatures, 2), size(deltashapes, 2));
tic;
parfor o = 1:size(deltashapes, 2)
    model = train(deltashapes(:, o), sparse(binaryfeatures), param);
    W(:, o) = model.w';
end
toc;

deltashapes_bar = binaryfeatures*W;


% Method 3: Neural Network with Dropout
%{
rng(0);
nn = nnsetup([size(binaryfeatures, 2) 50 50 size(deltashapes, 2)]);

nn.dropoutFraction = 0.0;      %  Dropout fraction 
opts.numepochs     = 500;      %  Number of full sweeps through data
opts.batchsize     = 50;      % size(binaryfeatures, 1);       %  Take a mean gradient step over this many samples

num_samples = opts.batchsize*floor(size(binaryfeatures, 1)/opts.batchsize);

W = nntrain(nn, binaryfeatures(1:num_samples, :), deltashapes(1:num_samples, :), bboxes(1:num_samples, :), dist_pupils(1:num_samples, :), opts);

[deltashapes_bar, ~] = nntest_reg(nn, binaryfeatures, deltashapes, bboxes, dist_pupils);
%}

% Predict the location of lanmarks using current regression matrix
deltashapes_bar_x = deltashapes_bar(:, 1:int8(size(deltashapes_bar, 2)/2));
deltashapes_bar_y = deltashapes_bar(:, uint8(size(deltashapes_bar, 2)/2)+1:end);

Delta = deltashapes_bar-deltashapes;

Delta_x = Delta(:, 1:int8(size(deltashapes_bar, 2)/2));
Delta_y = Delta(:, uint8(size(deltashapes_bar, 2)/2)+1:end);

for s = 1:dbsize
    Delta_x_imgcoord((s-1)*(params.augnumber + 1) + 1:s*(params.augnumber + 1),:) = ...
        bsxfun(@times, Delta_x((s-1)*(params.augnumber + 1) + 1:s*(params.augnumber + 1),:), Tr_Data{s}.bbox.width);
    Delta_y_imgcoord((s-1)*(params.augnumber + 1) + 1:s*(params.augnumber + 1),:) = ...
        bsxfun(@times, Delta_y((s-1)*(params.augnumber + 1) + 1:s*(params.augnumber + 1),:), Tr_Data{s}.bbox.height);
end



MRSE = 100*mean(mean(sqrt(Delta_x_imgcoord.^2+Delta_y_imgcoord.^2), 2)./dist_pupils);

MRSE_display = sprintf('Mean Root Square Error for %d Training Samples: %f', (dbsize*(params.augnumber + 1)), MRSE);
disp(MRSE_display);

for i = 1:dbsize*(params.augnumber + 1)
    k = floor((i-1)/(params.augnumber + 1)) + 1;
    s = mod(i-1, (params.augnumber + 1)) + 1;
    
    shapes_stage = Tr_Data{k}.intermediate_shapes{stage};
    shape_stage = shapes_stage(:, :, s);     
    
    % transform above delta shape into the coordinate of current intermmediate shape
    
    [delta_shape_interm_coord] = tforminv(Tr_Data{k}.tf2meanshape{s}, deltashapes_bar_x(i, :)', deltashapes_bar_y(i, :)');
    
    % derive the delta shape in the coordinate system of meanshape
    delta_shape_interm_coord = bsxfun(@times, delta_shape_interm_coord, [Tr_Data{k}.bbox.width Tr_Data{k}.bbox.height]);
        
    shape_newstage = shape_stage + delta_shape_interm_coord;
   
    Tr_Data{k}.intermediate_shapes{stage+1}(:, :, s) = shape_newstage;
    
    % update transformation of current intermediate shape to meanshape
    meanshape_resize = resetshape(Tr_Data{k}.bbox, params.meanshape);    
    
    Tr_Data{k}.tf2meanshape{s} = cp2tform(bsxfun(@minus, Tr_Data{k}.intermediate_shapes{stage+1}(:,:, s), mean(Tr_Data{k}.intermediate_shapes{stage+1}(:,:, s))), ...
        bsxfun(@minus, meanshape_resize, mean(meanshape_resize)), 'nonreflective similarity');
    
    shape_residual = bsxfun(@rdivide, Tr_Data{k}.shape_gt - shape_newstage, [Tr_Data{k}.bbox.width Tr_Data{k}.bbox.height]);    
    
    Tr_Data{k}.shapes_residual(:, :, s) = tformfwd(Tr_Data{k}.tf2meanshape{s}, shape_residual(:, 1), shape_residual(:, 2));   
    
    % drawshapes(Tr_Data{k}.img_gray, [Tr_Data{k}.shape_gt shape_newstage]);        
    % close;
end

end

