function LBFRegModel = train_model(dbnames)
%TRAIN_MODEL Summary of this function goes here
%   Function: train face alignment model
%   Detailed explanation goes here
%   Input:
%       dbnames: the names of database
% Configure the parameters for training model
global params;
global Tr_Data;
config_tr;

if size(dbnames) > 1 & sum(strcmp(dbnames, 'COFW')) > 0
    disp('Sorry, COFW cannnot be combined with others')
    return;
end

if sum(strcmp(dbnames, 'COFW')) > 0
    load('..\initial_shape\InitialShape_29.mat');
    params.meanshape        = S0;
else
    load('..\initial_shape\InitialShape_68.mat');
    params.meanshape        = S0(params.ind_usedpts, :);
end


if params.isparallel
    if isempty(gcp('nocreate')) %判断并行计算环境是否已然启动
        parpool(4); %若尚未启动，则启动并行环境
    else
        disp('Already initialized'); %说明并行环境已经启动。
    end
end


% load trainning data from hardware
Tr_Data    = [];
% Tr_Bboxes  = [];
for i = 1:length(dbnames)
    % load training samples (including training images, and groundtruth shapes)
    imgpathlistfile = strcat('D:\Projects_Face_Detection\Datasets\', dbnames{i}, '\Path_Images.txt');
    tr_data = loadsamples(imgpathlistfile, 2);
    Tr_Data = [Tr_Data; tr_data];
end

% Augmentate data for traing: assign multiple initial shapes to each image
Data = Tr_Data; % (1:10:end);
Param = params;

if Param.flipflag % if conduct flipping
    Data_flip = cell(size(Data, 1), 1);
    for i = 1:length(Data_flip)
        Data_flip{i}.img_gray    = fliplr(Data{i}.img_gray);
        Data_flip{i}.width_orig  = Data{i}.width_orig;
        Data_flip{i}.height_orig = Data{i}.height_orig;    
        Data_flip{i}.width       = Data{i}.width;
        Data_flip{i}.height      = Data{i}.height; 
        
        Data_flip{i}.shape_gt    = flipshape(Data{i}.shape_gt);        
        Data_flip{i}.shape_gt(:, 1)  =  Data{i}.width - Data_flip{i}.shape_gt(:, 1);
        
        Data_flip{i}.bbox_gt        = Data{i}.bbox_gt;
        Data_flip{i}.bbox_gt(1)     = Data_flip{i}.width - Data_flip{i}.bbox_gt(1) - Data_flip{i}.bbox_gt(3);       
        
        % Data_flip{i}.bbox_facedet        = Data{i}.bbox_facedet;
        % Data_flip{i}.bbox_facedet(1)     = Data_flip{i}.width - Data_flip{i}.bbox_facedet(1) - Data_flip{i}.bbox_facedet(3);   
    end
    Data = [Data; Data_flip];
end

% choose corresponding points for training
for i = 1:length(Data)
    Data{i}.shape_gt = Data{i}.shape_gt(params.ind_usedpts, :);
    Data{i}.bbox_gt = getbbox(Data{i}.shape_gt);
end

dbsize = length(Data);
    
% load('Ts_bbox.mat');

augnumber = Param.augnumber;

    
for i = 1:dbsize        
    % initializ the shape of current face image by randomly selecting multiple shapes from other face images       
    % indice = ceil(dbsize*rand(1, augnumber));  

    indice_rotate = ceil(dbsize*rand(1, augnumber));  
    indice_shift  = ceil(dbsize*rand(1, augnumber));  
    scales        = 1 + 0.3*(rand([1 augnumber]) - 0.5);
    
    Data{i}.intermediate_shapes = cell(1, Param.max_numstage);
    Data{i}.intermediate_bboxes = cell(1, Param.max_numstage);
    
    Data{i}.intermediate_shapes{1} = zeros([size(Param.meanshape), augnumber]);
    Data{i}.intermediate_bboxes{1} = zeros([augnumber, size(Data{i}.bbox_gt, 2)]);
    
    Data{i}.shapes_residual = zeros([size(Param.meanshape), augnumber]);
    Data{i}.tf2meanshape = cell(augnumber, 1);
    Data{i}.meanshape2tf = cell(augnumber, 1);
    
    
    % if Data{i}.isdet == 1
    %    Data{i}.bbox_facedet = Data{i}.bbox_facedet*ts_bbox;
    % end     
    
      
    
    for s = 1:Param.augnumber_shift+1
        for r = 1:Param.augnumber_rotate+1
            for e = 1:Param.augnumber_scale+1
            sr = (s-1)*(Param.augnumber_rotate+1)*(Param.augnumber_scale+1) + (r-1)*((Param.augnumber_scale+1)) + e;
            if s == 1 && r == 1 && e == 1% initialize as meanshape
                % estimate the similarity transformation from initial shape to mean shape
                Data{i}.intermediate_shapes{1}(:,:, sr) = resetshape(Data{i}.bbox_gt, Param.meanshape);
                Data{i}.intermediate_bboxes{1}(sr, :) = Data{i}.bbox_gt;
                
                meanshape_resize = resetshape(Data{i}.intermediate_bboxes{1}(sr, :), Param.meanshape);
                
                
                Data{i}.tf2meanshape{1} = cp2tform(bsxfun(@minus, Data{i}.intermediate_shapes{1}(1:end,:, 1), mean(Data{i}.intermediate_shapes{1}(1:end,:, 1))), ...
                    bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :))), 'nonreflective similarity');                
                Data{i}.meanshape2tf{1} = cp2tform(bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :))), ...
                    bsxfun(@minus, Data{i}.intermediate_shapes{1}(1:end,:, 1), mean(Data{i}.intermediate_shapes{1}(1:end,:, 1))), 'nonreflective similarity');
              
                
                % calculate the residual shape from initial shape to groundtruth shape under normalization scale
                shape_residual = bsxfun(@rdivide, Data{i}.shape_gt - Data{i}.intermediate_shapes{1}(:,:, 1), [Data{i}.intermediate_bboxes{1}(1, 3) Data{i}.intermediate_bboxes{1}(1, 4)]);                
                % transform the shape residual in the image coordinate to the mean shape coordinate
                Data{i}.shapes_residual(:, :, 1) = tformfwd(Data{i}.tf2meanshape{1}, shape_residual(:, 1), shape_residual(:, 2));
            else  % randomly shift and rotate the meanshape (or groundtruth of other ssubjects)
                % randomly rotate the shape
                
                shape = resetshape(Data{i}.bbox_gt, Param.meanshape);       % Data{indice_rotate(sr)}.shape_gt
                
                % shape = scaleshape(shape, scales(sr));

                % shape = rotateshape(shape);   
                
                % randomly shift the shape
                shape = translateshape(shape, Data{indice_shift(sr)}.shape_gt);
                                
                Data{i}.intermediate_shapes{1}(:, :, sr) = shape;
                Data{i}.intermediate_bboxes{1}(sr, :) = getbbox(shape);
                
                meanshape_resize = resetshape(Data{i}.intermediate_bboxes{1}(sr, :), Param.meanshape);
                
                
                Data{i}.tf2meanshape{sr} = cp2tform(bsxfun(@minus, Data{i}.intermediate_shapes{1}(1:end,:, sr), mean(Data{i}.intermediate_shapes{1}(1:end,:, sr))), ...
                    bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :))), 'nonreflective similarity');
                Data{i}.meanshape2tf{sr} = cp2tform(bsxfun(@minus, meanshape_resize(1:end, :), mean(meanshape_resize(1:end, :))), ...
                    bsxfun(@minus, Data{i}.intermediate_shapes{1}(1:end,:, sr), mean(Data{i}.intermediate_shapes{1}(1:end,:, sr))), 'nonreflective similarity');                
                
                
                shape_residual = bsxfun(@rdivide, Data{i}.shape_gt - Data{i}.intermediate_shapes{1}(:,:, sr), [Data{i}.intermediate_bboxes{1}(sr, 3) Data{i}.intermediate_bboxes{1}(sr, 4)]);  
                
                Data{i}.shapes_residual(:, :, sr) = tformfwd(Data{i}.tf2meanshape{sr}, shape_residual(:, 1), shape_residual(:, 2));
            end
            
            %{
            drawshapes(Data{i}.img_gray, [Data{i}.shape_gt Data{i}.intermediate_shapes{1}(:, :, sr)]);
            hold off;
            %}
            end
        end
    end
end

% train random forests for each landmark
randf = cell(Param.max_numstage, 1);
Ws    = cell(Param.max_numstage, 1);

%{
if nargin > 2
    n = size(LBFRegModel_initial.ranf, 1);
    for i = 1:n
      randf(1:n, :) = LBFRegModel_initial.ranf;                
    end
end
%}

for s = 1:Param.max_numstage
    % learn random forest for s-th stage
    disp('train random forests for landmarks...');

    %{
    if isempty(randf{s})        
        if exist(strcat('randfs\randf', num2str(s), '.mat'))
            load(strcat('randfs\randf', num2str(s), '.mat'));
        else
            tic;            
            randf{s} = train_randomfs(Data, Param, s);
            toc;
            save(strcat('randfs\randf', num2str(s), '.mat'), 'randf', '-v7.3');
        end
    end
    %}
    tic;            
    randf{s} = train_randomfs(Data, Param, s);
    toc;
              
    % derive binary codes given learned random forest in current stage
    disp('extract local binary features...');

    if exist(strcat('LBFeats\LBFeats', num2str(s), '.mat'))
        load(strcat('LBFeats\LBFeats', num2str(s), '.mat'));
    else
        tic;
        binfeatures = derivebinaryfeat(randf{s}, Data, Param, s);
        % save(strcat('LBFeats\LBFeats', num2str(s), '.mat'), 'binfeatures', '-v7.3');
        toc;
    end
    
    % learn global linear regrassion given binary feature
    disp('learn global regressors...');
    tic;
    [W, Data] = globalregression(binfeatures, Data, Param, s);        
    Ws{s} = W;    
    % save(strcat('Ws\W', num2str(s), '.mat'), 'W', '-v7.3');
    toc;      
    
end

LBFRegModel.ranf = randf;
LBFRegModel.Ws   = Ws;

end