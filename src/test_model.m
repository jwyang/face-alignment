function test_model(dbnames, LBFRegModel)
%TRAIN_MODEL Summary of this function goes here
%   Function: test face alignment model
%   Detailed explanation goes here
%   Input:
%       dbnames: the names of database
% Configure the parameters for training model
global params;
global Te_Data;
config_te;

if size(dbnames) > 1 & sum(strcmp(dbnames, 'COFW')) > 0
    disp('Sorry, COFW cannnot be combined with others')
    return;
end

if sum(strcmp(dbnames, 'COFW')) > 0
    load('..\Models\InitialShape_29.mat');
    params.meanshape        = S0;
else
    load('..\Models\InitialShape_68.mat');
    params.meanshape        = S0;
end 

if params.isparallel
    if matlabpool('size')<=0 %判断并行计算环境是否已然启动
        matlabpool('open','local',12); %若尚未启动，则启动并行环境
    else
        disp('Already initialized'); %说明并行环境已经启动。
    end
end

% load trainning data from hardware
Te_Data = [];
for i = 1:length(dbnames)
    % load training samples (including training images, and groundtruth shapes)
    imgpathlistfile = strcat('J:\jwyang\Face Alignment\Databases\', dbnames{i}, '\Path_Images.txt');
    te_data = loadsamples(imgpathlistfile, 1);
    Te_Data = [Te_Data; te_data];
end


% Augmentate data for traing: assign multiple initial shapes to each image

Data = Te_Data;
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
        
        Data_flip{i}.bbox_facedet        = Data{i}.bbox_facedet;
        Data_flip{i}.bbox_facedet(1)     = Data_flip{i}.width - Data_flip{i}.bbox_facedet(1) - Data_flip{i}.bbox_facedet(3);       
    end
    Data = [Data; Data_flip];
end

dbsize = length(Data);

% load('Ts_bbox.mat');

for i = 1:dbsize    
    augnumber = (Param.augnumber);
    
    % indice = ceil(dbsize*rand(1, augnumber));  

    indice_rotate = ceil(dbsize*rand(1, augnumber));  
    indice_shift  = ceil(dbsize*rand(1, augnumber));  
    
    Data{i}.intermediate_shapes = cell(1, Param.max_numstage);
    Data{i}.intermediate_bboxes = cell(1, Param.max_numstage);
    
    Data{i}.intermediate_shapes{1} = zeros([size(Param.meanshape), augnumber]);
    Data{i}.intermediate_bboxes{1} = zeros([augnumber, size(Data{i}.bbox_gt, 2)]);

    % Data{i}.bbox_facedet = Data{i}.bbox_facedet*ts_bbox;

    Data{i}.shapes_residual = zeros([size(Param.meanshape), augnumber]);
    Data{i}.tf2meanshape = cell(augnumber, 1);        
    
    meanshape_resize = resetshape(Data{i}.bbox_gt, Param.meanshape); 
    center_meanshape_resize = mean(meanshape_resize);
    
    for s = 1:Param.augnumber_shift+1
        for r = 1:Param.augnumber_rotate+1
            for e = 1:Param.augnumber_scale+1
            sr = (s-1)*(Param.augnumber_rotate+1)*(Param.augnumber_scale+1) + (r-1)*((Param.augnumber_scale+1)) + e;
            if s == 1 && r == 1 && e == 1% initialize as meanshape
                % estimate the similarity transformation from initial shape to mean shape
                Data{i}.intermediate_shapes{1}(:,:, sr) = resetshape(Data{i}.bbox_gt, Param.meanshape);
                Data{i}.intermediate_bboxes{1}(sr, :) = Data{i}.bbox_gt;                
                Data{i}.tf2meanshape{1} = cp2tform(bsxfun(@minus, Data{i}.intermediate_shapes{1}(:,:, sr), mean(Data{i}.intermediate_shapes{1}(:,:, sr))), ...
                    bsxfun(@minus, meanshape_resize, center_meanshape_resize), 'nonreflective similarity');                
            else  % randomly shift and rotate the meanshape (or groundtruth of other ssubjects)
                % randomly rotate the shape
                shape = resetshape(Data{i}.bbox_gt, Data{indice_rotate(sr)}.shape_gt);      
                % shape = rotateshape(meanshape_resize);                      
                % randomly shift the shape
                Data{i}.intermediate_shapes{1}(:, :, sr) = shape; % translateshape(shape, Data{indice_shift(sr)}.shape_gt);
                
                Data{i}.tf2meanshape{sr} = cp2tform(bsxfun(@minus, Data{i}.intermediate_shapes{1}(:,:, sr), mean(Data{i}.intermediate_shapes{1}(:,:, sr))), ...
                    bsxfun(@minus, meanshape_resize, center_meanshape_resize), 'nonreflective similarity');        
            end    
            %{
            drawshapes(Data{i}.img_gray, [Data{i}.intermediate_shapes{1}(:, :, 1) Data{i}.intermediate_shapes{1}(:, :, sr)]);
            hold off;
            %}
            end
        end
    end
end

% test random forests
% load('LBFRegModel_afw_lfpw.mat');

randf = LBFRegModel.ranf;
Ws    = LBFRegModel.Ws;

dbname_str = '';
for i = 1:length(dbnames)
    dbname_str = strcat(dbname_str, dbnames{i}, '_');
end
dbname_str = dbname_str(1:end-1);

if ~exist(dbname_str,'dir')
   mkdir(dbname_str);
end
    
for s = 1:params.max_numstage
    % derive binary codes given learned random forest in current stage
    
    disp('extract local binary features...');
    if ~exist(strcat(dbname_str, '\lbfeatures_', num2str(s), '.mat'))
        tic;
        binfeatures = derivebinaryfeat(randf{s}, Data, Param, s);
        toc;    
        save(strcat(dbname_str, '\lbfeatures_', num2str(s), '.mat'), 'binfeatures');
    else
        load(strcat(dbname_str, '\lbfeatures_', num2str(s), '.mat'));
    end
    % predict the locations of landmarks in current stage
    tic;
    disp('predict landmark locations...');

    Data = globalprediction(binfeatures, Ws{s}, Data, Param, s);        
    toc;        
    
end

end

