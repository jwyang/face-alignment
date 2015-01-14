function binfeatures = derivebinaryfeat(randf, Tr_Data, params, stage)
%DERIVEBINARYFEAT Summary of this function goes here
%   Function: Derive binary features for each sample given learned random forest
%   Detailed explanation goes here
%   Input:
%       lmarkID: the ID of landmark
%       randf:   learned random forest in current stage
%       Tr_Data: training data
%       params:  parameters for curent stage

% calculate the overall dimension of binary feature, concatenate the
% features for all landmarks, and the feature of one landmark is sum
% leafnodes of all random trees;

dims_binfeat = 0;

ind_bincode = zeros(size(randf));

for l = 1:size(randf, 1)
    for t = 1:size(randf, 2)
        ind_bincode(l, t) = randf{l, t}.num_leafnodes;
        dims_binfeat = dims_binfeat + randf{l, t}.num_leafnodes;
    end
end

% initilaize the memory for binfeatures
dbsize = length(Tr_Data);

% faster implementation
%{
tic;
binfeatures   = zeros(dbsize*(params.augnumber), dims_binfeat);
img_sizes     = zeros(2, dbsize*(params.augnumber));
shapes        = zeros([size(params.meanshape), dbsize*(params.augnumber)]);
tfs2meanshape = cell(1, dbsize*params.augnumber);

img_areas = zeros(1, dbsize);
parfor i = 1:dbsize
    img_areas(i) = Tr_Data{i}.width*Tr_Data{i}.height;
end

imgs_gray = zeros(dbsize, max(img_areas));

for i = 1:dbsize
    area = img_areas(i);
    imgs_gray(i, 1:area) = Tr_Data{i}.img_gray(:)';    
    shapes(:, :, (i-1)*params.augnumber+1:i*params.augnumber) = Tr_Data{i}.intermediate_shapes{stage};    
    
    for k = 1:params.augnumber
        img_sizes(:, (i-1)*params.augnumber+k) = [Tr_Data{i}.width; Tr_Data{i}.height];
        tfs2meanshape{(i-1)*params.augnumber+k} = Tr_Data{i}.tf2meanshape{k};
    end
end

binfeature_lmarks = cell(1, size(params.meanshape, 1));
num_leafnodes = zeros(1, size(params.meanshape, 1));    
for l = 1:size(params.meanshape, 1)
    [binfeature_lmarks{l}, num_leafnodes(l)] = lbf_faster(randf{l}, imgs_gray, img_sizes, shapes(l, :, :), tfs2meanshape, params, stage);
end

cumnum_leafnodes = [0 cumsum(num_leafnodes)];
binfeatures_faster = binfeatures;
for l = 1:size(params.meanshape, 1)
    binfeatures_faster(:, cumnum_leafnodes(l)+1:cumnum_leafnodes(l+1)) = binfeature_lmarks{l};
end

toc;
%}
feats = cell(size(params.meanshape, 1), 1);
isleaf = cell(size(params.meanshape, 1), 1);
threshs = cell(size(params.meanshape, 1), 1);
cnodes = cell(size(params.meanshape, 1), 1);

% prepare for the derivation of local binary codes
for l = 1:size(params.meanshape, 1)
    % concatenate all nodes of all random trees
    rf = randf(l, :);
    num_rfnodes = 0;
    for t = 1:params.max_numtrees
        num_rfnodes = num_rfnodes + rf{t}.num_nodes;
    end
    
    % fast implementation
    feats{l} = zeros(num_rfnodes, 4);
    isleaf{l} = zeros(num_rfnodes, 1);
    threshs{l} = zeros(num_rfnodes, 1);
    cnodes{l} = zeros(num_rfnodes, 2);
    
    id_rfnode = 1;
    
    for t = 1:params.max_numtrees
        feats{l}(id_rfnode:(id_rfnode + rf{t}.num_nodes - 1), :)   = rf{t}.feat(1:rf{t}.num_nodes, :);
        isleaf{l}(id_rfnode:(id_rfnode + rf{t}.num_nodes - 1), :)  = rf{t}.isleafnode(1:rf{t}.num_nodes, :);
        threshs{l}(id_rfnode:(id_rfnode + rf{t}.num_nodes - 1), :) = rf{t}.thresh(1:rf{t}.num_nodes, :);
        cnodes{l}(id_rfnode:(id_rfnode + rf{t}.num_nodes - 1), :)  = rf{t}.cnodes(1:rf{t}.num_nodes, :);
        
        id_rfnode = id_rfnode + rf{t}.num_nodes;
    end
end
    

% extract feature for each samples
parfor i = 1:dbsize*(params.augnumber)
    k = floor((i-1)/(params.augnumber)) + 1;
    s = mod(i-1, (params.augnumber)) + 1;
    img_gray = Tr_Data{k}.img_gray;
    bbox     = Tr_Data{k}.intermediate_bboxes{stage}(s, :);
    shapes   = Tr_Data{k}.intermediate_shapes{stage};
    shape    = shapes(:, :, s);
    
    % tf2meanshape = Tr_Data{k}.tf2meanshape{s};
    meanshape2tf = Tr_Data{k}.meanshape2tf{s};
    %{
    feats_cell   = cell(size(params.meanshape, 1), params.max_numtrees);
       
    for l = 1:size(params.meanshape, 1)
        % extract feature for each landmark from the random forest
        for t = 1:params.max_numtrees
            % extract feature for each ladnmark from a random tree
            bincode = traversetree(randf{l}{t}, img_gray, bbox, shape(l, :), tf2meanshape, params, stage);
            feats_cell{l, t} = bincode;
        end
    end
    
    binfeature = zeros(1, dims_binfeat);
    for l = 1:size(params.meanshape, 1)
        % extract feature for each landmark from the random forest
        offset = sum(ind_bincodes(1:l-1, end));
        for t = 1:params.max_numtrees
            ind_s = ind_bincodes(l, t) + 1 + offset;
            ind_e = ind_bincodes(l, t+1) + offset;
            % extract feature for each ladnmark from a random tree
            binfeature(ind_s:ind_e) = feats_cell{l, t};
        end
    end
    %}
    % fast implementation
    
    binfeature_lmarks = cell(1, size(params.meanshape, 1));
    num_leafnodes = zeros(1, size(params.meanshape, 1));
    for l = 1:size(params.meanshape, 1)
        % concatenate all nodes of all random trees
        [binfeature_lmarks{l}, num_leafnodes(l)]= lbf_fast(randf(l, :), feats{l}, isleaf{l}, threshs{l}, cnodes{l}, img_gray, bbox, shape(l, :), meanshape2tf, params, stage);
    end
    
    cumnum_leafnodes = [0 cumsum(num_leafnodes)];
    
    binfeature_alllmarks = zeros(1, cumnum_leafnodes(end));
    for l = 1:size(params.meanshape, 1)
        binfeature_alllmarks(cumnum_leafnodes(l)+1:cumnum_leafnodes(l+1)) = binfeature_lmarks{l};
    end
    
    
    %{
    % faster implementation (thinking...)
    binfeature_lmarks = cell(1, size(params.meanshape, 1));
    num_leafnodes = zeros(1, size(params.meanshape, 1));
    for l = 1:size(params.meanshape, 1)
        % concatenate all nodes of all random trees
        [binfeature_lmarks{l}, num_leafnodes(l)]= lbf_faster(randf{l}, img_gray, bbox, shape(l, :), tf2meanshape, params, stage);
    end
    
    cumnum_leafnodes = [0 cumsum(num_leafnodes)];
    
    binfeature_alllmarks = zeros(1, cumnum_leafnodes(end));
    for l = 1:size(params.meanshape, 1)
        binfeature_alllmarks(cumnum_leafnodes(l)+1:cumnum_leafnodes(l+1)) = binfeature_lmarks{l};
    end
    %}
    
    binfeatures(i, :) = binfeature_alllmarks;
end


end

function bincode = traversetree(tree, img_gray, bbox, shape, tf2meanshape, params, stage)

currnode = tree.rtnodes{1};

bincode = zeros(1, tree.num_leafnodes);

width = size(img_gray, 2);
height = size(img_gray, 1);

while(1)
    
    anglepairs = currnode.feat(1:2);
    radiuspairs = currnode.feat(3:4);
    
    % calculate the relative location under the coordinate of meanshape
    pixel_a_x_imgcoord = cos(anglepairs(:, 1)).*radiuspairs(:, 1)*params.max_raio_radius(stage)*bbox(3);
    pixel_a_y_imgcoord = sin(anglepairs(:, 1)).*radiuspairs(:, 1)*params.max_raio_radius(stage)*bbox(4);
    
    pixel_b_x_imgcoord = cos(anglepairs(:, 2)).*radiuspairs(:, 2)*params.max_raio_radius(stage)*bbox(3);
    pixel_b_y_imgcoord = sin(anglepairs(:, 2)).*radiuspairs(:, 2)*params.max_raio_radius(stage)*bbox(4);
    
    % transform the pixels from image coordinate (meanshape) to coordinate of current shape
    [pixel_a_x_lmcoord, pixel_a_y_lmcoord] = tforminv(tf2meanshape, pixel_a_x_imgcoord, pixel_a_y_imgcoord);
    
    [pixel_b_x_lmcoord, pixel_b_y_lmcoord] = tforminv(tf2meanshape, pixel_b_x_imgcoord, pixel_b_y_imgcoord);
    
    pixel_a_x = ceil(pixel_a_x_lmcoord + shape(1));
    pixel_a_y = ceil(pixel_a_y_lmcoord + shape(2));
    
    pixel_b_x = ceil(pixel_b_x_lmcoord + shape(1));
    pixel_b_y = ceil(pixel_b_y_lmcoord + shape(2));
    
    pixel_a_x = max(1, min(pixel_a_x, width));
    pixel_a_y = max(1, min(pixel_a_y, height));
    
    pixel_b_x = max(1, min(pixel_b_x, width));
    pixel_b_y = max(1, min(pixel_b_y, height));
    
    f = int16(img_gray(pixel_a_y + (pixel_a_x-1)*height)) - int16(img_gray(pixel_b_y + (pixel_b_x-1)*height));
    
    if f < (currnode.thresh)
        id_chilenode = currnode.cnodes(1);
        currnode = tree.rtnodes{id_chilenode};
    else
        id_chilenode = currnode.cnodes(2);
        currnode = tree.rtnodes{id_chilenode};
    end
    
    if isempty(currnode.cnodes)
        % if the current node is a leaf node, then stop traversin
        bincode(tree.id_leafnodes == id_chilenode) = 1;
        break;
    end
end

end

function [binfeature, num_leafnodes] = lbf_fast(rf, feats, isleaf, threshs, cnodes, img_gray, bbox, shape, meanshape2tf, params, stage)

%{
num_rfnodes = 0;
for t = 1:params.max_numtrees
    num_rfnodes = num_rfnodes + rf{t}.num_nodes;
end

% fast implementation
feats = zeros(num_rfnodes, 4);
isleaf = zeros(num_rfnodes, 1);
threshs = zeros(num_rfnodes, 1);
cnodes = zeros(num_rfnodes, 2);

id_rfnode = 1;

for t = 1:params.max_numtrees
    feats(id_rfnode:(id_rfnode + rf{t}.num_nodes - 1), :)   = rf{t}.feat(1:rf{t}.num_nodes, :);
    isleaf(id_rfnode:(id_rfnode + rf{t}.num_nodes - 1), :)  = rf{t}.isleafnode(1:rf{t}.num_nodes, :);
    threshs(id_rfnode:(id_rfnode + rf{t}.num_nodes - 1), :) = rf{t}.thresh(1:rf{t}.num_nodes, :);
    cnodes(id_rfnode:(id_rfnode + rf{t}.num_nodes - 1), :)  = rf{t}.cnodes(1:rf{t}.num_nodes, :);    
    
    id_rfnode = id_rfnode + rf{t}.num_nodes;
end
%}

width = size(img_gray, 2);
height = size(img_gray, 1);

anglepairs = feats(:, 1:2);
radiuspairs = feats(:, 3:4);

% calculate the relative location under the coordinate of meanshape
pixel_a_x_imgcoord = cos(anglepairs(:, 1)).*radiuspairs(:, 1)*params.max_raio_radius(stage)*bbox(3);
pixel_a_y_imgcoord = sin(anglepairs(:, 1)).*radiuspairs(:, 1)*params.max_raio_radius(stage)*bbox(4);

pixel_b_x_imgcoord = cos(anglepairs(:, 2)).*radiuspairs(:, 2)*params.max_raio_radius(stage)*bbox(3);
pixel_b_y_imgcoord = sin(anglepairs(:, 2)).*radiuspairs(:, 2)*params.max_raio_radius(stage)*bbox(4);

% no transformaiton
%{
pixel_a_x_lmcoord = pixel_a_x_imgcoord;
pixel_a_y_lmcoord = pixel_a_y_imgcoord;
    
pixel_b_x_lmcoord = pixel_b_x_imgcoord;
pixel_b_y_lmcoord = pixel_b_y_imgcoord;
%}

% transform the pixels from image coordinate (meanshape) to coordinate of current shape

[pixel_a_x_lmcoord, pixel_a_y_lmcoord] = transformPointsForward(meanshape2tf, pixel_a_x_imgcoord', pixel_a_y_imgcoord');
pixel_a_x_lmcoord = pixel_a_x_lmcoord';
pixel_a_y_lmcoord = pixel_a_y_lmcoord';

[pixel_b_x_lmcoord, pixel_b_y_lmcoord] = transformPointsForward(meanshape2tf, pixel_b_x_imgcoord', pixel_b_y_imgcoord');
pixel_b_x_lmcoord = pixel_b_x_lmcoord';
pixel_b_y_lmcoord = pixel_b_y_lmcoord';

pixel_a_x = ceil(pixel_a_x_lmcoord + shape(1));
pixel_a_y = ceil(pixel_a_y_lmcoord + shape(2));

pixel_b_x = ceil(pixel_b_x_lmcoord + shape(1));
pixel_b_y = ceil(pixel_b_y_lmcoord + shape(2));

pixel_a_x = max(1, min(pixel_a_x, width));
pixel_a_y = max(1, min(pixel_a_y, height));

pixel_b_x = max(1, min(pixel_b_x, width));
pixel_b_y = max(1, min(pixel_b_y, height));

pdfeats = double(img_gray(pixel_a_y + (pixel_a_x-1)*height)) - double(img_gray(pixel_b_y + (pixel_b_x-1)*height));
    % ./ double(img_gray(pixel_a_y + (pixel_a_x-1)*height)) + double(img_gray(pixel_b_y + (pixel_b_x-1)*height));

cind = (pdfeats >= threshs) + 1;

% obtain the indice of child nodes for all nodes, if the current node is
% leaf node, then the indice of its child node is 0
ind_cnodes = cnodes; % diag(cnodes(:, cind));

binfeature = zeros(1, sum(isleaf));

cumnum_nodes = 0;
cumnum_leafnodes = 0;
for t = 1:params.max_numtrees
    num_nodes = (rf{t}.num_nodes);
    id_cnode = 1;
    while(1)
        if isleaf(id_cnode + cumnum_nodes) 
            binfeature(cumnum_leafnodes + find(rf{t}.id_leafnodes == id_cnode)) = 1;
            cumnum_nodes = cumnum_nodes + num_nodes;
            cumnum_leafnodes = cumnum_leafnodes + rf{t}.num_leafnodes;   
            break;
        end
        id_cnode = ind_cnodes(cumnum_nodes + id_cnode, cind(cumnum_nodes + id_cnode));
    end    
end

num_leafnodes = sum(isleaf);

end
