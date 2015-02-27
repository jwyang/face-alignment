function rfs = train_randomfs(Tr_Data, params, stage)
%TRAIN_RANDOMFS Summary of this function goes here
%   Function: train random forest for each landmark
%   Detailed explanation goes here
%   Input:
%        lmarkID: ID of landmark
%        stage: the stage of training process
%   Output:
%        randf: learned random forest
dbsize = length(Tr_Data);

% rf = cell(1, params.max_numtrees);

overlap_ratio = params.bagging_overlap;

Q = floor(double(dbsize)/((1-params.bagging_overlap)*(params.max_numtrees)));


Data = cell(1, params.max_numtrees);
for t = 1:params.max_numtrees
    % calculate the number of samples for each random tree
    % train t-th random tree
    is = max(floor((t-1)*Q - (t-1)*Q*overlap_ratio + 1), 1);
    ie = min(is + Q, dbsize);
    Data{t} = Tr_Data(is:ie);
end


% divide local region into grid
params.radius = ([0:1/30:1]');
params.angles = 2*pi*[0:1/36:1]';

rfs = cell(length(params.meanshape), params.max_numtrees);

parfor i = 1:length(params.meanshape)
    rf = cell(1, params.max_numtrees);
    % disp(strcat(num2str(i), 'th landmark is processing...'));
    for t = 1:params.max_numtrees
        % disp(strcat('training', {''}, num2str(t), '-th tree for', {''}, num2str(lmarkID), '-th landmark'));
        
        % calculate the number of samples for each random tree
        % train t-th random tree
        is = max(floor((t-1)*Q - (t-1)*Q*overlap_ratio + 1), 1);
        ie = min(is + Q, dbsize);
        
        max_numnodes = 2^params.max_depth - 1;
        
        rf{t}.ind_samples = cell(max_numnodes, 1);
        rf{t}.issplit     = zeros(max_numnodes, 1);
        rf{t}.pnode       = zeros(max_numnodes, 1);
        rf{t}.depth       = zeros(max_numnodes, 1);
        rf{t}.cnodes      = zeros(max_numnodes, 2);
        rf{t}.isleafnode  = zeros(max_numnodes, 1);
        rf{t}.feat        = zeros(max_numnodes, 4);
        rf{t}.thresh      = zeros(max_numnodes, 1);
        
        rf{t}.ind_samples{1} = 1:(ie - is + 1)*(params.augnumber);
        rf{t}.issplit(1)     = 0;
        rf{t}.pnode(1)       = 0;
        rf{t}.depth(1)       = 1;
        rf{t}.cnodes(1, 1:2) = [0 0];
        rf{t}.isleafnode(1)  = 1;
        rf{t}.feat(1, :)     = zeros(1, 4);
        rf{t}.thresh(1)      = 0;
        
        num_nodes = 1;
        num_leafnodes = 1;
        stop = 0;
        while(~stop)
            num_nodes_iter = num_nodes;
            num_split = 0;
            for n = 1:num_nodes_iter
                if ~rf{t}.issplit(n)
                    if rf{t}.depth(n) == params.max_depth % || length(rf{t}.ind_samples{n}) < 20
                        if rf{t}.depth(n) == 1
                            rf{t}.depth(n) = 1;
                        end
                        rf{t}.issplit(n) = 1;
                    else
                        % separate the samples into left and right path
                        
                        [thresh, feat, lcind, rcind, isvalid] = splitnode(i, rf{t}.ind_samples{n}, Data{t}, params, stage);
                        
                        %{
                    if ~isvalid
                        rf{t}.feat(n, :)   = [0 0 0 0];
                        rf{t}.thresh(n)    = 0;
                        rf{t}.issplit(n)   = 1;
                        rf{t}.cnodes(n, :) = [0 0];
                        rf{t}.isleafnode(n)   = 1;
                        continue;
                    end
                        %}
                        
                        % set the threshold and featture for current node
                        rf{t}.feat(n, :)   = feat;
                        rf{t}.thresh(n)    = thresh;
                        rf{t}.issplit(n)   = 1;
                        rf{t}.cnodes(n, :) = [num_nodes+1 num_nodes+2];
                        rf{t}.isleafnode(n)   = 0;
                        
                        % add left and right child nodes into the random tree
                        
                        rf{t}.ind_samples{num_nodes+1} = lcind;
                        rf{t}.issplit(num_nodes+1)     = 0;
                        rf{t}.pnode(num_nodes+1)       = n;
                        rf{t}.depth(num_nodes+1)       = rf{t}.depth(n) + 1;
                        rf{t}.cnodes(num_nodes+1, :)   = [0 0];
                        rf{t}.isleafnode(num_nodes+1)     = 1;
                        
                        rf{t}.ind_samples{num_nodes+2} = rcind;
                        rf{t}.issplit(num_nodes+2)     = 0;
                        rf{t}.pnode(num_nodes+2)       = n;
                        rf{t}.depth(num_nodes+2)       = rf{t}.depth(n) + 1;
                        rf{t}.cnodes(num_nodes+2, :)   = [0 0];
                        rf{t}.isleafnode(num_nodes+2)  = 1;
                        
                        num_split = num_split + 1;
                        num_leafnodes = num_leafnodes + 1;
                        num_nodes     = num_nodes + 2;
                    end
                end
            end
            
            if num_split == 0
                stop = 1;
            else
                rf{t}.num_leafnodes = num_leafnodes;
                rf{t}.num_nodes     = num_nodes;           
                rf{t}.id_leafnodes = find(rf{t}.isleafnode == 1); 
            end            
        end
        
    end
    % disp(strcat(num2str(i), 'th landmark is over'));
    rfs(i, :) = rf;
end
end

function [thresh, feat, lcind, rcind, isvalid] = splitnode(lmarkID, ind_samples, Tr_Data, params, stage)

if isempty(ind_samples)
    thresh = 0;
    feat = [0 0 0 0];
    rcind = [];
    lcind = [];
    isvalid = 1;
    return;
end

% generate params.max_rand cndidate feature
% anglepairs = samplerandfeat(params.max_numfeat);
% radiuspairs = [rand([params.max_numfeat, 1]) rand([params.max_numfeat, 1])];
[radiuspairs, anglepairs] = getproposals(params.max_numfeats(stage), params.radius, params.angles);

angles_cos = cos(anglepairs);
angles_sin = sin(anglepairs);

% extract pixel difference features from pairs
   
pdfeats = zeros(params.max_numfeats(stage), length(ind_samples));

shapes_residual = zeros(length(ind_samples), 2);

for i = 1:length(ind_samples)
    s = floor((ind_samples(i)-1)/(params.augnumber)) + 1;
    k = mod(ind_samples(i)-1, (params.augnumber)) + 1;
    
    % calculate the relative location under the coordinate of meanshape
    pixel_a_x_imgcoord = (angles_cos(:, 1)).*radiuspairs(:, 1)*params.max_raio_radius(stage)*Tr_Data{s}.intermediate_bboxes{stage}(k, 3);
    pixel_a_y_imgcoord = (angles_sin(:, 1)).*radiuspairs(:, 1)*params.max_raio_radius(stage)*Tr_Data{s}.intermediate_bboxes{stage}(k, 4);
    
    pixel_b_x_imgcoord = (angles_cos(:, 2)).*radiuspairs(:, 2)*params.max_raio_radius(stage)*Tr_Data{s}.intermediate_bboxes{stage}(k, 3);
    pixel_b_y_imgcoord = (angles_sin(:, 2)).*radiuspairs(:, 2)*params.max_raio_radius(stage)*Tr_Data{s}.intermediate_bboxes{stage}(k, 4);
    
    % no transformation
    %{
    pixel_a_x_lmcoord = pixel_a_x_imgcoord;
    pixel_a_y_lmcoord = pixel_a_y_imgcoord;
    
    pixel_b_x_lmcoord = pixel_b_x_imgcoord;
    pixel_b_y_lmcoord = pixel_b_y_imgcoord;
    %}
    
    % transform the pixels from image coordinate (meanshape) to coordinate of current shape
    
    [pixel_a_x_lmcoord, pixel_a_y_lmcoord] = transformPointsForward(Tr_Data{s}.meanshape2tf{k}, pixel_a_x_imgcoord', pixel_a_y_imgcoord');    
    pixel_a_x_lmcoord = pixel_a_x_lmcoord';
    pixel_a_y_lmcoord = pixel_a_y_lmcoord';
    
    [pixel_b_x_lmcoord, pixel_b_y_lmcoord] = transformPointsForward(Tr_Data{s}.meanshape2tf{k}, pixel_b_x_imgcoord', pixel_b_y_imgcoord');
    pixel_b_x_lmcoord = pixel_b_x_lmcoord';
    pixel_b_y_lmcoord = pixel_b_y_lmcoord';    
    
    pixel_a_x = int32(bsxfun(@plus, pixel_a_x_lmcoord, Tr_Data{s}.intermediate_shapes{stage}(lmarkID, 1, k)));
    pixel_a_y = int32(bsxfun(@plus, pixel_a_y_lmcoord, Tr_Data{s}.intermediate_shapes{stage}(lmarkID, 2, k)));
    
    pixel_b_x = int32(bsxfun(@plus, pixel_b_x_lmcoord, Tr_Data{s}.intermediate_shapes{stage}(lmarkID, 1, k)));
    pixel_b_y = int32(bsxfun(@plus, pixel_b_y_lmcoord, Tr_Data{s}.intermediate_shapes{stage}(lmarkID, 2, k)));
    
    width = (Tr_Data{s}.width);
    height = (Tr_Data{s}.height);

    pixel_a_x = max(1, min(pixel_a_x, width));
    pixel_a_y = max(1, min(pixel_a_y, height));
    
    pixel_b_x = max(1, min(pixel_b_x, width));
    pixel_b_y = max(1, min(pixel_b_y, height));
    
    pdfeats(:, i) = double(Tr_Data{s}.img_gray(pixel_a_y + (pixel_a_x-1)*height)) - double(Tr_Data{s}.img_gray(pixel_b_y + (pixel_b_x-1)*height));
       %./ double(Tr_Data{s}.img_gray(pixel_a_y + (pixel_a_x-1)*height)) + double(Tr_Data{s}.img_gray(pixel_b_y + (pixel_b_x-1)*height));
    
    % drawshapes(Tr_Data{s}.img_gray, [pixel_a_x pixel_a_y pixel_b_x pixel_b_y]);
    % hold off;
    
    shapes_residual(i, :) = Tr_Data{s}.shapes_residual(lmarkID, :, k);
end

E_x_2 = mean(shapes_residual(:, 1).^2);
E_x = mean(shapes_residual(:, 1));

E_y_2 = mean(shapes_residual(:, 2).^2);
E_y = mean(shapes_residual(:, 2));
% 
var_overall = length(ind_samples)*((E_x_2 - E_x^2) + (E_y_2 - E_y^2));

% var_overall = length(ind_samples)*(var(shapes_residual(:, 1)) + var(shapes_residual(:, 2)));

% max_step = min(length(ind_samples), params.max_numthreshs);
% step = floor(length(ind_samples)/max_step);
max_step = 1;

var_reductions = zeros(params.max_numfeats(stage), max_step);
thresholds = zeros(params.max_numfeats(stage), max_step);

[pdfeats_sorted] = sort(pdfeats, 2);

% shapes_residual = shapes_residual(ind, :);

for i = 1:params.max_numfeats(stage)
    % for t = 1:max_step
    t = 1;
    ind = ceil(length(ind_samples)*(0.5 + 0.9*(rand(1) - 0.5)));
        threshold = pdfeats_sorted(i, ind);  % pdfeats_sorted(i, t*step); % 
        thresholds(i, t) = threshold;
        ind_lc = (pdfeats(i, :) < threshold);
        ind_rc = (pdfeats(i, :) >= threshold);
        
        % figure, hold on, plot(shapes_residual(ind_lc, 1), shapes_residual(ind_lc, 2), 'r.')
        % plot(shapes_residual(ind_rc, 1), shapes_residual(ind_rc, 2), 'g.')
        % close;
        % compute 
        
        E_x_2_lc = mean(shapes_residual(ind_lc, 1).^2);
        E_x_lc = mean(shapes_residual(ind_lc, 1));
        
        E_y_2_lc = mean(shapes_residual(ind_lc, 2).^2);
        E_y_lc = mean(shapes_residual(ind_lc, 2));
        
        var_lc = (E_x_2_lc + E_y_2_lc)- (E_x_lc^2 + E_y_lc^2);
        
        E_x_2_rc = (E_x_2*length(ind_samples) - E_x_2_lc*sum(ind_lc))/sum(ind_rc);
        E_x_rc = (E_x*length(ind_samples) - E_x_lc*sum(ind_lc))/sum(ind_rc);
        
        E_y_2_rc = (E_y_2*length(ind_samples) - E_y_2_lc*sum(ind_lc))/sum(ind_rc);
        E_y_rc = (E_y*length(ind_samples) - E_y_lc*sum(ind_lc))/sum(ind_rc);
        
        var_rc = (E_x_2_rc + E_y_2_rc)- (E_x_rc^2 + E_y_rc^2);
        
        var_reduce = var_overall - sum(ind_lc)*var_lc - sum(ind_rc)*var_rc;
        
        % var_reduce = var_overall - sum(ind_lc)*(var(shapes_residual(ind_lc, 1)) + var(shapes_residual(ind_lc, 2))) - sum(ind_rc)*(var(shapes_residual(ind_rc, 1)) + var(shapes_residual(ind_rc, 2)));
        var_reductions(i, t) = var_reduce;
    % end
    % plot(var_reductions(i, :));
end

[~, ind_colmax] = max(var_reductions);
ind_max = 1;

%{
if var_max <= 0
    isvalid = 0;
else
    isvalid = 1;
end
%}
isvalid = 1;

thresh =  thresholds(ind_colmax(ind_max), ind_max);

feat   = [anglepairs(ind_colmax(ind_max), :) radiuspairs(ind_colmax(ind_max), :)];

lcind = ind_samples(find(pdfeats(ind_colmax(ind_max), :) < thresh));
rcind = ind_samples(find(pdfeats(ind_colmax(ind_max), :) >= thresh));

end

