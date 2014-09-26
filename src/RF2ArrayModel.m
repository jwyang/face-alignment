function RF2ArrayModel(LBFRegModel)
if 0
    model_path = '..\Models\LBFRegModel_afw_lfpw_helen_5.mat';
    load(model_path)
end

params.max_raio_radius = [0.3 0.3 0.2 0.15 0.1 0.1 0.08 0.08 0.06 0.05];
M = LBFRegModel;
num_stage = size(M.ranf,1);
num_point = min( size(M.Ws{1}) )/2;
num_tree_per_point = size(M.ranf{1},2);
tree_depth = max( M.ranf{1,1}{1}.depth ) - 1;
node_step = 5;

num_node = 2^tree_depth-1;
num_leaf = 2^tree_depth;
dim_tree = node_step*num_node;
num_tree_per_stage = num_point*num_tree_per_point;
num_tree_total = num_stage*num_point*num_tree_per_point;
dim_feat = num_leaf*num_tree_per_stage;

%% header
precision_byte = 4;
header_length = 40;
Header = zeros(1,header_length);
% store info
Header(1) = header_length; 
Header(2) = precision_byte;  %element_byte bytes
Header(3) = num_point*2;   %mean shape element
Header(4) = num_stage*num_point*num_tree_per_point*dim_tree;  %RF element
Header(5) = num_tree_total*num_leaf*num_point*2;  %W element

%mean shape info
Header(11) = num_point;

% RF info
Header(21) = num_stage;
Header(22) = num_point;
Header(23) = num_tree_per_point;
Header(24) = tree_depth;
Header(25) = node_step;

% W info
Header(31) = num_stage;
Header(32) = dim_feat;
Header(33) = num_point*2;


RF = zeros(num_tree_total,dim_tree);
for stage=1:num_stage
    fprintf('stage=%d\n',stage);
    for p=1:num_point
        for t=1:num_tree_per_point
            Tree = M.ranf{stage}{p, t};
            assert( Tree.num_leafnodes==2^tree_depth );
            anglepairs = Tree.feat(:,1:2);
            radiuspairs = Tree.feat(:,3:4);
            ax = cos(anglepairs(:, 1)).*radiuspairs(:, 1)*params.max_raio_radius(stage);
            ay = sin(anglepairs(:, 1)).*radiuspairs(:, 1)*params.max_raio_radius(stage);
            bx = cos(anglepairs(:, 2)).*radiuspairs(:, 2)*params.max_raio_radius(stage);
            by = sin(anglepairs(:, 2)).*radiuspairs(:, 2)*params.max_raio_radius(stage);
            ax = ax(1:Tree.num_leafnodes-1);
            ay = ay(1:Tree.num_leafnodes-1);
            bx = bx(1:Tree.num_leafnodes-1);
            by = by(1:Tree.num_leafnodes-1);
            th = Tree.thresh(1:Tree.num_leafnodes-1);
            temp = [ax ay bx by th];
            temp = reshape(temp',1,numel(temp));
            k = (stage-1)*num_tree_per_stage + (p-1)*num_tree_per_point + t;
            RF(k,:) = temp;
        end
    end 
end

W = zeros(dim_feat*num_stage,num_point*2);
for stage=1:num_stage
    temp = M.Ws{stage};
    W(dim_feat*(stage-1)+1:dim_feat*stage,:) = temp;
end

Header = single(Header);
if precision_byte==4  
    RF = single(RF);
    W = single(W);
elseif precision_byte==8  
    RF = double(RF);
    W = double(W);
end
save('..\Models\Header.mat','Header');
save('..\Models\RF.mat','RF');
save('..\Models\W.mat','W');
fprintf('write success.\n')
end

%     pixel_a_x_imgcoord = cos(anglepairs(:, 1)).*radiuspairs(:, 1)*params.max_raio_radius(stage)*Tr_Data{s}.bbox.width;
%     pixel_a_y_imgcoord = sin(anglepairs(:, 1)).*radiuspairs(:, 1)*params.max_raio_radius(stage)*Tr_Data{s}.bbox.height;
%     
%     pixel_b_x_imgcoord = cos(anglepairs(:, 2)).*radiuspairs(:, 2)*params.max_raio_radius(stage)*Tr_Data{s}.bbox.width;
%     pixel_b_y_imgcoord = sin(anglepairs(:, 2)).*radiuspairs(:, 2)*params.max_raio_radius(stage)*Tr_Data{s}.bbox.height;
    
    