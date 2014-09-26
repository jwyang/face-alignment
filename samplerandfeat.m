function anglepairs = samplerandfeat(num_feats)
%SAMPLERANDFEAT Summary of this function goes here
%   Function: generate the locations of pixel pairs randomly
%   Detailed explanation goes here
%   Input:
%        num_feats: number of features
%        max_radius: the maximum radius of local region
%   Output:
%         anglepairs: the angles of pixel pairs

thetas_a = 2*pi*[0:1/(num_feats-1):1];
thetas_b = 2*pi*[0:1/(num_feats-1):1];

anglepairs = [thetas_a(randperm(length(thetas_a)))' thetas_b(randperm(length(thetas_b)))'];

end

