function [radiuspairs, anglepairs] = getproposals(num_proposals, radius_grid, angles_grid)
%GETPROPOSALS Summary of this function goes here
%   Detailed explanation goes here

num_radius = length(radius_grid);
num_angles = length(angles_grid);

Pro_a = randperm(num_radius*num_angles);
Pro_b = randperm(num_radius*num_angles);

exc = Pro_a == Pro_b;

Pro_a = Pro_a(exc == 0);
Pro_b = Pro_b(exc == 0);

Pro_a_choose = Pro_a(1:num_proposals);
Pro_b_choose = Pro_b(1:num_proposals);

id_radius_a = floor((Pro_a_choose - 1)/num_angles) + 1;
id_radius_b = floor((Pro_b_choose - 1)/num_angles) + 1;

id_angles_a = mod(Pro_a_choose, num_angles) + 1;
id_angles_b = mod(Pro_b_choose, num_angles) + 1;

radiuspairs = [radius_grid(id_radius_a) radius_grid(id_radius_b)];
anglepairs  = [angles_grid(id_angles_a) angles_grid(id_angles_b)];


end

