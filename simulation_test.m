% Simulation testing
% First, randomly generate a collection of cirles, each has a confidence
% score. These circles heavily ovelapp. The task is to select a
% subset of best circles (highest scores) such that no pair of circles
% overlap (w.r.t an threshold ie 0.25). This problem has a wide range of
% applications such as object detetion, each object is represented as a
% bounding box.

clear all
close all

%% Number of cirles to be generated
N = 1000;
circles = zeros(N,3);

%% Random centers
circles(:,1:2) = (0.1*N).*rand(N,2);

%% Random radii
circles(:,3) = rand(N,1) + 3;

%% Random confidence
circles(:,4) = rand(N,1);

%% Ploting data
subplot(1,2,1)
for i=1:N
    x = circles(i,1);
    y = circles(i,2);
    r = circles(i,3);
    w = circles(i,4);
    draw_circle(x,y,r, 'b', 0.5);
    hold on;
end
axis equal
axis([0 N*0.1 0 N*0.1]);
title('Input circles');

disp('Press any key to continue.')
pause

%% Unary cost
unary_energy = -circles(:,4);

% Compute pairwise overlaps
pairwise_energy  = area_intersect_circle_analytical(circles(:,1:3));
pairwise_energy(1:N+1:N*N) = 0;
pairwise_energy = (pairwise_energy + pairwise_energy')/2;

%% Overlap threshold. Pairs of circles overlap more than a threshold are prohibited.
ov_th = 0.01;

%% Pairwise cost
pairwise_energy(pairwise_energy>ov_th) = 1e5;

%% Run optimisation to select the best circles.
tic
[labels, E] = lsa_tr_optimisation_tpham(unary_energy, pairwise_energy);
toc

%% Display result
subplot(1,2, 2)
for i=1:N
    x = circles(i,1);
    y = circles(i,2);
    r = circles(i,3);
    if labels(i) == 1   
        draw_circle(x,y,r,'r', 1.25);
        hold on;
    else
        draw_circle(x,y,r,'c', 0.25);
        hold on;
    end
end
axis equal
axis([0 N*0.1 0 N*0.1]);
title('Red circles are selected. Cyan circles are rejected.');