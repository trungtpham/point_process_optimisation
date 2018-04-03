function M = area_intersect_two_circles_analytical(G1,G2)
% Written and developed by Seyed Hamid Rezatofighi

% Compute the overlap area between 2 group of circles defined in two arrays.
% Computation is vectorized, and intersection area are computed an
% analytical way.
%   
% Input: Each circles data presented in an array G of three columns.
%        G1 and G2 contains parameters of the n and m circles respectively
%           . G(1:n,1) - x-coordinate of the center of circles,
%           . G(1:n,2) - y-coordinate of the center of circles,
%           . G(1:n,3) - radii of the circles 
%        Each row of the array contains the information for one circle.
% 
% 
% 
% Output: Square matrix M(n,m) containing intersection areas between
% circles
%         M(i,j) contains the intersection area between circle i from G1 & 
%         circle j from  G2
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For 2 circles i & j, three cases are possible depending on the distance
% d(i,j) of the centers of the circles i and j.
%   Case 1: Circles i & j do not overlap, there is no overlap area M(i,j)=0;
%             Condition: d(i,j)>= ri+rj
%             M(i,j) = 0;
%   Case 2: Circles i & j fully overlap, the overlap area has to be computed.
%             Condition: d(i,j)<= abs(ri-rj)
%            M(i,j) = pi*min(ri,rj).^2
%   Case 3: Circles i & j partially overlap, the overlap area has to be computed
%            decomposing the overlap area.
%             Condition: (d(i,j)> abs(ri-rj)) & (d(i,j)<(ri+rj))
%            M(i,j) = f(xi,yi,ri,xj,yj,rj)
%                   = ri^2*arctan2(yk,xk)+ ...
%                     rj^2*arctan2(yk,d(i,j)-xk)-d(i,j)*yk
%             where xk = (ri^2-rj^2+d(i,j)^2)/(2*d(i,j))
%                   yk = sqrt(ri^2-xk^2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 


  
if nargin==2
    xy1 = G1(:,1:2);
    xy2 = G2(:,1:2);
    r1 = G1(:,3);
    r2 = G2(:,3); 
    
else
    error('The number of input arguments must be 2')
end


% Checking if there is any negative or null radius
if any(r1<=0)||any(r2<=0)
    warning('Circles with null or negative radius won''t be taken into account in the computation.')
    temp1 = (r1>0);
    temp2 = (r2>0);
    xy1 = xy1(temp1,:);
    xy2 = xy2(temp2,:);
    
    r1 = r1(temp1);
    r2 = r2(temp2);
end


% Computation of distance between all circles, which will allow to
% determine which cases to use.

D=pdist2(xy1,xy2);
SG1=size(r1,1);
SG2=size(r2,1);
[R2,R1] = meshgrid(r2,r1);

sumR=R1+R2;

difR=abs(R1-R2);

minA = pi*min(R1,R2).^2;

% Creating the resulting vector
M = zeros(SG1,SG2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Case 2: Circles i & j fully overlap
% One of the circles is inside the other one.
C1    = (D<=difR);
M(C1) = pi*min(R1(C1),R2(C1)).^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Case 3: Circles i & j partially overlap
% Partial intersection between circles i & j
C2 = (D>difR)&(D<sumR);
Xi=zeros(SG1,SG2);Yi=zeros(SG1,SG2);
% Computation of the coordinates of one of the intersection points of the
% circles i & j
Xi(C2) = (R2(C2).^2-R1(C2).^2+D(C2).^2)./(2*D(C2));
Yi(C2) = sqrt(R2(C2).^2-Xi(C2).^2);
% Computation of the partial intersection area between circles i & j
M(C2) = R2(C2).^2.*atan2(Yi(C2),Xi(C2))+...
          R1(C2).^2.*atan2(Yi(C2),(D(C2)-Xi(C2)))-...
          D(C2).*Yi(C2);
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
% Creating the lower part of the matrix
%M = M + tril(M',-1);

M  = M./minA;



