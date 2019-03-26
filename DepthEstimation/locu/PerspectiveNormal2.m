%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   This form also take the rotation and translation into
%   account;
%
%   Dizhong.zhu 10/May/2018
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [n]=PerspectiveNormal2(z0,fmask,varmask,Dx,Dy,P)
%% Preparing
[rows,cols]=size(varmask);
Z=nan(rows,cols);
Z(varmask)=z0;

c1=P(1,4)*P(2,2)*P(3,1);
c2=P(1,2)*P(2,4)*P(3,1);
c3=P(1,4)*P(2,1)*P(3,2);
c4=P(1,1)*P(2,4)*P(3,2);
c5=P(1,2)*P(2,1)*P(3,4);
c6=P(1,1)*P(2,2)*P(3,4);
C1=c1-c2-c3+c4+c5-c6;
c7=P(1,3)*P(2,2)*P(3,1);
c8=P(1,2)*P(2,3)*P(3,1);
c9=P(1,3)*P(2,1)*P(3,2);
c10=P(1,1)*P(2,3)*P(3,2);
c11=P(1,2)*P(2,1)*P(3,3);
c12=P(1,1)*P(2,2)*P(3,3);
C2=c10+c11-c12+c7-c8-c9;

% x and y range
[x,y]=meshgrid(1:cols,1:rows);
denorm=-P(1,1)*P(2,2)+P(2,2)*P(3,1)*x+...
    P(1,1)*P(3,2)*y+P(1,2)*(P(2,1)-P(3,1)*y);

Zx=NaN(rows,cols);
Zy=NaN(rows,cols);

% pnx = NaN(rows,cols);
% pny = NaN(rows,cols);
PN  = NaN(rows,cols,3);
%% Forward computation
% B 1. Partial of z
Zx(fmask)=Dx*Z(varmask);
Zy(fmask)=Dy*Z(varmask);

% B 2. Normal of components x,y,z
pnx=-((-P(1,1)+P(3,1)*x).*Zx+(-P(2,1)+P(3,1)*y).*Zy).*(C1+C2*Z)./denorm.^2;
pny=((P(1,2)-P(3,2)*x).*Zx+(P(2,2)-P(3,2)*y).*Zy).*(C1+C2*Z)./denorm.^2;
pnz=-(C1+denorm.*((-P(1,3)+P(3,3)*x).*Zx+(-P(2,3)+P(3,3)*y).*Zy)+C2*Z).*(C1+C2*Z)./denorm.^3;

PN(:,:,1)=pnx;
PN(:,:,2)=pny;
PN(:,:,3)=pnz;

% B 3. compute the squre of the magnitude
sumN=pnx.^2+pny.^2+pnz.^2;

% B 4. magnitude
mag=sqrt(sumN);
zero_idx=(mag==0);
% B 5. Reciprocal of the magnitude
magr1=1./mag;
magr1(zero_idx)=0;
% B 6. Normalize the normal
N=PN.*magr1;

nx=N(:,:,1);
ny=N(:,:,2);
nz=N(:,:,3);
normN(:,1)=nx(fmask);
normN(:,2)=ny(fmask);
normN(:,3)=nz(fmask);

n=normN;
end