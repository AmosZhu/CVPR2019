close all;
clear;
clc;

addpath('locu')
addpath('../data') 

%% Load and set datas
load('horse_corrected_normal_Large.mat');  % Corrected normal from Graphic model
load('horse_disparity_large.mat');         % Disparity map from stereo
[rows,cols]=size(cam1.mask);

% Only use one view for reconstruction
CC=bwconncomp(cam1.mask);
varmask=logical(zeros(rows,cols));
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
varmask(CC.PixelIdxList{idx}) = 1;

K=cam1.cameraParam.IntrinsicMatrix';
P=cam1.P;
spec1=logical(cam1.specmask);
mask1=varmask;

[~,Dx,Dy,fmask]=GradientMatrix2(varmask,varmask,'ff');

dilateRadius=0;
se=strel('disk',dilateRadius,0);
spec1=imdilate(spec1,se); spec1=spec1&varmask;

l=cam1.light';
noofValidFuntions=sum(fmask(:));
noofValidPixels=sum(varmask(:));
[L,~]=LaplacianMatrix(mask1);

%% Get polarisation information and pointcloud constraint

rho_est=cam1.rho_est;
phi_est=cam1.phi_est;
Iun_est=cam1.Iun_est;

theta_est = rho_diffuse(rho_est,1.5);

[rows,cols,nchannels]=size(Iun_est);
Iunv=zeros(noofValidFuntions,nchannels);
for i=1:nchannels
    Ic=Iun_est(:,:,i);
    Iunv(:,i)=Ic(fmask);
end

offset=maskoffset(mask1);
cpt_pos=zeros(length(xyZ),1);
cpts=zeros(size(xyZ));
% diffuseMask=boundarymask(fmask);
diffuseMask=fmask&~spec1;
count=1;
for i=1:length(xyZ)
    x=round(xyZ(i,1));
    y=round(xyZ(i,2));
    if diffuseMask(y,x)==1
        loc=sub2ind(size(mask1),y,x)-offset(y,x);
        cpt_pos(count)=loc;
        cpts(count,:)=xyZ(i,:);
        count=count+1;
    end
end
cpts=cpts(1:count-1,:);
cpt_pos=cpt_pos(1:count-1);

%% Estimate albedo and depth

[albedo,N,Zr]=EstimateAlbedoGuideByCorrectedNspec2(theta_est,phi_est,Iun_est,double(normal1),fmask,mask1,spec1,Dx,Dy,K,eye(3,3),zeros(3,1),l,cpts,cpt_pos,true,L);
centroid=mean(Zr(fmask));
Zr(Zr<centroid*0.7)=nan;
Zr(Zr>centroid*1.3)=nan;

figure;imshow(albedo);title('estimated albedo map');
zo=Zr;
[rows,cols]=size(zo);
[x,y]=meshgrid(1:cols,1:rows);
pt_3d_rep=Depth2CloudPoint([x(:) y(:)],zo(:),P);

figure; surf(reshape(pt_3d_rep(:,1),[rows cols]),reshape(pt_3d_rep(:,2),[rows cols]),reshape(pt_3d_rep(:,3),[rows cols])); axis equal; axis off; grid off; shading interp;
ax = gca;               % get the current axis
ax.Clipping = 'off';    % turn clipping off
figure; warp(reshape(pt_3d_rep(:,1),[rows cols]),reshape(pt_3d_rep(:,2),[rows cols]),reshape(pt_3d_rep(:,3),[rows cols]),albedo); axis equal

