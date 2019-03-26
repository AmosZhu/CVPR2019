% Perspective version
function [ALBEDO,Ndis,Z]=EstimateAlbedoGuideByCorrectedNspec2(theta,phi,Iun,N_guide,fmask,varmask,specmask,Dx,Dy,K,R,t,l,cpts,cpt_pos,smooth,L,guide_weight)

P=K*[R t];
l=l./norm(l);
noofValidFuntions=sum(fmask(:));
noofValidPixels=sum(varmask(:));

% Dialte the specular mask a little bit
% dilateRadius=4;
% se=strel('disk',dilateRadius,0);
% specmask=imdilate(specmask,se);
specmask=fmask&specmask;
diffuseMask=fmask&(~specmask);

indices=1:noofValidFuntions;
noofValidDiffuse=sum(diffuseMask(:));
noofValidSpec=sum(specmask(:));

[rows,cols,nchannels]=size(Iun);
Iunv=zeros(noofValidFuntions,nchannels);
for i=1:nchannels
    Ic=Iun(:,:,i);
    Iunv(:,i)=Ic(fmask);
end
thetafv=theta(fmask);

T=[-1 0 0;0 -1 0;0 0 1];

%% A. Correcting normal from theta and phi by interpolation depth
% 1. get valid normal from guide normal
nx=N_guide(:,:,1);
ny=N_guide(:,:,2);
nz=N_guide(:,:,3);
PN=[nx(fmask) ny(fmask) nz(fmask)];
figure;DisplayNormals(PN,Iun,fmask);title('Normal of the guide depth');

% 2. Compute the initial normal
nx=sin(theta(fmask)).*cos(phi(fmask));
ny=sin(theta(fmask)).*sin(phi(fmask));
nz=cos(theta(fmask));

N=[nx ny nz];
% N=N./repmat(sqrt(sum(N.^2,2)),[1 3]);

figure;DisplayNormals(N,Iun,fmask); title('Normals recover from theta&phi');
figure;DisplayNormals(N*T,Iun,fmask); title('Normals recover from theta&phi with T');
% 3. The diffuse normal should close to the PN   Just for testing, for
% disabmbiugation has been done through step2 already.
% idx=sum((N*T.*PN),2)>sum((N.*PN),2);
% N(idx,:)=N(idx,:)*T;
% figure;DisplayNormals(N,Iun,fmask); title('Normals after disambiguation');

% Ndis=N;
Ndis=PN;
N=PN;
N_spec=N(specmask(fmask),:);

row_idx=1:noofValidFuntions;
col_idx=1:noofValidFuntions;

%% B. Prepare the (1).matrix for linearising the perspective normals. (2) the view directions

% (1) Prepare transformed perspective normal matrix
fx=K(1,1);fy=K(2,2);x0=K(1,3);y0=K(2,3);
[y_pos,x_pos]=find(fmask==1);

% Numerical robust version, times fx,fy in left side
p1=sparse(row_idx,col_idx,ones(noofValidFuntions,1)*(-fx),noofValidFuntions,3*noofValidFuntions);
p2=sparse(row_idx,col_idx+noofValidFuntions,ones(noofValidFuntions,1)*(-fy),noofValidFuntions,3*noofValidFuntions);
p3=[sparse(row_idx,col_idx,(x_pos-x0),noofValidFuntions,noofValidFuntions)...
    sparse(row_idx,col_idx,(y_pos-y0),noofValidFuntions,noofValidFuntions)...
    sparse(row_idx,col_idx,ones(noofValidFuntions,1),noofValidFuntions,noofValidFuntions)];

PT=[p1;p2;p3];

PT=PT*[-Dx;-Dy;JacobPattern(fmask,varmask)];

% (2) Prepare for the view direction
% v=[0 0 1];
v=[(x_pos-x0)/fx (y_pos-y0)/fy ones(noofValidFuntions,1)];
v=v./(sqrt(sum(v.^2,2)));

% l_est=LightEstimation(N_spec,v(specmask(fmask),:));
% l=l_est./norm(l_est);

H=(v+l)/2;
H=H./(sqrt(sum(H.^2,2)));

Vf=nan(rows,cols,3);
Vd=zeros(noofValidDiffuse,3);
Hf=nan(rows,cols,3);
Hs=zeros(noofValidSpec,3);
for i=1:3
    vv=nan(rows,cols);
    vv(fmask)=v(:,i);
    Vf(:,:,i)=vv;
    Vd(:,i)=vv(diffuseMask);
    hh=nan(rows,cols);
    hh(fmask)=H(:,i);
    Hf(:,:,i)=hh;
    Hs(:,i)=hh(specmask);
end

[Dax,Day]=DepthGradient2(diffuseMask,fmask);

disp('Estimating albedo of the object!');

niter=0;
maxIter=1;
converged=false;
row_idx_A=1:noofValidDiffuse;
col_idx_A=indices(diffuseMask(fmask));
while ~converged&&niter<maxIter
    % A. Given normal estimate albedo
    
    % diffuse reflectance and albedo constraints
    Nl=sparse(row_idx_A,col_idx_A,N(diffuseMask(fmask),:)*l',noofValidDiffuse,noofValidFuntions);
    RHS=Iunv(diffuseMask(fmask),:);

    AA=[Dax;Day]; w=3;
    ARHS=[Dax; Day]*Iunv;
    ARHS(ARHS<0.01)=0;
    
    % Compute the albedo
    LHS=[Nl;w*AA]; RHS=[RHS;w.*ARHS];
    albedo=LHS\RHS;
    ALBEDO=zeros(rows,cols,nchannels);
    for i=1:nchannels
        a=zeros(size(fmask));
        a(fmask)=albedo(:,i);
        a(specmask)=nan;
        a=fillmissing(a,'pchip');
        ALBEDO(:,:,i)=a;
    end
    ALBEDO=min(ALBEDO,1);
    ALBEDO=max(ALBEDO,0);
%     figure; imshow(ALBEDO)
%     figure;DisplayALBEDO(albedo,Iun,fmask);title('Estimated albedo');
    
    % B. Given albedo estimate surface normal (no flip, in contrast with EstimateAlbedo function)
    row_idx=1:noofValidDiffuse;
    col_idx=indices(diffuseMask(fmask));
    % 1.a Diffuse: Intensity constraint
    w_a=1;
    LHS=[];
    RHS=[];
    for i=1:nchannels
        I_t=Iun(:,:,i);
        a_t=ALBEDO(:,:,i);
        
        LHS_i=w_a*[sparse(row_idx,col_idx,l(1)*a_t(diffuseMask).*cos(theta(diffuseMask))-I_t(diffuseMask).*Vd(:,1),noofValidDiffuse,noofValidFuntions)...
            sparse(row_idx,col_idx,l(2)*a_t(diffuseMask).*cos(theta(diffuseMask))-I_t(diffuseMask).*Vd(:,2),noofValidDiffuse,noofValidFuntions)...
            sparse(row_idx,col_idx,l(3)*a_t(diffuseMask).*cos(theta(diffuseMask))-I_t(diffuseMask).*Vd(:,3),noofValidDiffuse,noofValidFuntions)];
        LHS=[LHS;LHS_i];
    end
    RHS=zeros(nchannels*noofValidDiffuse,1);
    
    % 1.b Diffuse: Phase angle constraint
    w_phi=0.3;
    LHS = [LHS; w_phi*sparse(row_idx,col_idx,-sin(phi(diffuseMask)),noofValidDiffuse,noofValidFuntions)...
        w_phi*sparse(row_idx,col_idx,cos(phi(diffuseMask)),noofValidDiffuse,noofValidFuntions)...
        w_phi*sparse(1,1,0,noofValidDiffuse,noofValidFuntions)];
    RHS = [RHS; zeros(noofValidDiffuse,1)];
    
    % 1.c Specular: Phase angle constraint
    phi_spec=mod(phi+pi/2,pi);
    w_s=w_phi;
    row_idx=1:noofValidSpec;
    col_idx=indices(specmask(fmask));
    LHS = [LHS; w_s*sparse(row_idx,col_idx,-sin(phi_spec(specmask)),noofValidSpec,noofValidFuntions)...
        w_s*sparse(row_idx,col_idx,cos(phi_spec(specmask)),noofValidSpec,noofValidFuntions)...
        w_s*sparse(noofValidSpec,noofValidFuntions)];
    RHS = [RHS; zeros(noofValidSpec,1)];
    
    % 2. Normal from depth should be at the same direction to the guide
    % normal. Hopefully!.
    if ~exist('guide_weight','var') || isempty(guide_weight)
        w_n=1;
    else
        w_n=guide_weight(fmask);
    end

    w_n=w_n*0.5;
    row_idx=1:noofValidFuntions;
    col_idx=1:noofValidFuntions;
    N1_sparse=sparse(row_idx,col_idx,w_n.*N(:,1),noofValidFuntions,noofValidFuntions);
    N2_sparse=sparse(row_idx,col_idx,w_n.*N(:,2),noofValidFuntions,noofValidFuntions);
    N3_sparse=sparse(row_idx,col_idx,w_n.*N(:,3),noofValidFuntions,noofValidFuntions);
    zero_sparse=sparse(noofValidFuntions,noofValidFuntions);
    S_LHS = [zero_sparse -N3_sparse N2_sparse;...
        N3_sparse zero_sparse -N1_sparse;...
        -N2_sparse N1_sparse zero_sparse];
    
    LHS=[LHS; S_LHS];
    RHS = [RHS; zeros(3*noofValidFuntions,1)]; % Takes all normal
%     RHS = [RHS; w_n*N(diffuseMask(fmask),1);w_n*N(diffuseMask(fmask),2);w_n*N(diffuseMask(fmask),3)]; % Take diffuse normals only
    
    LHS=LHS*PT;
    
    % cloud points constraint
    w_cp=0.1;
    noofcps=size(cpts,1);
    LHS=[LHS;w_cp*sparse(1:noofcps,cpt_pos,ones(noofcps,1),noofcps,noofValidPixels)];
    RHS=[RHS; w_cp*cpts(:,3)];
    
    
    if smooth
        w=100;
        LHS=[LHS;w*L];
        RHS=[RHS;zeros(size(L,1),1)];
    end
    
    zeta0=LHS\RHS;
    zeta0=zeta0-min(zeta0(:));
    Zeta=nan(size(fmask));
    Zeta(varmask)=zeta0;
    Zeta(~fmask)=nan;
    Z=Zeta;
    
    PN=PerspectiveNormal2(Z(varmask),fmask,varmask,Dx,Dy,P);
    PN=-PN;
    PN(:,3)=-PN(:,3);
    figure;DisplayNormals(PN,Iun,fmask); title('Normal after albedo process');
    
    %% back projection
    zo=Z;
    [rows,cols]=size(zo);   
    Nnew=PN;
    residual=norm(cross(Nnew,N));
    disp(['Residual = ' num2str(residual)]);
    if residual<eps
        converged=true;
    end
    
    N=Nnew;
    niter=niter+1;
    if niter>maxIter
        break;
    end
    
end


end