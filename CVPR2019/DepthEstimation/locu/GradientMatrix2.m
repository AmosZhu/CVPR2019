%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Compute the gradient matrix and surface normals
%
%   Input para:
%       @z: depth function
%       @mask: compute the gradient based on this mask
%       @nsamplePoint: 2,3,5. can add more, but in future
%       @method: direction of forward
%
%   Output para:
%       @N: Normal from depth
%       @Dx: gradient matrix of z in x direction
%       @Dy: gradient matrix of z in y direction
%       @newmask: updated mask
%
%   DIZHONG ZHU 06/Nov/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [N,Dx,Dy,newmask]=GradientMatrix2(z,mask,method,order)
if nargin<4
    order=2;
elseif nargin<3
    method='ff';
    order=2;
end

[rows,cols]=size(mask);

if order==2
    [Dx,Dy,newmask]=finiteDifference2(mask);
elseif order==3
    [Dx,Dy,newmask]=finiteDifference3(mask);
elseif order==5
    [Dx,Dy,newmask]=finiteDifference5(mask);
else
    error('invalid orders');
end

if strcmp(method(1),'b')
    Dx=-Dx;
end

if strcmp(method(2),'b')
    Dy=-Dy;
end

N=nan(rows,cols,3);
p=nan(rows,cols);
q=nan(rows,cols);

p(newmask)=Dx*z(mask);
q(newmask)=Dy*z(mask);

N(:,:,1)=-p;
N(:,:,2)=-q;
N(:,:,3)=1;
% Normalise to unit vectors
norms = sqrt(sum(N.^2,3));
N = N./repmat(norms,[1 1 3]);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   take 2 samples
%
%   df(x)=f(x+1)-f(x)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Dx,Dy,newmask]=finiteDifference2(mask)
[rows,cols] = size(mask);
rows=rows+2;cols=cols+2;N=rows*cols;
maskpad=zeros(rows,cols);
maskpad(2:end-1,2:end-1)=mask;

% valid pixel use to form a new mask
validPixels=logical(zeros(rows,cols));

for m=1:rows
    for n=1:cols
        if maskpad(m,n)&&maskpad(m,n-1)&&maskpad(m-1,n)
            validPixels(m,n)=true;
        end
    end
end

newmask=maskpad&validPixels;
disp(['Removing ' num2str(sum(mask(:))-sum(newmask(:))) ' pixels']);

idx=find(validPixels);
idx_v=find(maskpad);

onesvec=ones(length(idx),1);
Dx=sparse(idx,idx-rows,-onesvec,N,N)+sparse(idx,idx,onesvec,N,N);
Dy=sparse(idx,idx-1,-onesvec,N,N)+sparse(idx,idx,onesvec,N,N);
Dx=Dx(idx,idx_v);
Dy=Dy(idx,idx_v);


newmask = newmask(2:end-1,2:end-1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   take 3 samples
%
%   df(x)={f(x+1)-f(x-1)}/2
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Dx,Dy,newmask]=finiteDifference3(mask)
[rows,cols] = size(mask);
rows=rows+2;cols=cols+2;N=rows*cols;
maskpad=zeros(rows,cols);
maskpad(2:end-1,2:end-1)=mask;

% valid pixel use to form a new mask
validPixels=logical(zeros(rows,cols));

for m=1:rows
    for n=1:cols
        if maskpad(m,n)&&maskpad(m,n-1)&&maskpad(m-1,n)&&maskpad(m,n+1)&&maskpad(m+1,n)
            validPixels(m,n)=true;
        end
    end
end

newmask=maskpad&validPixels;
disp(['Removing ' num2str(sum(mask(:))-sum(newmask(:))) ' pixels']);

idx=find(validPixels);
idx_v=find(maskpad);

onesvec=ones(length(idx),1);
Dx=sparse(idx,idx-rows,-onesvec,N,N)+sparse(idx,idx+rows,onesvec,N,N);
Dy=sparse(idx,idx-1,-onesvec,N,N)+sparse(idx,idx+1,onesvec,N,N);
Dx=Dx(idx,idx_v)/2;
Dy=Dy(idx,idx_v)/2;

newmask = newmask(2:end-1,2:end-1);
end

function [Dx,Dy,newmask]=finiteDifference3_2(mask)
[rows,cols] = size(mask);
rows=rows+2;cols=cols+2;N=rows*cols;
maskpad=zeros(rows,cols);
maskpad(1:end-2,1:end-2)=mask;

% valid pixel use to form a new mask
validPixels=logical(zeros(rows,cols));

for m=1:rows
    for n=1:cols
        if maskpad(m,n)&&maskpad(m,n-1)&&maskpad(m-1,n)&&maskpad(m,n-2)&&maskpad(m-2,n)
            validPixels(m,n)=true;
        end
    end
end

newmask=maskpad&validPixels;
disp(['Removing ' num2str(sum(mask(:))-sum(newmask(:))) ' pixels']);

idx=find(validPixels);
idx_v=find(maskpad);

onesvec=ones(length(idx),1);
Dx=sparse(idx,idx-2*rows,onesvec,N,N)-4*sparse(idx,idx-rows,onesvec,N,N)+...
    3*sparse(idx,idx,onesvec,N,N);
Dy=sparse(idx,idx-2,onesvec,N,N)-4*sparse(idx,idx-1,onesvec,N,N)+...
    3*sparse(idx,idx,onesvec,N,N);
Dx=Dx(idx,idx_v)/2;
Dy=Dy(idx,idx_v)/2;

newmask = newmask(1:end-2,1:end-2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   take 5 samples
%
%   df(x)={f(x+1)-f(x-1)}/12
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Dx,Dy,newmask]=finiteDifference5(mask)
[rows,cols] = size(mask);
rows=rows+4;cols=cols+4;N=rows*cols;
maskpad=zeros(rows,cols);
maskpad(3:end-2,3:end-2)=mask;

% valid pixel use to form a new mask
validPixels=logical(zeros(rows,cols));

for m=1:rows
    for n=1:cols
        if maskpad(m,n)&&maskpad(m,n-1)&&maskpad(m-1,n)&&...
                maskpad(m,n-2)&&maskpad(m-2,n)&&...
                maskpad(m,n+1)&&maskpad(m+1,n)&&...
                maskpad(m,n+2)&&maskpad(m+2,n)
            validPixels(m,n)=true;
        end
    end
end

newmask=maskpad&validPixels;
disp(['Removing ' num2str(sum(mask(:))-sum(newmask(:))) ' pixels']);

idx=find(validPixels);
idx_v=find(maskpad);

onesvec=ones(length(idx),1);

Dx=sparse(idx,idx-2*rows,onesvec,N,N)-8*sparse(idx,idx-rows,onesvec,N,N)+...
    8*sparse(idx,idx+rows,onesvec,N,N)-sparse(idx,idx+2*rows,onesvec,N,N);
Dy=sparse(idx,idx-2,-onesvec,N,N)-8*sparse(idx,idx-1,onesvec,N,N)+...
    8*sparse(idx,idx+1,onesvec,N,N)-sparse(idx,idx-2,-onesvec,N,N);
    
Dx=Dx(idx,idx_v)/12;
Dy=Dy(idx,idx_v)/12;
newmask = newmask(3:end-2,3:end-2);

end

function [Dx,Dy,newmask]=finiteDifference5_2(mask)
[rows,cols] = size(mask);
rows=rows+4;cols=cols+4;N=rows*cols;
maskpad=zeros(rows,cols);
maskpad(1:end-4,1:end-4)=mask;

% valid pixel use to form a new mask
validPixels=logical(zeros(rows,cols));

for m=1:rows
    for n=1:cols
        if maskpad(m,n)&&maskpad(m,n-1)&&maskpad(m-1,n)&&...
                maskpad(m,n-2)&&maskpad(m-2,n)&&...
                maskpad(m,n-3)&&maskpad(m-3,n)&&...
                maskpad(m,n-4)&&maskpad(m-4,n)
            validPixels(m,n)=true;
        end
    end
end

newmask=maskpad&validPixels;
disp(['Removing ' num2str(sum(mask(:))-sum(newmask(:))) ' pixels']);

idx=find(validPixels);
idx_v=find(maskpad);

onesvec=ones(length(idx),1);

Dx=3*sparse(idx,idx-4*rows,onesvec,N,N)-16*sparse(idx,idx-3*rows,onesvec,N,N)+...
    36*sparse(idx,idx-2*rows,onesvec,N,N)-48*sparse(idx,idx-rows,onesvec,N,N)+...
    25*sparse(idx,idx,onesvec,N,N);
Dy=3*sparse(idx,idx-4,onesvec,N,N)-16*sparse(idx,idx-3,onesvec,N,N)+...
    36*sparse(idx,idx-2,onesvec,N,N)-48*sparse(idx,idx-1,onesvec,N,N)+...
    25*sparse(idx,idx,onesvec,N,N);
    
Dx=Dx(idx,idx_v)/12;
Dy=Dy(idx,idx_v)/12;
newmask = newmask(1:end-4,1:end-4,:);
end