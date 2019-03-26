%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Output param:
%   @m => The sparse matrix of laplacian matrix
%
%   Amos.zhu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L,Jac,Lmask]=LaplacianMatrix(mask)

newmask=logical(zeros(size(mask)));

[rows,cols]=size(mask);

offsetm=maskoffset(mask);
noofPixelsUsed=sum(mask(:));

count=1;
jidx_v=zeros(noofPixelsUsed*5,1); %jidx_v_count=1; % Jacobian matrix variable colume idx
jidx_f=zeros(noofPixelsUsed*5,1); %jidx_f_count=1; % Jacobian martrix function row idx
jidx_value=ones(noofPixelsUsed*5,1);

% jidx_v=[]; % Jacobian matrix variable colume idx
% jidx_f=[]; % Jacobian martrix function row idx

for m=1:rows
    for n=1:cols
        if mask(m,n)
            if n-1>0&&n+1<=cols&&m-1>0&& m+1<=rows
                if mask(m,n-1)&&mask(m,n+1)&&mask(m-1,n)&&mask(m+1,n)
                    newmask(m,n)=1;
                end
            end
        end
    end
end

offsetnm=maskoffset(newmask);
noofFunctions=sum(newmask(:));

for m=1:rows
    for n=1:cols
        if newmask(m,n)
            
            fidx=(n-1)*rows+m-offsetnm(m,n); % jacobian function idx;
            vidx=(n-1)*rows+m-offsetm(m,n); % jacobian variable idx;
            jidx_v(count)=vidx; jidx_f(count)=fidx;jidx_value(count)=-4;count=count+1;
            
            vidx=(n-2)*rows+m-offsetm(m,n-1); % jacobian variable idx;
            jidx_v(count)=vidx; jidx_f(count)=fidx;jidx_value(count)=1;count=count+1;
            vidx=n*rows+m-offsetm(m,n+1); % jacobian variable idx;
            jidx_v(count)=vidx; jidx_f(count)=fidx;jidx_value(count)=1;count=count+1;
            vidx=(n-1)*rows+m-1-offsetm(m-1,n); % jacobian variable idx;
            jidx_v(count)=vidx; jidx_f(count)=fidx;jidx_value(count)=1;count=count+1;
            vidx=(n-1)*rows+m+1-offsetm(m+1,n); % jacobian variable idx;
            jidx_v(count)=vidx; jidx_f(count)=fidx;jidx_value(count)=1;count=count+1;
        end
    end
end


jidx_f=jidx_f(1:count-1);
jidx_v=jidx_v(1:count-1);
jidx_value=jidx_value(1:count-1);
L=sparse(jidx_f,jidx_v,jidx_value,noofFunctions,noofPixelsUsed);
Jac=sparse(jidx_f,jidx_v,ones(length(jidx_v),1),noofFunctions,noofPixelsUsed);
Lmask=newmask;
end