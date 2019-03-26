%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Output param:
%   @Jac => The sparse matrix of jacobian matrix
%           A skew version of Identity matrix as well
%
%   Amos.zhu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Jac=JacobPattern(mask,baseMask)

rows=size(mask,1);
colums=size(mask,2);

offsetRow=maskoffset(mask);
offsetCol=maskoffset(baseMask);

noofRowPixels=sum(mask(:));
noofColPixels=sum(baseMask(:));
count=1;

jidx_v=zeros(noofRowPixels,1); %jidx_v_count=1; % Jacobian matrix variable colume idx
jidx_f=zeros(noofRowPixels,1); %jidx_f_count=1; % Jacobian martrix function row idx
jidx_value=ones(noofRowPixels,1);

% jidx_v=[]; % Jacobian matrix variable colume idx
% jidx_f=[]; % Jacobian martrix function row idx

for m=1:rows
    for n=1:colums
        if mask(m,n)
            row_idx=(n-1)*rows+m-offsetRow(m,n); % jacobian function idx;
            col_idx=(n-1)*rows+m-offsetCol(m,n); % jacobian variable idx;
            jidx_v(count)=col_idx; jidx_f(count)=row_idx;jidx_value(count)=1;count=count+1;
      
        end
    end
end


jidx_f=jidx_f(1:count-1);
jidx_v=jidx_v(1:count-1);
jidx_value=jidx_value(1:count-1);
Jac=sparse(jidx_f,jidx_v,jidx_value,noofRowPixels,noofColPixels);

end