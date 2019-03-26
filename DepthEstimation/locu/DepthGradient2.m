%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%   Amos.Zhu add jacobian matrix and finite depth partial
%   derivative matrices
%
%   Output para:
%       @N: gradient of depth function
%       @Jac: Jacobian matrix of related gradient
%       @Gx: gradient matrix of z in x direction
%       @Gy: gradient matrix of z in y direction
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Gx,Gy,JAC]=DepthGradient2(mask,baseMask)


rows = size(mask,1);
cols = size(mask,2);

offsetRow=maskoffset(mask);
offsetCol=maskoffset(baseMask);
% Pad to avoid boundary problems
mask2 = zeros(rows+2,cols+2);
mask2(2:rows+1,2:cols+1)=mask;
mask = mask2;
clear mask2

baseMask2 = zeros(rows+2,cols+2);
baseMask2(2:rows+1,2:cols+1)=baseMask;
baseMask = baseMask2;
clear baseMask2;

offsetm2=zeros(rows+2,cols+2);
offsetm2(2:rows+1,2:cols+1)=offsetRow;
offsetRow=offsetm2;
clear offsetm2;

offsetm2=zeros(rows+2,cols+2);
offsetm2(2:rows+1,2:cols+1)=offsetCol;
offsetCol=offsetm2;
clear offsetm2;

rows = rows+2;
cols = cols+2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Dizhong.Zhu revised for testing or others
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
noofColPixels=sum(baseMask(:));
noofRowPixels=sum(mask(:));

% shrink after procedure finished
% No more than its double size, for forward or backward only
Gx_idx_val=zeros(noofRowPixels*2,3);  Gx_count=1;
Gy_idx_val=zeros(noofRowPixels*2,3);  Gy_count=1;

jidx_v=zeros(noofRowPixels*3,1); jidx_v_count=1; % Jacobian matrix variable colume idx
jidx_f=zeros(noofRowPixels*3,1); jidx_f_count=1; % Jacobian martrix function row idx

for row=1:rows
    for col=1:cols
        if mask(row,col)
            % Now decide which combination of neighbours are present
            % This determines which version of the numerical
            % approximation to the surface gradients will be used
            
            
            row_idx=(rows-2)*(col-2)+row-1-offsetRow(row,col); % jacobian function idx;
            col_idx=(rows-2)*(col-2)+row-1-offsetCol(row,col); % jacobian variable idx;
            
            jidx_v(jidx_v_count)=col_idx; jidx_v_count=jidx_v_count+1;
            jidx_f(jidx_f_count)=row_idx; jidx_f_count=jidx_f_count+1;
            
            % x direction
            if mask(row,col+1)
                % Only forward in X
                Gx_idx_val(Gx_count,:)=[row_idx,col_idx,-1]; Gx_count=Gx_count+1;
                col_idx=(rows-2)*(col-1)+row-1-offsetCol(row,col+1);
                Gx_idx_val(Gx_count,:)=[row_idx,col_idx,1]; Gx_count=Gx_count+1;
                jidx_v(jidx_v_count)=col_idx; jidx_v_count=jidx_v_count+1;
                jidx_f(jidx_f_count)=row_idx; jidx_f_count=jidx_f_count+1;
            elseif mask(row,col-1)
                % Only backward in X
                Gx_idx_val(Gx_count,:)=[row_idx,col_idx,1]; Gx_count=Gx_count+1;
                col_idx=(rows-2)*(col-3)+row-1-offsetCol(row,col-1);
                Gx_idx_val(Gx_count,:)=[row_idx,col_idx,-1]; Gx_count=Gx_count+1;
                jidx_v(jidx_v_count)=col_idx; jidx_v_count=jidx_v_count+1;
                jidx_f(jidx_f_count)=row_idx; jidx_f_count=jidx_f_count+1;
            end
            %y direction
            col_idx=(rows-2)*(col-2)+row-1-offsetCol(row,col);
            if mask(row+1,col)
                % Only forward in Y
                Gy_idx_val(Gy_count,:)=[row_idx,col_idx,-1]; Gy_count=Gy_count+1;
                col_idx=(rows-2)*(col-2)+row-offsetCol(row+1,col);
                Gy_idx_val(Gy_count,:)=[row_idx,col_idx,1]; Gy_count=Gy_count+1;
                jidx_v(jidx_v_count)=col_idx; jidx_v_count=jidx_v_count+1;
                jidx_f(jidx_f_count)=row_idx; jidx_f_count=jidx_f_count+1;
            elseif mask(row-1,col)
                Gy_idx_val(Gy_count,:)=[row_idx,col_idx,1]; Gy_count=Gy_count+1;
                col_idx=(rows-2)*(col-2)+row-2-offsetCol(row-1,col);
                Gy_idx_val(Gy_count,:)=[row_idx,col_idx,-1]; Gy_count=Gy_count+1;
                jidx_v(jidx_v_count)=col_idx; jidx_v_count=jidx_v_count+1;
                jidx_f(jidx_f_count)=row_idx; jidx_f_count=jidx_f_count+1;
            end
            % Finished with a pixel
        end
    end
end


% Construct a sparse gradient matrix


Gx_idx_val=Gx_idx_val(1:Gx_count-1,:);
Gy_idx_val=Gy_idx_val(1:Gy_count-1,:);
Gx=sparse(Gx_idx_val(:,1),Gx_idx_val(:,2),Gx_idx_val(:,3),noofRowPixels,noofColPixels);
Gy=sparse(Gy_idx_val(:,1),Gy_idx_val(:,2),Gy_idx_val(:,3),noofRowPixels,noofColPixels);

jidx_f=jidx_f(1:jidx_f_count-1);
jidx_v=jidx_v(1:jidx_v_count-1);
JAC=sparse(jidx_f,jidx_v,ones(length(jidx_v),1),noofRowPixels,noofColPixels);

end