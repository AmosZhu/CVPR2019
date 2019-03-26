%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Compute the offset of the mask
%   Start from column
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function offm=maskoffset(mask)

[m,n]=size(mask);
offm=zeros(m,n);
maskCount=0;

% for matlab is column stack, start count from column
for j=1:n
    for i=1:m
        if mask(i,j)==0
            maskCount=maskCount+1;
        else
            offm(i,j)=maskCount;
        end
    end
end

end