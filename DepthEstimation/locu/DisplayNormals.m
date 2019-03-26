function DisplayNormals(N,Iun,mask)

[rows,cols,~]=size(Iun);

normals=nan(rows,cols,3);
for i=1:3
    n=nan(rows,cols);
    n(mask)=N(:,i);
    normals(:,:,i)=n;
end

% m=integrability_costimg(normals,mask);
% imshow(m);title('Integrability map');
imshow((normals+1)./2);

end