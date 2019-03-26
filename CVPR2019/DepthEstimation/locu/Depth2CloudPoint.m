%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Input para:
%       @ img_pts: coordinate on image planes
%       @ Z: world depth
%       @ P: camera matrix
%      optional:
%
%   Output para:
%       @ pt_3d: 3D cloud points
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pt_3d=Depth2CloudPoint(imgpts,Z,P)

N=size(imgpts,1);

h_imgpts=[imgpts ones(N,1)];

M=P(:,1:3);
p4=P(:,4);

% camera center
C=-M\p4;

% point on infinity plane
x_infinity=(M\h_imgpts')';

mu=(Z-C(3))./x_infinity(:,3);

pt_3d=(mu.*x_infinity) + C';

end