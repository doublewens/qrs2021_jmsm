function [ pXs,pXt ] = JMSM( Xs, Xt, options)

Xs = Xs';
Xt = Xt';

[dt,nt] = size(Xt);
[ds,ns] = size(Xs);

dim = min(ds,dt)-1;

Xs= Xs*diag(1./sqrt(sum(Xs.^2)));
Xt= Xt*diag(1./sqrt(sum(Xt.^2)));

% initialize Us and Ut
Us = eye(ds,dim);
Ut = eye(dt,dim);

% block data
X = blkdiag(Xs,Xt); % (ds+dt) x (ns+nt)

% MMD term
Lr11 = ones(ns,ns)/(ns*ns);
Lr22 = ones(nt,nt)/(nt*nt);
Lr12 = -ones(ns,nt)/(ns*nt);
Lr21 = Lr12';
Lr = [Lr11,Lr12;Lr21,Lr22];
Lr = Lr / norm(Lr,'fro');
Fr = X*Lr*X';
Fr = max(Fr,Fr');

mu = options.mu;
epsilon = options.ep;
T = options.T;

for k =1:T
    % compute R
    rs = 1./(2*sqrt(sum(Us.^2,2)+epsilon));
    Rs = diag(rs);
    rt = 1./(2*sqrt(sum(Ut.^2,2)+epsilon));
    Rt = diag(rt);
    R = blkdiag(Rs,Rt);
    
    AA = Fr+mu*R;
    
    [U,DD] = eig(AA);
    
    diagD = diag(DD);
    [~,sidx] = sort(diagD);
    U = U(:,sidx);
    U = U(:,1:dim);
    Us = U(1:ds,:);
    Ut = U(ds+1:end,:);
end
Z = U'*X;
pXs = Z(:,1:ns)';
pXt = Z(:,ns+1:ns+nt)';
end