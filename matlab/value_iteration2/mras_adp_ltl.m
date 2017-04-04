function [muVec] = mras_adp_ltl(sys)
% approximate optimal contro
%% Initialization
maxiter= 500;
n = zeros(maxiter, 1); % the number of samples to be collected at each iteration
n(1)= 1000;
epsilon = 0.01; % improvement parameter.
lambda= 0.2;
a=0.1;
nominalSample = 1000;
%% Initialization

muVec =cell(maxiter,1);
nvar = sys.nbasis*sys.nQ; % the number of basis
muVec{1} = zeros(1,nvar);
sigmaVec = cell(maxiter,1);
sigmaVec{1} = eye(nvar);
kappa=1000; % a random value to start with.
rho_his = zeros(maxiter,1);
rho = 0.1;
optJ_his = zeros(maxiter,1);
%%Todo: Modify the following before running the algorithm
% % for nonquadratic cost
% cvar = eye(nvar);
% bvar = [0;0;0;0;-1;-1];
%% for dubins car, NONE:

goodW = [];
goodJ = [];
update=1;
samplesize_his=zeros(maxiter,1);
for k=1:maxiter
    rho_his(k) = rho;
    [W] = mvnrnd( muVec{k}, sigmaVec{k}, n(k)); % sampling.
    % W= filter_unstable(sys, W)
    % normr(W);
    %[W, ~, ~, ~]  = rmvnrnd(mu,sigma,n(k),cvar,bvar); % truncated multi-variant normal, Ax <=b
    % W  = [W; zeros(1,sys.nbasis*2)];
    [rows,~] = size(W);
    n(k) = rows;    % Update the n(k)
    J = zeros(n(k),1); %     evaluate the costs J
    tic;
    figure
    for i =1:n(k)
        J(i) = get_lincost_value2(sys, W(i,:));
    end
    toc;
    [orderedW,orderedJ] =  resort(W, J); % descend order
    optJ_his(k) = orderedJ(end);
    
    kappa = orderedJ(ceil((1-rho)*n(k)));
    if k == 1;
        samplesize_his(k) = n(k);
        gamma = kappa;
        n(k+1) = nominalSample;
        %update parameter
        update=1;
    else
        samplesize_his(k) =  samplesize_his(k-1)+ n(k);
        if kappa <= gamma- epsilon % if the current best is improved by epsilon
            gamma = kappa; % update gamma
            %n(k+1) = n(k); % maintain the same number of samples
            n(k+1) = nominalSample;
            orderedW = orderedW(orderedJ <= gamma,:); % add new elite samples
            orderedJ = orderedJ(orderedJ <= gamma);
            update=1;
        else    % To find the largest rho' such that kappa' < gamma-epsilon
            if any(orderedJ <= gamma - epsilon) && size(orderedJ(orderedJ <=gamma-epsilon),1)>1 % if any value is smaller than gamma -epsilon.
                indices = find(orderedJ <= gamma-epsilon);
                kappaprime = orderedJ(indices(1)); % the smallest indices that is small than gamma-epsilon
                rhoprime= indices(1)/n(k);
                disp('update rhoprime');
                disp(rhoprime);
                kappa = kappaprime;
                gamma = kappaprime;
                
                rho = rhoprime;
                n(k+1) = nominalSample;
                orderedW = orderedW(orderedJ <= gamma,:); % add new elite samples
                orderedJ = orderedJ(orderedJ <= gamma);
                update=1;
            else % rho', does not improve
                n(k+1) = ceil((1+a)*n(k)); % increase the number of samples
                %                 disp('increase the size of samples');
                %                 disp(n(k));
                %                 disp(n(k+1));
                %                 if n(k+1)>=500
                %                     disp('too large sample set, assume converge');
                %                     muVec{k+1} = muVec{k};
                %                 else
                muVec{k+1} = muVec{k};
                sigmaVec{k+1} = sigmaVec{k};
                % no update in this round, but sample more data.
                update=0;
            end
        end
    end
    save('history.mat');
    
    if update
        % for Dubins car : NEED TO HANDLE THE NAN error.
        [tempmu, tempsigma, fail] = param_update(orderedJ, orderedW, muVec{k}, sigmaVec{k},gamma,k,sys.nvar*sys.nQ);
        % everything else
        %         temp1 = isnan(tempsigma);
        %         temp2 = isinf(tempsigma);
        %         if sum(temp1(:))>=1 || sum(temp2(:))>=1
        if fail == 1
            disp('return');
            return;
        end
        muVec{k+1}=  lambda * muVec{k} + (1-lambda)*tempmu;
        sigma0 =   lambda * sigmaVec{k} + (1-lambda)*tempsigma;
        sigmaVec{k+1} = nearestSPD(sigma0);
    end
    if k>5 && norm(muVec{k+1}-muVec{k}) <= 0.01 && norm(sigmaVec{k+1}- sigmaVec{k})<= 0.01 && norm(sigmaVec{k+1})<=0.5% norm(muVec{k+1} - goodW(end,:)) <=0.01
        disp('converge...');
        y = muVec{k+1};
        break;
    else
        disp(orderedJ(end));
    end
    
    %     %% generate the trajectory under the current best policy
    %     [T, Z] = ode45(@(t,z)ode_dubinscar_three_obs_rbk(t,z,sys, muVec{k+1}, 0), [0,sys.tf], z0, opts);
    %     plot(Z(:,1),Z(:,2),'color',[1,1,1]-k*0.05);
    %     hold on
    
end
% y=muVec{k+1};
%
% muMat = cell2mat(muVec(2:k+1,1));
% sigmaMat  = cell2mat(sigmaVec(2:k+1,1));

end
