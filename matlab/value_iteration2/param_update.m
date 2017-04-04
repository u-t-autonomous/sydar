function [mu,sigma, fail] = param_update(goodJ, goodW, mu, sigma, gamma, k, nvar)
% POLICYFUNC
% Input: Wk, theta_k
% Output: The parametes of the distribution
    fm1 = zeros(1,nvar);
    fm2 = 0;
    fail = 0;
    
   % disp(goodJ');
    nw = size(goodW,1);
    P=zeros(nw,1);
    temp=zeros(nw,1);
    
    % Updating Mu
 
    for i=1:nw
        P(i) = mvnpdf(goodW(i,:)',mu',sigma);
        temp(i)= get_S(goodJ(i))^k/P(i)*(goodJ(i)<=gamma);
        fm1 = fm1 + temp(i)*goodW(i,:);    
        fm2 = fm2 + temp(i);
    end
    mu = fm1./fm2;
    if fm2 == 0
        disp('fm2 is 0');
        fail = 1;
        return
    end
    fs1 = zeros(size(sigma));
    % Updating Sigma
    for i = 1:nw
        fs1 = fs1 + temp(i)*transpose(goodW(i,:) - mu)*(goodW(i,:)-mu);
    end
     sigma = fs1./fm2;
%      temp1 = isnan(sigma);
%      temp2 = isinf(sigma);
%      if sum(temp1(:))>=1 || sum(temp2(:))>=1
%          disp(sigma);
%      end
end