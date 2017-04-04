function [newW]= filter_unstable(sys, W)
qf=3; 
newW =  zeros(size(W));
n=sizeq(W,1);
temp=1;
for i=1:n
    w=W(i,:);
wvec4 = w( qf * sys.nbasis+1: qf*sys.nbasis + sys.nbasis);
P= [ wvec4(1), 0.5* wvec4(3); 0.5* wvec4(3), wvec4(2)];
Q= sys.A*P + P'*sys.A';
if all(eig(P) > eps) && all(eig(Q)<= - eps)
    newW(temp) = W(i,:);
    temp=temp+1;
end
   
newW=newW(1:temp-1,:);
end