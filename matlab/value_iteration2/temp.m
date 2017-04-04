syms w1 w2 w3 w4 w5 x1 x2  u 
x=[x1;x2];
%V= w1*x1 + w2*x2+ w3*x1^2 + w4*x2^2 + w5*x1*x2;  
V = w1 + w2*x1^2 + w3*x2^2+ w4*x1*x2;
obj = jacobian(V, [x1,x2])*(sys.A*x + sys.B*u) + sys.l(x,u)
solve(gradient(obj, u)==0,u)