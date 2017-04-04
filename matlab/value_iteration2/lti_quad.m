% linear system with a nonquadratic cost functional
clear all;
clc;
% xdot = A*x+B*u
A = [
    2 -2;
    1 0
    ];
B = [
    1;
    1
    ];
syms x1 x2 w1 w2;
x0 = [0.5;0.5];	% Intial State

syms symx1 symx2

 basis = [1, symx1*symx1, symx2*symx2, symx1*symx2]; % quadratic value function with a constant term
 nvar = size(basis,2);
 sys.Q = [0,1,2,3];
 sys.nQ = size(sys.Q,2);
 sys.q0 = get_trans_ex(0, x0);
 sys.nbasis = size(basis,2);
 sys.basis  = basis; 
 sys.nvar=nvar;

 sys.A = A;
sys.B = B;
Q = eye(2);
R = 1;

sys.f=@(x,u) A*x+B*u;
%sys.l=@(x,u) x'*x + u'*u + 0.5* norm(x)^4+ 0.8* norm(x)^6;
sys.l = @(x,u) x.'*Q*x + u.'*R*u;
sys.terminal = @(x,u) 100*norm(x)^2;
sys.tf=10;
sys.x0 = x0;
sys.maxcost=1000000;