
function dz = ode_lti_ltl_value(t, z, sys, w,noise)
% z= x, q, J.
    n = size(z,1);
    x = z(1:n-1,1);
    global q
    nq = get_trans_ex(q,x);
    if nq ~= q
         q = nq; 
     % disp(nq);
    end
   % basisval = [x(1), x(2), x(1)^2, x(2)^2 , x(1)*x(2)];
   % wvec = w(q* sys.nbasis+1: q*sys.nbasis + sys.nbasis);
   % u  = - wvec(1)/2 - wvec(2)/2 - wvec(3)*x(1) - wvec(4)*x(2) - (wvec(5)*x(1))/2 - (wvec(5)*x(2))/2;

   %%% another basis function
    basisval = [1, x(1)^2, x(2)^2 , x(1)*x(2)];
    wvec = w(q* sys.nbasis+1: q*sys.nbasis + sys.nbasis);

    u = - wvec(2)*x(1) - wvec(3)*x(2) - (wvec(4)*x(1))/2 - (wvec(4)*x(2))/2;

   
    %u = min(5, max(-5,u)); % bound the input    
    if noise==1
        dx = sys.fnoise(x,u);
    else
        dx = sys.f(x,u);
    end
    dJ = sys.l(x,u);
    dz = [dx;dJ];
end
