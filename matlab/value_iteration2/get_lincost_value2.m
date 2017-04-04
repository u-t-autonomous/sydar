function totalcost = get_lincost_value2(sys, w)
% the weighted sum of costs
% COSTFUNC
% Input: State, Input Signal
% Output: Cost
% V=get_value(w,sys.basis,sys.x0, sys.q0)
beta=1; % the weight
wvec0 = w( sys.q0 * sys.nbasis+1: sys.q0*sys.nbasis + sys.nbasis);
%basisval0= [sys.x0(1), sys.x0(2), sys.x0(1)^2, sys.x0(2)^2, sys.x0(1)*sys.x0(2)];
basisval0= [1, sys.x0(1)^2, sys.x0(2)^2, sys.x0(1)*sys.x0(2)];
V = wvec0*basisval0'; % V(0) - cost given the current value function.

tspan = [0,sys.tf]; % second
J0 =0;
z0 = [sys.x0; J0];
%[t, z] = ode45(@dubins, tspan, z0, [], sys, w);
noise= 0;
global q
q=0;
opts=odeset('Events',@(t,z)terminate_event(t,z, sys),'AbsTol',1e-7,'RelTol',1e-6); % example); % terminates of value is zero.

[T, Z] = ode45(@(t,z)ode_lti_ltl_value(t,z, sys,w,noise), tspan, z0, opts);
n = size(sys.x0, 1);
m = size(z0, 1);
J = Z(:,n+1:m);
totalcost = J(end);
xvec = Z(end, 1:n)';
terminal_cost = sys.terminal(xvec(end,:)'); % the terminal cost
totalcost = totalcost + terminal_cost;
if q==3
    totalcost=totalcost+ 1000;
else
    if q==2
        totalcost= totalcost+2000;
    else if q==1
            totalcost=totalcost+3000;
        else if q==0
                totalcost=totalcost+4000;
            end
        end
    end
end

totalcost= (1-beta)*norm(totalcost- V)+ beta* totalcost;
plot(Z(:,1),Z(:,2));
hold on
end



