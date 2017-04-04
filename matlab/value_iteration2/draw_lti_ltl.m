% draw result
J0 =0;
z0 = [sys.x0; J0];
%[t, z] = ode45(@dubins, tspan, z0, [], sys, w);
noise= 0;
global q
q=0;

%opts=odeset('Events',@(t,z)terminate_event(t,z, sys),'AbsTol',1e-7,'RelTol',1e-7); % example); % terminates of value is zero.
opts= odeset('AbsTol',1e-7,'RelTol',1e-7);
[T, Z] = ode45(@(t,z)ode_lti_ltl_value(t,z, sys,muVec{k},noise), [0,10*sys.tf], z0, opts);

%[T, Z] = ode45(@(t,z)ode_lti_ltl_value(t,z, sys,w,noise), [0,sys.tf], z0, opts);


set(gcf,'defaultLineLineWidth',4)
figure
plot(Z(:,1), Z(:,2),'LineWidth',4);
hold on
grid

figure
plot(T, Z(:,1),'--','LineWidth',4);
hold on
plot(T, Z(:,2), 'LineWidth',4);
grid

figure
plot(T,Z(:,3),'LineWidth',4)
grid
