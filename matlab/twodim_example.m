% 2-dimensional dynamics
A = [2, -2; 1, 0];
B = [1; 1];
Q = eye(2);
R = eye(1);
n = size(A,1);
m = size(B,2);

% state transition cost
s = 1;
% regions
a = 1;
x0 = [0.5; 0];

% LQR solution
% [K,~,~] = lqr(A,B,Q,R,[]);
% Acl = A-B*K;
% [X,Y] = meshgrid(-2:.2:2);
% UV = Acl*[X(:),Y(:)]';
% U = reshape(UV(1,:),[size(X,1),size(Y,2)]);
% V = reshape(UV(2,:),[size(X,1),size(Y,2)]);
% quiver(X,Y,U,V);


% Value function definitions
Nq = 4;

cvx_begin sdp;
    variable P(n,n,Nq);
    variable r(n,1,Nq);
    variable t(1,Nq);
    
    variable tau(8,1);
    
    maximize( quad_form(x0,P(:,:,1)) + 2*r(:,:,1)'*x0 + t(:,1) );
    
    % slack vars
    for i=1:Nq
        P(:,:,i) == semidefinite(n);
    end
    tau(:) >= 0;
    r(:) == 0;
    
    % state 1
    0 <= [A'*P(:,:,1) + P(:,:,1)'*A + Q, P(:,:,1)*B,  A'*r(:,:,1); ...
          B'*P(:,:,1)',                  R,           B'*r(:,:,1); ...
          r(:,:,1)'*A,                   r(:,:,1)'*B, 0  ];
    
    % state 2
    0 <= [A'*P(:,:,2) + P(:,:,2)'*A + Q, P(:,:,2)*B,  A'*r(:,:,2); ...
          B'*P(:,:,2)',                  R,           B'*r(:,:,2); ...
          r(:,:,2)'*A,                   r(:,:,2)'*B, 0  ] ...
          - tau(1) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), a]; % on A
      
    % state 3
    0 <= [A'*P(:,:,3) + P(:,:,3)'*A + Q, P(:,:,3)*B,  A'*r(:,:,3); ...
          B'*P(:,:,3)',                  R,           B'*r(:,:,3); ...
          r(:,:,3)'*A,                   r(:,:,3)'*B, 0  ] ...
          - tau(2) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on B
    
    % state 4
    0 <= [A'*P(:,:,4) + P(:,:,4)'*A + Q, P(:,:,4)*B,  A'*r(:,:,4); ...
          B'*P(:,:,4)',                  R,           B'*r(:,:,4); ...
          r(:,:,4)'*A,                   r(:,:,4)'*B, 0  ];
    
    % transition 1 -> 2
    0 <= [P(:,:,2),    r(:,:,2),   zeros(n,1); ...
          r(:,:,2)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,2) + s] ...
       - [P(:,:,1),    r(:,:,1),   zeros(n,1); ...
          r(:,:,1)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,1)] ...
       - tau(3) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), a]; % on A
   
   % transition 1 -> 3
   0 <= [P(:,:,3),    r(:,:,3),   zeros(n,1); ...
          r(:,:,3)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,3) + s] ...
       - [P(:,:,1),    r(:,:,1),   zeros(n,1); ...
          r(:,:,1)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,1)] ...
       - tau(4) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on B
   
   % transition 2 -> 4
   0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
          r(:,:,4)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,4) + s] ...
       - [P(:,:,2),    r(:,:,2),   zeros(n,1); ...
          r(:,:,2)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,2)] ...
       - tau(5) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), a] ... % on A
       - tau(6) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on B
   
   % transition 3 -> 4
   0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
          r(:,:,4)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,4) + s] ...
       - [P(:,:,3),    r(:,:,3),   zeros(n,1); ...
          r(:,:,3)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,3)] ...
       - tau(7) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), a] ... % on A
       - tau(8) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on B
   
   % final condition
   t(:,4)==0;
cvx_end;

%%
% evaluate value function
Vq = @(x,q) quad_form(x,P(:,:,q)) + 2*r(:,:,q)'*x + t(:,q);
V = @(x) max([Vq(x, 1), Vq(x, 2), Vq(x, 3), Vq(x, 4)]);

% plot value function
[X,Y] = meshgrid(-2:.1:2, -2:.1:2);
Z = zeros(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        Z(i,j) = V([X(i,j); Y(i,j)]);
    end
end
[DX,DY] = gradient(Z,.1,.1);
DX = -DX; DY = -DY;
figure(1);
subplot(121);
contour(X,Y,Z);
hold on;
quiver(X,Y,DX,DY);
hold off;
xlabel('x1');
ylabel('x2');

subplot(122);
surfc(X,Y,Z);
xlabel('x1');
ylabel('x2');

%%
% Plot a trajectory




%%
Z1 = zeros(size(X));
Z2 = zeros(size(X));
Z3 = zeros(size(X));
Z4 = zeros(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        Z1(i,j) = Vq([X(i,j); Y(i,j)], 1);
        Z2(i,j) = Vq([X(i,j); Y(i,j)], 2);
        Z3(i,j) = Vq([X(i,j); Y(i,j)], 3);
        Z4(i,j) = Vq([X(i,j); Y(i,j)], 4);
    end
end
%Z1(X > a) = NaN;
%Z2(X > a) = NaN;
Z3(X > a) = NaN;
Z4(X < a) = NaN;

figure(2);
mesh(X,Y,Z3);
hold on;
surf(X,Y,Z4);
hold off;
xlabel('x1');
ylabel('x2');
zlim([0,5]);


%% draw solution

KK = -inv(R)*B'*P(:,:,1);
odefun = @(t,x) (A+B*KK)*x;
[tout,xout] = ode45(odefun, [0,5], x0);
xout = xout';

figure(3);
plot(xout(1,:), xout(2,:));
xlim([-2,2]);
ylim([-2,2]);
axis equal;











