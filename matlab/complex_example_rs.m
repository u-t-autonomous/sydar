% 2-dimensional dynamics
clc
%clear all
%close all
A = [2, -2; 1, 0];
%A = [-2, 2; -1, 0];
B = [1; 1];
%Q = eye(2);
Q = 0*eye(2);
R = eye(1);
n = size(A,1);
m = size(B,2);

% state transition cost
s = ones (4,4); % modified transition cost.
s(logical(eye(size(s)))) = 0;
% g = 0; % if include the cost, the trajectory is shorter. Why?
% s(1,2)= - g;
% s(2,1)= g; 
% s(1,3)= - g;
% s(3,1) = g;
% s(2,4) = -g;
% s(4,2)= g;
% s(3,4) = -g;
% s(4,3) = g;

% regions
a = 1;
%x0 = [0.5; -0.5];
x0 = [-0.5; -0.5];
%x0 = [-0.5; 0];
xf = [0; 0];

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
    
    variable tau(16,1);  % S-procedure multipliers
    
    maximize( quad_form(x0,P(:,:,1)) + 2*r(:,:,1)'*x0 + t(:,1) );
    
    % slack vars
    for i=1:Nq
        %P(:,:,i) == semidefinite(n);
        [P(:,:,i), r(:,:,i); r(:,:,i)', t(i)] == semidefinite(n+1);
    end
    tau(:) >= 0;
    %r(:) == 0;
    
    % state 1
    0 <= [A'*P(:,:,1) + P(:,:,1)'*A + Q, P(:,:,1)*B,  A'*r(:,:,1); ...
          B'*P(:,:,1)',                  R,           B'*r(:,:,1); ...
          r(:,:,1)'*A,                   r(:,:,1)'*B, 0  ];
    
    % state 2 with A
    0 <= [A'*P(:,:,2) + P(:,:,2)'*A + Q, P(:,:,2)*B,  A'*r(:,:,2); ...
          B'*P(:,:,2)',                  R,           B'*r(:,:,2); ...
          r(:,:,2)'*A,                   r(:,:,2)'*B, 0  ] ...
          - tau(1) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), -a]; % on A

    % state 3 with B
    0 <= [A'*P(:,:,3) + P(:,:,3)'*A + Q, P(:,:,3)*B,  A'*r(:,:,3); ...
          B'*P(:,:,3)',                  R,           B'*r(:,:,3); ...
          r(:,:,3)'*A,                   r(:,:,3)'*B, 0  ] ...
          - tau(2) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2]; % on B
      
    % state 2 with C
    0 <= [A'*P(:,:,2) + P(:,:,2)'*A + Q, P(:,:,2)*B,  A'*r(:,:,2); ...
          B'*P(:,:,2)',                  R,           B'*r(:,:,2); ...
          r(:,:,2)'*A,                   r(:,:,2)'*B, 0  ] ...
          - tau(3) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on C
    
    % state 4 with B
    0 <= [A'*P(:,:,4) + P(:,:,4)'*A + Q, P(:,:,4)*B,  A'*r(:,:,4); ...
          B'*P(:,:,4)',                  R,           B'*r(:,:,4); ...
          r(:,:,4)'*A,                   r(:,:,4)'*B, 0  ] ...
          - tau(4) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2]; % on B
     % state 4 with A
        0 <= [A'*P(:,:,4) + P(:,:,4)'*A + Q, P(:,:,4)*B,  A'*r(:,:,4); ...
          B'*P(:,:,4)',                  R,           B'*r(:,:,4); ...
          r(:,:,4)'*A,                   r(:,:,4)'*B, 0  ] ...
          - tau(5) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), -a]; % on A 
      
     % state 4 with C
        0 <= [A'*P(:,:,4) + P(:,:,4)'*A + Q, P(:,:,4)*B,  A'*r(:,:,4); ...
          B'*P(:,:,4)',                  R,           B'*r(:,:,4); ...
          r(:,:,4)'*A,                   r(:,:,4)'*B, 0  ] ...
          - tau(6) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on C
      
    % transition 1 -> 2 on A
    0 <= [P(:,:,2),    r(:,:,2),   zeros(n,1); ...
          r(:,:,2)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,2) + s(1,2)] ...
       - [P(:,:,1),    r(:,:,1),   zeros(n,1); ...
          r(:,:,1)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,1)] ...
       - tau(7) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), -a]; % on A
   
   % transition 1 -> 2 on C
   
    0 <= [P(:,:,2),    r(:,:,2),   zeros(n,1); ...
          r(:,:,2)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,2) + s(1,2)] ...
       - [P(:,:,1),    r(:,:,1),   zeros(n,1); ...
          r(:,:,1)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,1)] ...
       - tau(8) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on C
   
    % transition 1 -> 3 on B
    0 <= [P(:,:,3),    r(:,:,3),   zeros(n,1); ...
          r(:,:,3)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,3) + s(1,3)] ...
       - [P(:,:,1),    r(:,:,1),   zeros(n,1); ...
          r(:,:,1)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,1)] ...
       - tau(9) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2]; % on B
   

    
    % transition 2 -> 4 on B
    0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
          r(:,:,4)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,4) + s(2,4)] ...
       - [P(:,:,2),    r(:,:,2),   zeros(n,1); ...
          r(:,:,2)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,2)] ...
       - tau(10) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2]; % on B
    
      %% Jie: The transition from 3 to 4 with either A or C seems problemmatic. Is the usage of S-precedure correct here? Or should we design V(x,q,\sigma) for different sigma

    % transition 3 -> 4 on C
    0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
          r(:,:,4)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,4) + s(3,4)] ...
       - [P(:,:,3),    r(:,:,3),   zeros(n,1); ...
          r(:,:,3)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,3)] ...
       - tau(11)  * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on C

       % transition 3 -> 4 on A
    0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
          r(:,:,4)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,4) + s(3,4)] ...
       - [P(:,:,3),    r(:,:,3),   zeros(n,1); ...
          r(:,:,3)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,3)] ...
       - tau(12)  * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), -a]; % on A
   
   %%
     % transition 1 -> 4 on A and B
    0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
          r(:,:,4)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,4) + s(1,4)] ...
       - [P(:,:,1),    r(:,:,1),   zeros(n,1); ...
          r(:,:,1)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,1)] ...
       - tau(13) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), -a] ...% on A
       - tau(14) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2]; % on B
       
     % transition 1 -> 4 on C and B
    0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
          r(:,:,4)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,4) + s(1,4)] ...
       - [P(:,:,1),    r(:,:,1),   zeros(n,1); ...
          r(:,:,1)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,1)] ...
       - tau(15) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2] ...% on B
       - tau(16) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on C


   
    % final condition
    quad_form(xf,P(:,:,4)) + 2*r(:,:,4)'*xf + t(:,4) == 0;
cvx_end;


%
% Simulate a trajectory

% value function
Vq = @(x,q) quad_form(x,P(:,:,q)) + 2*r(:,:,q)'*x + t(:,q);
uq = @(x,q) -inv(R) * [B'*P(:,:,q), B'*r(:,:,q)] * [x; 1];

% region and guard functions
epsilon = 1e-2;
inA = @(x) (x(1) < -a-epsilon);
inB = @(x) (-a + epsilon < x(1)) & (x(1) < a - epsilon);
inC = @(x) (x(1) > a + epsilon);
inAB = @(x) (abs(x(1)+a) < epsilon);
inBC = @(x) (abs(x(1)-a) < epsilon);

dt    = 1e-3;
tspan = [0:dt:10];
N = length(tspan);
q_hist = zeros(1,N);
x_hist = zeros(n,N);

% initial conditions
q_hist(1) = 1;
x_hist(:,1) = x0;

for i=1:N-1
    % where am I now
    xc = x_hist(:,i);
    qc = q_hist(i);
    
    % determine optimal input and landing spot
    u = uq(xc,qc);
    xn = xc + (A*xc + B*u)*dt;
    
    % determine available next modes
    qn = [];
    %fprintf('%d, %d, %d', inA(xc), inB(xc), inC(xc));
    % revised transition function
    if qc == 1 && (inA(xc) || inC(xc))
        disp('in A or C');
        qn = [qn, 2];
    elseif qc == 1 && inB(xc) && ~inA(xc) && ~inC(xc) 
        disp('in B');
        qn = [qn, 3];
    elseif qc == 1 && (inB(xc) && inA(xc))
        disp('in A and B');
        qn = [qn,4];
    elseif qc == 1 && (inB(xc) && inC(xc))
        disp('in C and B');
        qn = [qn,4];
    elseif qc == 1 && ~inB(xc) && ~inA(xc) && ~inC(xc)
       % disp('not in B and not A and not C')
        qn = [qn, 1];
    elseif qc == 2 && ~inB(xc) && (inA(xc) || inC(xc)) 
        disp('in A or C')
        qn = [qn, 2];
    elseif qc == 2 && inB(xc) 
        disp('in B')
        qn = [qn, 4];
    elseif qc == 2 && ~inB(xc) && ~inA(xc) && ~inC(xc)
       % disp('not in B and not A and not C')
        qn = [qn, 2];
    elseif qc == 3 && inB(xc) && ~inA(xc) && ~inC(xc)
       % disp('in B and not A and not C')
        qn = [qn, 3];
   elseif qc == 3 && ~inB(xc) && ~inA(xc) && ~inC(xc)
       % disp('not in B and not A and not C')
        qn = [qn, 3];
    elseif qc == 3 && (inA(xc) || inC(xc))
        disp('in A or C');
        qn = [qn, 4];
    elseif qc == 4 
       % disp('reach final')
        qn = [qn, 4];
    end
    
    assert(length(qn) <= 1);
    
    % decide whether to switch modes
%     if length(qn) == 1
%         if (Vq(xn,qn(1)) < Vq(xn,qc)) % if the value decrease, switch, otherwise, stay.
%             rr = qn(1);
%         else
%             rr = qc;
%         end
%     end
%     
    rr = qn(1);
    % save results
    x_hist(:,i+1) = xn;
    q_hist(i+1) = rr;
end

% determine value function vs time
V_hist = zeros(1,N);
for i=1:N
    V_hist(i) = Vq(x_hist(:,i), q_hist(i));
end


figure(1);
subplot(311);
plot(x_hist(1,:), x_hist(2,:));
hold on;
modechangeidx = logical([diff(q_hist) ~= 0, 0]);
plot(x_hist(1,modechangeidx), x_hist(2,modechangeidx), 'x');
hold off;
xlim([-5,5]);
ylim([-5,5]);
xlabel('x1');
ylabel('y1');

subplot(312);
plot(tspan, q_hist, 'o');
xlabel('t');
ylabel('q(t)');

subplot(313);
plot(tspan, V_hist);
xlabel('t');
ylabel('Vq(t)');

%%
% evaluate value function
q=1;
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
figure(2);
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

figure(3)
plot(x_hist(1,:), x_hist(2,:));
hold on

