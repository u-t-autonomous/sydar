% 2-dimensional dynamics
A = [2, -2; 1, 0];
%A = [-2, 2; -1, 0];
B = [1; 1];
%Q = eye(2);
Q = 0*eye(2);
R = eye(1);
n = size(A,1);
m = size(B,2);

% state transition cost
s = 1;
% regions
a = 1;
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
Nq = 5;

cvx_begin sdp;
    variable P(n,n,Nq);
    variable r(n,1,Nq);
    variable t(1,Nq);
    
    variable tau(15,1);  % S-procedure multipliers
    
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
    
    % state 2
    0 <= [A'*P(:,:,2) + P(:,:,2)'*A + Q, P(:,:,2)*B,  A'*r(:,:,2); ...
          B'*P(:,:,2)',                  R,           B'*r(:,:,2); ...
          r(:,:,2)'*A,                   r(:,:,2)'*B, 0  ] ...
          - tau(1) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), -a]; % on A

    % state 3
    0 <= [A'*P(:,:,3) + P(:,:,3)'*A + Q, P(:,:,3)*B,  A'*r(:,:,3); ...
          B'*P(:,:,3)',                  R,           B'*r(:,:,3); ...
          r(:,:,3)'*A,                   r(:,:,3)'*B, 0  ] ...
          - tau(2) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2]; % on B
      
    % state 4
    0 <= [A'*P(:,:,4) + P(:,:,4)'*A + Q, P(:,:,4)*B,  A'*r(:,:,4); ...
          B'*P(:,:,4)',                  R,           B'*r(:,:,4); ...
          r(:,:,4)'*A,                   r(:,:,4)'*B, 0  ] ...
          - tau(3) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on C
    
    % state 5
    0 <= [A'*P(:,:,5) + P(:,:,5)'*A + Q, P(:,:,5)*B,  A'*r(:,:,5); ...
          B'*P(:,:,5)',                  R,           B'*r(:,:,5); ...
          r(:,:,5)'*A,                   r(:,:,5)'*B, 0  ] ...
          - tau(4) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2]; % on B
      
    
      
    % transition 1 -> 2
    0 <= [P(:,:,2),    r(:,:,2),   zeros(n,1); ...
          r(:,:,2)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,2) + s] ...
       - [P(:,:,1),    r(:,:,1),   zeros(n,1); ...
          r(:,:,1)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,1)] ...
       - tau(5) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), -a]; % on A
   
    % transition 1 -> 3
    0 <= [P(:,:,3),    r(:,:,3),   zeros(n,1); ...
          r(:,:,3)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,3) + s] ...
       - [P(:,:,1),    r(:,:,1),   zeros(n,1); ...
          r(:,:,1)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,1)] ...
       - tau(6) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2]; % on B
   
    % transition 1 -> 4
    0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
          r(:,:,4)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,4) + s] ...
       - [P(:,:,1),    r(:,:,1),   zeros(n,1); ...
          r(:,:,1)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,1)] ...
       - tau(7) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a]; % on C
    
    % transition 2 -> 5
    0 <= [P(:,:,5),    r(:,:,5),   zeros(n,1); ...
          r(:,:,5)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,5) + s] ...
       - [P(:,:,2),    r(:,:,2),   zeros(n,1); ...
          r(:,:,2)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,2)] ...
       - tau(8) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), -a] ...
       - tau(9) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2]; % on A & B
    
    % transition 3 -> 2
    0 <= [P(:,:,2),    r(:,:,2),   zeros(n,1); ...
          r(:,:,2)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,2) + s] ...
       - [P(:,:,3),    r(:,:,3),   zeros(n,1); ...
          r(:,:,3)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,3)] ...
       - tau(10) * [zeros(n), zeros(n,m), [-0.5;0] ; zeros(m,n), zeros(m), 0; [-0.5,0], zeros(1,m), -a] ...
       - tau(11) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2]; % on A & B
    
    % transition 3 -> 4
    0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
          r(:,:,4)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,4) + s] ...
       - [P(:,:,3),    r(:,:,3),   zeros(n,1); ...
          r(:,:,3)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,3)] ...
       - tau(12) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2] ...
       - tau(13) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a];  % on B & C
    
    % transition 4 -> 5
    0 <= [P(:,:,5),    r(:,:,5),   zeros(n,1); ...
          r(:,:,5)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,5) + s] ...
       - [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
          r(:,:,4)',   zeros(m),   zeros(m,1); ...
          zeros(1,n),  zeros(1,m), t(:,4)] ...
       - tau(14) * [ [-1,0;0,0], zeros(n,m), zeros(n,1); zeros(m,n), zeros(m), 0; zeros(1,n), zeros(1,m), a^2] ...
       - tau(15) * [zeros(n), zeros(n,m), [0.5;0] ; zeros(m,n), zeros(m), 0; [0.5,0], zeros(1,m), -a];  % on B & C
   
    % final condition
    quad_form(xf,P(:,:,5)) + 2*r(:,:,5)'*xf + t(:,5) == 0;
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
    if qc == 1 && inA(xc)
        qn = [qn, 2];
    elseif qc == 1 && inB(xc)
        qn = [qn, 3];
    elseif qc == 1 && inC(xc)
        qn = [qn, 4];
    elseif qc == 2 && inAB(xc)
        disp('inAB');
        qn = [qn, 5];
    elseif qc == 3 && inAB(xc)
        disp('inAB');
        qn = [qn, 2];
    elseif qc == 3 && inBC(xc)
        disp('inBC');
        qn = [qn, 4];
    elseif qc == 4 && inBC(xc)
        disp('inBC');
        qn = [qn, 5];
    end
    
    assert(length(qn) <= 1);
    
    % decide whether to switch modes
    if length(qn) == 1
        if (Vq(xn,qn(1)) < Vq(xn,qc))
            rr = qn(1);
        else
            rr = qc;
        end
    end
    
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
xlim([-2,2]);
ylim([-2,2]);
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

%% Draw level sets of Lyapunov function applicable in all but initial mode

figure(2);
clf;
hold on;

cs = 1:10:21;

% state 2
for c=cs
    xy2 = ellip2d(P(:,:,2),r(:,:,2),t(:,2),c);
    xy2(:,~(xy2(1,:) <= -a)) = NaN;
    plot(xy2(1,:), xy2(2,:), 'LineWidth', 2, ...
                             'Color', 0.75*[1,1,1]);
end

% state 3
for c=cs
    xy3 = ellip2d(P(:,:,3),r(:,:,3),t(:,3),c);
    xy3(:,~(abs(xy3(1,:)) <= a)) = NaN;
    plot(xy3(1,:), xy3(2,:), 'LineWidth', 2, ...
                             'Color', 0.75*[1,1,1]);
end

% state 4
for c=cs
    xy4 = ellip2d(P(:,:,4),r(:,:,4),t(:,4),c);
    xy4(:,~(xy4(1,:) >= a)) = NaN;
    plot(xy4(1,:), xy4(2,:), 'LineWidth', 2, ...
                             'Color', 0.75*[1,1,1]);
end

% state 5
for c=cs
    xy5 = ellip2d(P(:,:,5),r(:,:,5),t(:,5),c);
    xy5(:,~(abs(xy5(1,:)) <= a)) = NaN;
    plot(xy5(1,:), xy5(2,:), '--', 'LineWidth', 2, ...
                             'Color', 0.75*[1,1,1]);
end

% draw trajectory again
plot(x_hist(1,:), x_hist(2,:), 'b', 'LineWidth', 2);
modechangeidx = logical([diff(q_hist) ~= 0, 0]);
plot(x_hist(1,modechangeidx), x_hist(2,modechangeidx), 'x');

hold off;

xlim([-2,2]);
ylim([-2,2]);
xlabel('x1');
ylabel('y1');
axis equal;

%% redraw Lyapunov function
figure(3);
subplot(211);
plot(tspan, q_hist, 'o', 'LineWidth', 2);
xlabel('t');
ylabel('q(t)');
title('Mode vs time');

subplot(212);
plot(tspan, V_hist, 'LineWidth', 2);
xlabel('t');
ylabel('Vq(t)');
