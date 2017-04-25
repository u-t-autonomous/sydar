%dymanics
A = [2.0000e+00 -2.0000e+00 ; 1.0000e+00 0.0000e+00 ];
B = [1.0000e+00 ; 1.0000e+00 ];
n = 2;
m = 1;
Q = [1.0000e+00 0.0000e+00 ; 0.0000e+00 1.0000e+00 ];
R = [1.0000e+00 ];
x0 = [5.0000e-01 ; 0.0000e+00 ];
xf = [0.0000e+00 ; 0.0000e+00 ];
%state transition cost
s = 1;
%number of states
Nq = 4;
cvx_begin sdp;
	 variable P(n,n,Nq);
	 variable r(n,1,Nq);
	 variable t(1,Nq);
	 variable tau(10,1);
	 maximize( quad_form(x0,P(:,:,1)) + 2*r(:,:,1)'*x0 + t(:,1) );
	 for i=1:Nq
		 P(:,:,i) == semidefinite(n);
	 end
	 tau(:) >= 0;
%q_3 

	 0 <= [A'*P(:,:,4) + P(:,:,4)'*A + Q, P(:,:,4)*B,  A'*r(:,:,4); ...
	       B'*P(:,:,4)',                  R,           B'*r(:,:,4); ...
	       r(:,:,4)'*A,                   r(:,:,4)'*B, 0  ];
%q_2 

	 0 <= [A'*P(:,:,3) + P(:,:,3)'*A + Q, P(:,:,3)*B,  A'*r(:,:,3); ...
	       B'*P(:,:,3)',                  R,           B'*r(:,:,3); ...
	       r(:,:,3)'*A,                   r(:,:,3)'*B, 0  ]...
		- tau(1)*[zeros(n), zeros(n,1), 0.5*[1.0000e+00 0.0000e+00 ]' ; zeros(1,n), 0, 0; 0.5*[1.0000e+00 0.0000e+00 ], 0, -1*-1];
%q_1 

	 0 <= [A'*P(:,:,2) + P(:,:,2)'*A + Q, P(:,:,2)*B,  A'*r(:,:,2); ...
	       B'*P(:,:,2)',                  R,           B'*r(:,:,2); ...
	       r(:,:,2)'*A,                   r(:,:,2)'*B, 0  ]...
		- tau(2)*[zeros(n), zeros(n,1), 0.5*[1.0000e+00 0.0000e+00 ]' ; zeros(1,n), 0, 0; 0.5*[1.0000e+00 0.0000e+00 ], 0, -1*1];
%q_0 

	 0 <= [A'*P(:,:,1) + P(:,:,1)'*A + Q, P(:,:,1)*B,  A'*r(:,:,1); ...
	       B'*P(:,:,1)',                  R,           B'*r(:,:,1); ...
	       r(:,:,1)'*A,                   r(:,:,1)'*B, 0  ];

%('q_1', 'q_3') 
	 0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
	       r(:,:,4)',            0,            0; ...
	       zeros(1,n),             0, t(:,4) + s] ...
	     -[P(:,:,2),    r(:,:,2),   zeros(n,1); ...
	       r(:,:,2)',            0,            0; ...
	       zeros(1,n),    0, t(:,2) + s] ...
		- tau(3)*[zeros(n), zeros(n,m), 0.5*[1.0000e+00 ; 0.0000e+00 ] ; zeros(m,n), zeros(m), 0; 0.5*[1.0000e+00 ; 0.0000e+00 ]', zeros(1,m), -1*1]...
		- tau(4)*[zeros(n), zeros(n,1), 0.5*[1.0000e+00 0.0000e+00 ]' ; zeros(1,n), 0, 0; 0.5*[1.0000e+00 0.0000e+00 ], 0, -1*-1];

%('q_2', 'q_3') 
	 0 <= [P(:,:,4),    r(:,:,4),   zeros(n,1); ...
	       r(:,:,4)',            0,            0; ...
	       zeros(1,n),             0, t(:,4) + s] ...
	     -[P(:,:,3),    r(:,:,3),   zeros(n,1); ...
	       r(:,:,3)',            0,            0; ...
	       zeros(1,n),    0, t(:,3) + s] ...
		- tau(5)*[zeros(n), zeros(n,m), 0.5*[1.0000e+00 ; 0.0000e+00 ] ; zeros(m,n), zeros(m), 0; 0.5*[1.0000e+00 ; 0.0000e+00 ]', zeros(1,m), -1*1]...
		- tau(6)*[zeros(n), zeros(n,1), 0.5*[1.0000e+00 0.0000e+00 ]' ; zeros(1,n), 0, 0; 0.5*[1.0000e+00 0.0000e+00 ], 0, -1*-1];

%('q_0', 'q_2') 
	 0 <= [P(:,:,3),    r(:,:,3),   zeros(n,1); ...
	       r(:,:,3)',            0,            0; ...
	       zeros(1,n),             0, t(:,3) + s] ...
	     -[P(:,:,1),    r(:,:,1),   zeros(n,1); ...
	       r(:,:,1)',            0,            0; ...
	       zeros(1,n),    0, t(:,1) + s] ...
		- tau(7)*[zeros(n), zeros(n,1), 0.5*[1.0000e+00 0.0000e+00 ]' ; zeros(1,n), 0, 0; 0.5*[1.0000e+00 0.0000e+00 ], 0, -1*-1];

%('q_0', 'q_1') 
	 0 <= [P(:,:,2),    r(:,:,2),   zeros(n,1); ...
	       r(:,:,2)',            0,            0; ...
	       zeros(1,n),             0, t(:,2) + s] ...
	     -[P(:,:,1),    r(:,:,1),   zeros(n,1); ...
	       r(:,:,1)',            0,            0; ...
	       zeros(1,n),    0, t(:,1) + s] ...
		- tau(8)*[zeros(n), zeros(n,1), 0.5*[1.0000e+00 0.0000e+00 ]' ; zeros(1,n), 0, 0; 0.5*[1.0000e+00 0.0000e+00 ], 0, -1*1];
% final 
	quad_form(xf,P(:,:,4)) + 2*r(:,:,4)'*xf + t(:,4) == 0;
cvx_end;