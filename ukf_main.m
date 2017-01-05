% Reference: R. van der Merwe and E. Wan. 
% The Square-Root Unscented Kalman Filter for State and Parameter-Estimation, 2001
%
% By Zhe Hu at City University of Hong Kong, 05/01/2017
n=3;      %number of state
q=0.1;    %std of process 
r=0.1;    %std of measurement
Qs=q*eye(n); % std matrix of process
Rs=r;        % std of measurement  
f=@(x)[x(2);x(3);0.05*x(1)*(x(2)+x(3))];  % nonlinear state equations
h=@(x)x(1);                               % measurement equation
s=[0;0;1];                                % initial state
x=s+q*randn(3,1); %initial state          % initial state with noise
S = eye(n);                               % initial square root of state covraiance
N=20;                                     % total dynamic steps
xV = zeros(n,N);          %estmate        % allocate memory
sV = zeros(n,N);          %actual
zV = zeros(1,N);
for k=1:N
  z = h(s) + r*randn;                     % measurments
  sV(:,k)= s;                             % save actual state
  zV(k)  = z;                             % save measurment
  [x, S] = ukf(f,x,S,h,z,Qs,Rs);            % ekf 
  xV(:,k) = x;                            % save estimate
  s = f(s) + q*randn(3,1);                % update process 
end
for k=1:3                                 % plot results
  subplot(3,1,k)
  plot(1:N, sV(k,:), '-', 1:N, xV(k,:), '--')
end