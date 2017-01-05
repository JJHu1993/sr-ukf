function [x,S]=ukf(fstate,x,S,hmeas,z,Qs,Rs)
% SR-UKF   Square Root Unscented Kalman Filter for nonlinear dynamic systems
% [x, S] = ukf(f,x,S,h,z,Qs,Rs) returns state estimate, x and state covariance, P 
% for nonlinear dynamic system (for simplicity, noises are assumed as additive):
%           x_k+1 = f(x_k) + w_k
%           z_k   = h(x_k) + v_k
% where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
%       v ~ N(0,R) meaning v is gaussian noise with covariance R
% Inputs:   f: function handle for f(x)
%           x: "a priori" state estimate
%           S: "a priori" estimated the square root of state covariance
%           h: fanction handle for h(x)
%           z: current measurement
%           Qs: process noise standard deviation
%           Rs: measurement noise standard deviation
% Output:   x: "a posteriori" state estimate
%           S: "a posteriori" square root of state covariance
%
% Example:
%{
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
%}
% Reference: R. van der Merwe and E. Wan. 
% The Square-Root Unscented Kalman Filter for State and Parameter-Estimation, 2001
%
% By Zhe Hu at City University of Hong Kong, 05/01/2017
%
L=numel(x);                                 %numer of states
m=numel(z);                                 %numer of measurements
alpha=1e-3;                                 %default, tunable
ki=0;                                       %default, tunable
beta=2;                                     %default, tunable
lambda=alpha^2*(L+ki)-L;                    %scaling factor
c=L+lambda;                                 %scaling factor
Wm=[lambda/c 0.5/c+zeros(1,2*L)];           %weights for means
Wc=Wm;
Wc(1)=Wc(1)+(1-alpha^2+beta);               %weights for covariance
c=sqrt(c);
X=sigmas(x,S,c);                            %sigma points around x
[x1,X1,S1,X2]=ut(fstate,X,Wm,Wc,L,Qs);          %unscented transformation of process
% X1=sigmas(x1,P1,c);                         %sigma points around x1
% X2=X1-x1(:,ones(1,size(X1,2)));             %deviation of X1
[z1,Z1,S2,Z2]=ut(hmeas,X1,Wm,Wc,m,Rs);       %unscented transformation of measurments
P12=X2*diag(Wc)*Z2';                        %transformed cross-covariance
K=P12/S2/S2';
x=x1+K*(z-z1);                              %state update
%S=cholupdate(S1,K*P12,'-');                %covariance update
U = K*S2';
for i = 1:m
    S1 = cholupdate(S1, U(:,i), '-');
end
S=S1;

function [y,Y,S,Y1]=ut(f,X,Wm,Wc,n,Rs)
%Unscented Transformation
%Input:
%        f: nonlinear map
%        X: sigma points
%       Wm: weights for mean
%       Wc: weights for covraiance
%        n: numer of outputs of f
%        Rs: additive std
%Output:
%        y: transformed mean
%        Y: transformed smapling points
%        S: transformed square root of covariance
%       Y1: transformed deviations

L=size(X,2);
y=zeros(n,1);
Y=zeros(n,L);
for k=1:L                   
    Y(:,k)=f(X(:,k));       
    y=y+Wm(k)*Y(:,k);       
end
Y1=Y-y(:,ones(1,L));
residual=Y1*diag(sqrt(abs(Wc)));
% residual=Y1*diag(sqrt(Wc));                   %It is also right(plural)
[~,S]=qr([residual(:,2:L) Rs]',0);
if Wc(1)<0
    S=cholupdate(S,residual(:,1),'-');
else
    S=cholupdate(S,residual(:,1),'+');
end
% S=cholupdate(S,residual(:,1));                %It is also right(plural)
%P=Y1*diag(Wc)*Y1'+R;          

function X=sigmas(x,S,c)
%Sigma points around reference point
%Inputs:
%       x: reference point
%       S: square root of covariance
%       c: coefficient
%Output:
%       X: Sigma points

A = c*S';
Y = x(:,ones(1,numel(x)));
X = [x Y+A Y-A]; 



