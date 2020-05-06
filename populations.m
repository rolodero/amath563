clear all; close all; clc;

%% DMD
dt = 2;
t = 1845:dt:1903;
hare = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18 10 9 65]; 
lynx = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35 12 12 25];
X = [hare; lynx];
r = 2;
[Phi, Lambda, b] = DMD(X(:,1:end-1),X(:,2:end),r);
Omega = diag(log(diag(Lambda))/dt);
t_expand = 1845:dt:1903;
for k = 1:size(t_expand,2)
    X_dmd(:,k) = Phi*(Lambda^(k-1))*b;
end
figure(1)
plot(t,hare,'-o',t,lynx,'-o','LineWidth',1.5); hold on;
plot(t_expand, X_dmd(1,:),t_expand, X_dmd(2,:),'LineWidth',1.5);
legend('data hare','data lynx','model hare','model lynx')
ylim([-2,155])
ylabel('Pelts (thousands)')
xlabel('Year')
% -> nonlinear data so DMD does not work properly

%% Time Delay DMD
% Hankel matrices

H4 = [X(:,1:end-3);
      X(:,2:end-2);
      X(:,3:end-1);
      X(:,4:end)];
n = 16;
Hn = [];
for k =1:n
    Hn((2*k-1):(2*k),:) = X(:,k:end-(n-k));
end

[U, S, V] = svd(Hn,'econ');
figure(2)
subplot(2,1,1), plot(diag(S)/sum(diag(S)),'bo','Linewidth',2);
%subplot(3,1,2), plot(U(:,1:3),'Linewidth',1.5)
subplot(2,1,2), plot(V(:,1:3),'Linewidth',1.5)
legend('Mode 1','Mode 2','Mode 3')

r = 11;
[Phi, Lambda, b] = DMD(Hn(:,1:end-1),Hn(:,2:end),r);
Omega = diag(log(diag(Lambda))/dt);
t_expand = 1845:dt:1903; % 1903
X_dmd_t = [];
for k = 1:size(t_expand,2)
    X_dmd_t(:,k) = Phi*(Lambda^(k-1))*b;
end
figure(3)
plot(t,hare,':o',t,lynx,':o','LineWidth',1.5); hold on;
plot(t_expand, real(X_dmd_t(1,:)),t_expand, real(X_dmd_t(2,:)),'LineWidth',1.5);
legend('data hare','data lynx','model hare','model lynx')
ylim([-2,155])
ylabel('Pelts (thousands)')
xlabel('Year')

%% Compute derivatives
%dxdt(1:2) = (hare(2:3)-hare(1:2))./dt;
%dxdt(3:28) = (-hare(5:end)+8.*hare(4:end-1)-8.*hare(2:end-3)+hare(1:end-4))./(12*dt);
%dxdt(29:30) = (hare(29:30)-hare(28:29))./dt;
dxdt = (-hare(5:end)+8.*hare(4:end-1)-8.*hare(2:end-3)+hare(1:end-4))./(12*dt);
figure(4)
subplot(2,1,1), plot(t(3:end-2),dxdt); hold on;

%dydt(1:2) = (lynx(2:3)-lynx(1:2))./dt;
%dydt(3:28) = (-lynx(5:end)+8.*lynx(4:end-1)-8.*lynx(2:end-3)+lynx(1:end-4))./(12*dt);
%dydt(29:30) = (lynx(29:30)-lynx(28:29))./dt;
dydt = (-lynx(5:end)+8.*lynx(4:end-1)-8.*lynx(2:end-3)+lynx(1:end-4))./(12*dt);
figure(4)
subplot(2,1,2), plot(t(3:end-2),dydt); hold on;

%% Fit PDEs
% fminsearch with derivatives
x0 = X(1,3:end-2); y0 = X(2,3:end-2);
adapterx = @(p) xerr(p(1),p(2),x0,y0,dxdt);
[px, xerror] = fminsearch(@(p) adapterx(p),[0.8151,0.0221])
adaptery = @(p) yerr(p(1),p(2),x0,y0,dydt);
[py, yerror] = fminsearch(@(p) adaptery(p),[0.0098,0.4907]);
b = px(1), p = px(2), r = py(1), d = py(2),

figure(4) % derivatives
subplot(2,1,1), plot(t(3:end-2),(px(1)-px(2).*y0).*x0);
subplot(2,1,2), plot(t(3:end-2),(py(1)*x0-py(2)).*y0);

modelODE = @(n) [(px(1)-px(2).*n(2)).*n(1); (py(1)*n(1)-py(2)).*n(2)];
[t_new,X_fit_diff] = ode45(@(t,n) modelODE(n),t,[20,32]);
figure(3)
plot(t, real(X_fit_diff(:,1)),t, real(X_fit_diff(:,2)),'LineWidth',1.5);
%%
% fminsearch withOUT derivatives
% X(1,:) = x // X(2,:) = y
adapter = @(p) err_ode45(p(1),p(2),p(3),p(4),t(2:end),X(:,2:end));
%[0.80,0.024,0.028,0.55] 
[p_vals, error, exitflag] = fminsearch(@(p) adapter(p),[0.81,0.022,0.01,0.47],optimset('MaxFunEvals',1000,'MaxIter',1000))
b = p_vals(1); p = p_vals(2); r = p_vals(3); d = p_vals(4);
fitODE = @(n) [(b-p.*n(2)).*n(1); (r.*n(1)-d).*n(2)];
[t_fit,X_fit] = ode45(@(t,n) fitODE(n),t(2:end),[20,50]);
plot(t,hare,':o',t,lynx,':o','LineWidth',1.5); hold on; % delete later
plot(t(2:end), real(X_fit(:,1)),t(2:end), real(X_fit(:,2)),'LineWidth',1.5);
legend('data hare','data lynx','model hare','model lynx')
ylim([-2,155])
ylabel('Pelts (thousands)')
xlabel('Year')

%% Fit with MODEL DISCOVERY

x0 = X(1,3:end-2)';
y0 = X(2,3:end-2)';
%A = [x0 y0 x0.*y0]
%A = [x0 y0 x0.*y0 y0.^2 x0.^3 (y0.^2).*x0 y0.^3];
    %sin(0.57.*(t(3:end-2)'-1847)) cos(0.6.*(t(3:end-2)'-1843))];%sin(0.5.*t(3:end-2)')];
A = [x0 y0 x0.*y0 x0.^2 y0.^2 x0.^3 (y0.^2).*x0 (x0.^2).*y0 y0.^3 ...
    x0.^4 (x0.^3).*y0 (x0.^2).*y0.^2 (x0).*y0.^3 y0.^4];
 
% Backslash 

xi=A\dxdt.'; b = xi(1); p = xi(2);
yi=A\dydt.'; d = yi(1); r = yi(2);
% % fncts = categorical({'x' 'y' 'x*y' 'y^2' 'x^3' '(y^2)x' 'y^3'});
% % fncts = reordercats(fncts,{'x' 'y' 'x*y' 'y^2' 'x^3' '(y^2)x' 'y^3'});
% % subplot(2,1,1), bar(fncts,xi,1)
% % subplot(2,1,2), bar(fncts,yi,1)

%% Lasso

xi= lasso(A,dxdt.','Lambda',0.176)%0.6 // full library 0.21 / 0.12
yi= lasso(A,dydt.','Lambda',0.12)%0.3
% 0.175 0.12 ohne thres
% % [Xi, FitInfo] = lasso(A,dxdt.','CV',10);
% % xi = Xi(:,FitInfo.IndexMinMSE);
% % [Yi, FitInfo] = lasso(A,dydt.','CV',10);
% % yi = Yi(:,FitInfo.IndexMinMSE);

fncts = categorical({'x' 'y' 'x*y' 'x^2' 'y^2' 'x^3' 'y^2x' 'x^2y' 'y^3' 'x^4' 'x^3y' 'x^2y^2' 'xy^3' 'y^4'});
fncts = reordercats(fncts,{'x' 'y' 'x*y' 'x^2' 'y^2' 'x^3' 'y^2x' 'x^2y' 'y^3' 'x^4' 'x^3y' 'x^2y^2' 'xy^3' 'y^4'});
figure(6)
subplot(2,1,1), bar(fncts,xi,1)
subplot(2,1,2), bar(fncts,yi,1)

thres = 0.00000001; % Bigger truncation gives error
smallinds = (abs(xi)<thres); % Find small coefficients 
xi(smallinds)=0;
smallinds = (abs(yi)<thres); % Find small coefficients 
yi(smallinds)=0;
xi,yi %% shows 15 terms
% modelODE = @(xy) [xi(1).*xy(1) +  xi(2).*xy(2) + xi(3).*xy(1).*xy(2);...
%                   yi(1).*xy(1) +  yi(2).*xy(2) + yi(3).*xy(1).*xy(2)];
modelODE = @(xy) ([xy(1) xy(2) xy(1).*xy(2) xy(1).^2 xy(2).^2 xy(1).^3 ... 
    (xy(2).^2).*xy(1) (xy(1).^2).*xy(2) xy(2).^3 xy(1).^4 (xy(1).^3).*xy(2) ...
    (xy(1).^2).*xy(2).^2 (xy(1)).*xy(2).^3 xy(2).^4]*[xi, yi])';
[t_fit,X_model_ode] = ode45(@(t,xy) modelODE(xy),t,[140,110]);
X_model = X_model_ode + [300*ones(30,1),-50*ones(30,1)];

h=0.01; % step's size
t_ode = 1845:h:1903;
N=size(t_ode,2); % number of steps
w1(1)=20;
w2(1)=32;
for n=1:N % Forward Euler for cases in which ode45 takes too long
% w1(n+1)= w1(n) + h.*(xi(1).*w1(n) +  xi(2).*w2(n) + xi(3).*w1(n).^2 + ...
%          xi(4).*w1(n).*w2(n) + xi(5).*w2(n).^2 + xi(9).*sin(0.57.*(t_ode(n)'-1847)) + ...
%          xi(10)*cos(0.6.*(t_ode(n))'-1843));
% w2(n+1)= w2(n) + h.*(yi(1).*w1(n) +  yi(2).*w2(n) + yi(3).*w1(n).^2 + ...
%          yi(4).*w1(n).*w2(n) + xi(5).*w2(n).^2 + yi(9).*sin(0.57.*(t_ode(n)'-1847)) + ...
%          yi(10)*cos(0.6.*(t_ode(n)'-1843)));
w1(n+1)= w1(n) + h.*(xi(1).*w1(n) +  xi(2).*w2(n) + ...
         + xi(3).*w1(n).*w2(n) );
w2(n+1)= w2(n) + h.*(yi(1).*w1(n) +  yi(2).*w2(n) + ...
         + yi(3).*w1(n).*w2(n) );
end
figure(66)
plot(t,hare,':o',t,lynx,':o','LineWidth',1.5); hold on; % delete later
%plot(t_ode, w1(1:end-1),t_ode, w2(1:end-1),'LineWidth',1.5);
plot(t, real(X_model(:,1)),t, real(X_model(:,2)),'LineWidth',1.5);
legend('data hare','data lynx','model hare','model lynx')
ylim([-100,155])
ylabel('Pelts (thousands)')
xlabel('Year')

%% KL Divergence

f = reshape(X,1,[]);
val = -80:3:450;
KL_dmd = KL(f,reshape(real(X_dmd),1,[]),val)
KL_dmd_t = KL(f,reshape(real(X_dmd_t(1:2,:)),1,[]),val)
KL_fit = KL(reshape(X(:,2:end),1,[]),reshape(real(X_fit'),1,[]),val)
KL_model = KL(f,reshape(real(X_model),1,[]),val)

%% AIC BIC

% Compute average error over both time series
model = reshape(real(X_dmd_t(1:2,:)),1,[]);
E_avg = mean(abs(f-model));
aic_dmd_t = 60*log(sum((f-reshape(real(X_dmd_t(1:2,:)),1,[])).^2)/60)+2*11
aic_fit = 60*log(sum((reshape(X(:,2:end),1,[])-reshape(real(X_fit'),1,[])).^2)/60)+2*4
aic_model = 60*log(sum((f-reshape(real(X_model),1,[])).^2)/60)+2*15

bic_dmd_t = 60*log(sum((f-reshape(real(X_dmd_t(1:2,:)),1,[])).^2)/60)+log(60)*11
bic_fit = 60*log(sum((reshape(X(:,2:end),1,[])-reshape(real(X_fit'),1,[])).^2)/60)+log(60)*4
bic_model = 60*log(sum((f-reshape(real(X_model),1,[])).^2)/60)+log(60)*15

%% Functions
function [Phi, Lambda, b] = DMD(X,Xprime,r)
[U,Sigma,V] = svd(X,'econ'); % Step 1
Ur = U(:,1:r);
Sigmar = Sigma(1:r,1:r);
Vr = V(:,1:r);
Atilde = Ur'*Xprime*Vr/Sigmar; % Step 2
[W,Lambda] = eig(Atilde); % Step 3
Phi = Xprime*(Vr/Sigmar)*W; % Step 4
alpha1 = Sigmar*Vr(1,:)';
b = (W*Lambda)\alpha1;
end

function err = xerr(b,p,x0,y0,dxdt)
    err = sum((dxdt-(b-p.*y0).*x0).^2);
end
function err = yerr(r,d,x0,y0,dydt)
    err = sum((dydt-(r*x0-d).*y0).^2);
end
function err = err_ode45(b,p,r,d,tspan,X)
    modelODE = @(n) [(b-p.*n(2)).*n(1); (r.*n(1)-d).*n(2)];
    [t,X_fit] = ode45(@(t,n) modelODE(n),tspan,[20,50]);
    err = norm(X'-X_fit,2);
end
function I = KL(f,g,val)
    figure
    hist(g,val)
    f=hist(f,val)+0.001; % generate PDFs
    g=hist(g,val)+0.001;
    f=f/trapz(val,f); % normalize data
    g=g/trapz(val,g);
    Int=f.*log(f./g);
    I=trapz(val,Int);
end