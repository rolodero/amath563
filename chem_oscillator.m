clear all; close all; clc;
load('BZ.mat')
%%
[m,n,T]=size(BZ_tensor); % x vs y vs time data
% for j=1:T
% A=BZ_tensor(:,:,j);
% pcolor(A), shading interp, pause(0.005)
% end
%%
X = zeros(158301,1200);
for j=1:T
    X(:,j) = reshape(BZ_tensor(:,:,j),[],1);
end
%%
r = 20;
[U,Sigma,V] = svd(X,'econ'); % Step 1
Ur1 = U(:,1:r);
Sigmar1 = Sigma(1:r,1:r);
Vr1 = V(:,1:r);
%% not the right plots
figure(1)
subplot(2,2,1), plot(diag(Sigmar1)/sum(diag(Sigmar1)),'bo','Linewidth',2);
title("Regular DMD")
subplot(2,2,3), plot(Vr1(:,1:3),'Linewidth',1.5)
legend('Mode 1','Mode 2','Mode 3')
subplot(2,2,2), plot(diag(Sigmar)/sum(diag(Sigmar)),'bo','Linewidth',2);
title("Time Delay DMD")
subplot(2,2,4), plot(Vr(:,1:3),'Linewidth',1.5)
legend('Mode 1','Mode 2','Mode 3')
Sigmar1-Sigmar
%%

[Phi, Lambda, b] = DMD(X(:,1:end-1),X(:,2:end),r);

for k = 1:1200
    X_dmd(:,k) = Phi*(Lambda^(k-1))*b;
end

%%

for k=1200
    subplot(1,2,1), pcolor(BZ_tensor(:,:,k)), shading interp;
    subplot(1,2,2), pcolor(reshape(real(X_dmd(:,k)),351,451)), shading interp, pause(0.000001)
end


%% Time delay DMD
n=4;
Hn = zeros(n*158301,1200-(n-1));
for k =1:n
    Hn((158301*(k-1)+1):(158301*k),:) = X(:,k:end-(n-k));
end

%% not change
r = 20;
[Phi, Lambda, b, Sigmar, Vr] = DMD(Hn(:,1:end-1),Hn(:,2:end),r);

for k = 1:1200
    X_tdmd(:,k) = Phi*(Lambda^(k-1))*b;
end
%%

subplot(3,2,1), pcolor(BZ_tensor(:,:,1)), shading interp;
ylabel("t = 1")
title("Data")
subplot(3,2,2), pcolor(reshape(real(X_tdmd(1:158301,1)),351,451)), shading interp,
title("Time Delay DMD")
subplot(3,2,3), pcolor(BZ_tensor(:,:,600)), shading interp;
ylabel("t = 600")
subplot(3,2,4), pcolor(reshape(real(X_tdmd(1:158301,600)),351,451)), shading interp,
subplot(3,2,5), pcolor(BZ_tensor(:,:,1200)), shading interp;
ylabel("t = 1200")
subplot(3,2,6), pcolor(reshape(real(X_tdmd(1:158301,1200)),351,451)), shading interp,


%%
figure(2)
subplot(2,1,1), plot(diag(Sigmar)/sum(diag(Sigmar)),'bo','Linewidth',2);
%subplot(3,1,2), plot(Ur(:,1:3),'Linewidth',1.5)
subplot(2,1,2), plot(Vr(:,3:5),'Linewidth',1.5)
legend('Mode 1','Mode 2','Mode 3')
  
%% Functions
function [Phi, Lambda, b, Sigmar, Vr] = DMD(X,Xprime,r)
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