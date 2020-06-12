clear all; close all; clc

% lambda-omega reaction-diffusion system
%  u_t = lam(A) u - ome(A) v + d1*(u_xx + u_yy) = 0
%  v_t = ome(A) u + lam(A) v + d2*(v_xx + v_yy) = 0
%
%  A^2 = u^2 + v^2 and
%  lam(A) = 1 - A^2
%  ome(A) = -beta*A^2


t=0:0.05:10;
d1=0.1; d2=0.1; beta=1.0;
L=20; n=512; N=n*n;
x2=linspace(-L/2,L/2,n+1); x=x2(1:n); y=x;
kx=(2*pi/L)*[0:(n/2-1) -n/2:-1]; ky=kx;

% INITIAL CONDITIONS

[X,Y]=meshgrid(x,y);
[KX,KY]=meshgrid(kx,ky);
K2=KX.^2+KY.^2; K22=reshape(K2,N,1);

m=1; % number of spirals

u = zeros(length(x),length(y),length(t));
v = zeros(length(x),length(y),length(t));

u(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*cos(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));
v(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*sin(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));

% REACTION-DIFFUSION
uvt=[reshape(fft2(u(:,:,1)),1,N) reshape(fft2(v(:,:,1)),1,N)].';
[t,uvsol]=ode45('reaction_diffusion_rhs',t,uvt,[],K22,d1,d2,beta,n,N);


for j=1:length(t)-1
ut=reshape((uvsol(j,1:N).'),n,n);
vt=reshape((uvsol(j,(N+1):(2*N)).'),n,n);
u(:,:,j+1)=real(ifft2(ut));
v(:,:,j+1)=real(ifft2(vt));

figure(1)
pcolor(x,y,v(:,:,j+1)); shading interp; colormap(hot); colorbar; drawnow; 
end

%save('reaction_diffusion_big.mat','t','x','y','u','v')

%%
load reaction_diffusion_big
%%
for k = 1:201
    figure(1)
    pcolor(x,y,u(:,:,k)); shading interp; colormap(hot); drawnow;
    figure(2)
    pcolor(x,y,v(:,:,k)); shading interp; colormap(hot); drawnow;
end
%%
subplot(1,2,1), pcolor(x,y,u(:,:,1)); shading interp; %colormap(hot); drawnow;
title("t = 0");
subplot(1,2,2), pcolor(x,y,u(:,:,200)); shading interp;% colormap(hot); drawnow;
title("t = 200");

%%
u_series = zeros(512*512,201);
v_series = zeros(512*512,201);
for k = 1:201
    u_series(:,k) = reshape(u(:,:,k),[],1);
    v_series(:,k) = reshape(v(:,:,k),[],1);
end
[U_u,S_u,V_u] = svd(u_series,'econ');
[U_v,S_v,V_v] = svd(v_series,'econ');
%%
plot(diag(S_u)./sum(diag(S_u)),'*','LineWidth',1.5); % 12 eigenmodes
ylim([0,0.5])
xlim([0,50])
xlabel("mode number");
ylabel("relative value")
%%
for k = 1:12
   subplot(2,6,k), pcolor(x,y,reshape(U_u(:,k),512,512)); shading interp; colormap(hot);   
end
%%
r = 12;
u_trunc = (U_u(:,1:r)'*u_series);
v_trunc = (U_v(:,1:r)'*v_series);
u_trunc1 = (V_u(1:r,:));
for k = 1:201
    figure(1)
    pcolor(x,y,u(:,:,k)); shading interp; colormap(hot); drawnow;
    figure(2)
    pcolor(x,y,reshape(U_u(:,1:r)*u_trunc(:,k),512,512)); shading interp; colormap(hot); drawnow;
end
%% NET
% input = cell(200,1);
% for k = 1:200
%     input{k} = u_trunc(:,k);
% end

input = u_trunc(:,1:end-1);
output = u_trunc(:,2:end);

layers = [
    sequenceInputLayer(r);
    lstmLayer(800);
    lstmLayer(600);
    fullyConnectedLayer(r);
    regressionLayer;
    ];
%options = trainingOptions('sgdm');
maxEpochs = 250;
miniBatchSize = 25;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');
% 'InitialLearnRate',0.1, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',4, ...
%     'LearnRateDropFactor',0.02, ...

rng('default') % For reproducibility
netRD = trainNetwork(input,output,layers,options);
%%
save('netRD.mat','netRD')

%% Predict
load('netRD.mat')
Titer = 200;
out = zeros(r,Titer);
out(:,1) = input(1:12,1);
%net2 = predictAndUpdateState(net2,input);
[netRD,out(:,2)] = predictAndUpdateState(netRD,out(:,1));
for i = 3:Titer
    [netRD,out(:,i)] = predictAndUpdateState(netRD,out(:,i-1),'ExecutionEnvironment','cpu');
end
for k = 1:201
    figure(1)
    pcolor(x,y,reshape(U_u(:,1:r)*input(1:12,k),512,512)); shading interp; colormap(hot); drawnow;
    figure(2)
    pcolor(x,y,reshape(U_u(:,1:r)*out(:,k),512,512)); shading interp; colormap(hot); drawnow;
end

%% NET 2
input = [u_trunc(:,1:end-1); v_trunc(:,1:end-1)];
output = [u_trunc(:,2:end); v_trunc(:,2:end)];
%%
layers = [
    sequenceInputLayer(2*r);
    lstmLayer(1800);
    lstmLayer(1200);
    fullyConnectedLayer(2*r);
    regressionLayer;
    ];
%options = trainingOptions('sgdm');
maxEpochs = 100;
miniBatchSize = 50;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');

% 'InitialLearnRate',0.2, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',15, ...
%     'LearnRateDropFactor',0.5, ...


rng('default') % For reproducibility
netRD2 = trainNetwork(input,output,layers,options);
%%
save('netRD2.mat','netRD2')

%% Predict
load('netRD2.mat')
Titer = 200;
out = zeros(2*r,Titer);
out(:,1) = input(:,1);
%net2 = predictAndUpdateState(net2,input);
[netRD2,out(:,2)] = predictAndUpdateState(netRD2,out(:,1));
for i = 3:Titer
    [netRD2,out(:,i)] = predictAndUpdateState(netRD2,out(:,i-1),'ExecutionEnvironment','cpu');
end
for k = 1:Titer
    figure(1)
    pcolor(x,y,reshape(U_u(:,1:r)*input(1:12,k),512,512)); shading interp; colormap(hot); drawnow;
    figure(2)
    pcolor(x,y,reshape(U_u(:,1:r)*out(1:12,k),512,512)); shading interp; colormap(hot); drawnow;
end

%% Side by side plot

input_u = U_u(:,1:r)*input(1:12,:);
subplot(1,2,1), imagesc(input_u(1:1000:end,:)); xlabel("t"); ylabel("every 1000th pixel");
title("ODE solver");

res_u = U_u(:,1:r)*out(1:12,:);
subplot(1,2,2), imagesc(res_u(1:1000:end,:)); xlabel("t"); ylabel("every 1000th pixel");
title("Neural Network");




