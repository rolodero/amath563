clear all; close all; clc
%%
% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - u_xxxx,  periodic BCs 
input = []; output = [];
inNum = 1;
for k = 1:inNum

div1 = 20*rand() + 5;
div2 = 20*rand() + 5;
    
N = 1024;
x = 32*pi*(1:N)'/N;
u = cos(x/div1).*(1+sin(x/div2)); % 16
v = fft(u);

% % % % % %
%Spatial grid and initial condition:
h = 0.025;
k = [0:N/2-1 0 -N/2+1:-1]'/16;
L = k.^2 - k.^4;
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2)); 
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)); 
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop:
uu = u; tt = 0;
tmax = 100; nmax = round(tmax/h); nplt = floor((tmax/250)/h); g = -0.5i*k;
iter = 0;
for n = 1:nmax
t = n*h;
Nv = g.*fft(real(ifft(v)).^2);
a = E2.*v + Q.*Nv;
Na = g.*fft(real(ifft(a)).^2);
b = E2.*v + Q.*Na;
Nb = g.*fft(real(ifft(b)).^2);
c = E2.*a + Q.*(2*Nb-Nv);
Nc = g.*fft(real(ifft(c)).^2);
v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; 
if mod(n,nplt)==0
        u = real(ifft(v));
uu = [uu,u]; tt = [tt,t]; end
end
input = [input, uu(:,1:end-1)];
output = [output, uu(:,2:end)];
end
%% Plot results:
surf(tt,x,uu), shading interp, colormap(hot), axis tight
xlabel("t"), ylabel("x"), zlabel("u")
% view([-90 90]), colormap(autumn); 
set(gca,'zlim',[-5 50]) 
%%
save('kuramoto_sivishinky.mat','x','tt','uu')

%%
figure(2), pcolor(x,tt,uu.'),shading interp, colormap(hot),axis off

%%

net = feedforwardnet([20 10]);
net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'radbas';
net.layers{2}.transferFcn = 'purelin';
net = train(net, input, output)

%%
Titer = size(uu,2) - 1;
out = zeros(N,Titer);
for titer = 1:Titer
out(:,titer) = net(input(:,titer));
end
surf(tt(1:Titer),x,out), shading interp, colormap(hot), axis tight
set(gca,'zlim',[-5 50]) 
xlabel("t"), ylabel("x"), zlabel("u")
%% NET 2

layers = [
    sequenceInputLayer(N);
    lstmLayer(500);
    lstmLayer(300);
    fullyConnectedLayer(N);
    regressionLayer;
    ];
%options = trainingOptions('sgdm');
maxEpochs = 100;
miniBatchSize = 250;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');
% 'InitialLearnRate',0.05, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',2, ...
%     'LearnRateDropFactor',0.2, ...
rng('default') % For reproducibility
net2 = trainNetwork(input,output,layers,options);
%% predict
Titer = size(uu,2) - 1;
out = zeros(N,Titer);
out(:,1) = input(:,1);
out(:,2) = predict(net2,out(:,1))
for titer = 3:Titer
out(:,titer) = predict(net2,out(:,titer-1));
end
surf(tt(1:Titer),x,out), shading interp, colormap(hot), axis tight
set(gca,'zlim',[-5 50]) 
xlabel("t"), ylabel("x"), zlabel("u")

%% predict & update
load('net2.mat')
Titer = size(uu,2) - 1;
out = zeros(N,Titer);
out(:,1) = input(:,1);
%net2 = predictAndUpdateState(net2,input);
[net2,out(:,2)] = predictAndUpdateState(net2,out(:,1));

for i = 2:Titer
    [net2,out(:,i)] = predictAndUpdateState(net2,out(:,i-1),'ExecutionEnvironment','cpu');
end
surf(tt(1:Titer),x,out), shading interp, colormap(hot), axis tight
set(gca,'zlim',[-5 50]) 
xlabel("t"), ylabel("x"), zlabel("u")

%%
%save('net2.mat','net2')

%% %%% NET 3 %%% Trained on div1 = 15.8329 / div2 = 12.9597
layers = [
    sequenceInputLayer(N);
    lstmLayer(700);
    fullyConnectedLayer(N);
    regressionLayer;
    ];
%options = trainingOptions('sgdm');
maxEpochs = 120;
miniBatchSize = 25;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');
% 'InitialLearnRate',0.05, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',2, ...
%     'LearnRateDropFactor',0.2, ...
rng('default') % For reproducibility
net3 = trainNetwork(input,output,layers,options);
%%
save('net3.mat','net3')
%%
load('net3.mat')
Titer = size(uu,2) - 1;
out = zeros(N,Titer);
out(:,1) = input(:,1);
%net2 = predictAndUpdateState(net2,input);
[net3,out(:,2)] = predictAndUpdateState(net3,out(:,1));
for i = 3:Titer
    [net3,out(:,i)] = predictAndUpdateState(net3,out(:,i-1),'ExecutionEnvironment','cpu');
end
surf(tt(1:Titer),x,out), shading interp, colormap(hot), axis tight
set(gca,'zlim',[-5 50]) 
xlabel("t"), ylabel("x"), zlabel("u")

%% Side by side plot

subplot(1,2,1), surf(tt,x,uu), shading interp, colormap(hot), axis tight
xlabel("t"), ylabel("x"), zlabel("u"),
title("ODE solver")
set(gca,'zlim',[-5 50])

subplot(1,2,2), surf(tt(1:Titer),x,out), shading interp, colormap(hot), axis tight
set(gca,'zlim',[-5 50])
title("Neural Network")
xlabel("t"), ylabel("x"), zlabel("u")

%% %%% NET 4 %%% train several time series

layers = [
    sequenceInputLayer(N);
    lstmLayer(600);
    fullyConnectedLayer(N);
    regressionLayer;
    ];
%options = trainingOptions('sgdm');
maxEpochs = 50;
miniBatchSize = 25;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');
% 'InitialLearnRate',0.05, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',2, ...
%     'LearnRateDropFactor',0.2, ...
rng('default') % For reproducibility
for k = 1:inNum
    net4 = trainNetwork(input(:,((k-1)*250+1):((k-1)*250+250)),output(:,((k-1)*250+1):((k-1)*250+250)),layers,options);
end
%%
save('net4.mat','net4')
%%
load('net4.mat')
Titer = 250;
out = zeros(N,Titer);
out(:,1) = input(:,1);
%net2 = predictAndUpdateState(net2,input);
[net4,out(:,2)] = predictAndUpdateState(net4,out(:,1));
for i = 3:Titer
    [net4,out(:,i)] = predictAndUpdateState(net4,out(:,i-1),'ExecutionEnvironment','cpu');
end
surf(tt(1:Titer),x,out), shading interp, colormap(hot), axis tight
set(gca,'zlim',[-5 50]) 
xlabel("t"), ylabel("x"), zlabel("u")

%% Side by side plot

subplot(1,2,1), surf(tt(1:Titer),x,input(:,1:250)), shading interp, colormap(hot), axis tight
xlabel("t"), ylabel("x"), zlabel("u"),
title("ODE solver")
set(gca,'zlim',[-5 50])

subplot(1,2,2), surf(tt(1:Titer),x,out), shading interp, colormap(hot), axis tight
set(gca,'zlim',[-5 50])
title("Neural Network")
xlabel("t"), ylabel("x"), zlabel("u")