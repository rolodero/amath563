clc; clear; close all;

% Simulate Lorenz system
dt=0.01; T=12; t=0:dt:T;
b=8/3; %r=28;
sig=10;
rvals = [10, 28, 40];
x0 = [12.3; 13; 9];

Lorenz = @(t,x,r)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
%%
input=[]; output=[];
for m = 1:50
    for r = rvals % training trajectories
        %x0=30*(rand(3,1)-0.5);
        [t,y] = ode45(@(t,y) Lorenz(t,y,r),t,x0);
        %input=[input; y(1:end-1,:)];
        input=[input; [y(1:end-1,:), repmat(r,size(t,1)-1,1)]];
        output=[output; y(2:end,:)];
    end
    x0=30*(rand(3,1)-0.5);
end

plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(y(1,1),y(1,2),y(1,3),'ro')
plot3(y(end,1),y(end,2),y(end,3),'bo')
grid on, view(-23,18)


%% FEEDFORWARD NET
net = feedforwardnet([30 40 40 30]);
net.layers{1}.transferFcn = 'tansig';%'logsig';
net.layers{2}.transferFcn = 'radbas';%'radbas';
net.layers{3}.transferFcn = 'logsig';%'radbas';
net.layers{4}.transferFcn = 'purelin';
net = train(net,input.',output.');

%%
save('netLoFeedForward.mat','net')

%%
load('netLoFeedForward.mat');
index = 1;
for r = [17, 35]%[10, 28, 40]% [17, 35]

%r = 10;
%x0=20*(rand(3,1)-0.5);
x0 = [12.3; 13; 9];
[t,y] = ode45(@(t,y) Lorenz(t,y,r),t,x0);
subplot(1,2,index), plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1:3,1)= x0;
for jj=2:length(t)
    ynn(:,jj)=net([ynn(:,jj-1); r]);
end
plot3(ynn(1,:),ynn(2,:),ynn(3,:),':','Linewidth',[2])
title(['\rho = ', num2str(r)]);
index = index +1;
end
%% LSTM NET

layers = [
    sequenceInputLayer(4);
    lstmLayer(200,'OutputMode','sequence')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(3)
    regressionLayer;
    ];
%options = trainingOptions('sgdm');
maxEpochs = 200;
miniBatchSize = 100;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',0.1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',40, ...
    'LearnRateDropFactor',0.75, ...
    'Plots','training-progress');


rng('default') % For reproducibility
netLo = trainNetwork(input.',output.',layers,options);

%%

sigma = 10;
out = zeros(3,length(t)-1);
out(:,1) = input(1,1:3).';
for i = 2:length(t)
    out(:,i) = predict(netLo, out(:,i-1));
end

figure(2)
%x0=20*(rand(3,1)-0.5);
[t,y] = ode45(@(t,y) Lorenz(t,y,sigma),t,out(:,1));
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

plot3(out(1,:),out(2,:),out(3,:),':','Linewidth',[2])

%%
save('netLo.mat','netLo')

%% Predict % update
load('netLo.mat')
sigma = 10;
out = zeros(3,length(t)-1);
out(:,1) = input(1,1:3).';
for i = 2:length(t)
    [netLo,out(:,i)] = predictAndUpdateState(netLo,[out(:,i-1);sigma],'ExecutionEnvironment','cpu');
end

figure(2)
%x0=20*(rand(3,1)-0.5);
[t,y] = ode45(@(t,y) Lorenz(t,y,sigma),t,out(:,1));
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

plot3(out(1,:),out(2,:),out(3,:),':','Linewidth',[2])

