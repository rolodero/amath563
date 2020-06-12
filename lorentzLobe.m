clear all, close all

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];
frame=25;

for j=1:100  % training trajectories
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end-frame,:)];
    output=[output; y(1+frame:end,1)>0];
end

%% network

net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');


%% plots

x0=20*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);

ynn=[];

for jj=1+frame:length(t)
    pred=net(y(jj-frame, :)');
    ynn=[ynn, pred];
end

figure(4)
plot(t,y(:,1),'k','Linewidth',1.5)
hold on
plot(t(1+frame:end),ynn(1, :),'m--','Linewidth',1.5)
xlabel("time"); ylabel("y");




