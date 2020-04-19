close all; clear all; clc;
count = 60000; offset = 0;
[imgs, labels] = readMNIST('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', count, offset);
pixel = size(reshape(imgs(:,:,1),[],1),1);
B = zeros(count,10); A = zeros(count,pixel);

for c = 1:count
    B(c,labels(c)+1) = 1;
    A(c,:) = reshape(imgs(:,:,c),1,[]);
end

%% SOLVERS
 
X1=pinv(A)*B; 
X2=A\B;

% cvx
rows = numel(imgs(:,:,1));
cols = 10;
lambda1 = 0.05;
cvx_begin; % promotes sparse matrix
variable X3_1(rows,cols)
minimize(norm(A*X3_1-B,2) + lambda1*norm(X3_1,1));
cvx_end;

% REGRESSIONS
X3 = zeros(784,10); X4_uncut = zeros(785,10); X5_uncut = zeros(785,10);
for k = 1:10
    [XL3, FitInfo] = lasso(A,B(:,k),'CV',10);
    X3(:,k) = XL3(:,FitInfo.IndexMinMSE);
    X4_uncut(:,k) = robustfit(A,B(:,k)); 
    X5_uncut(:,k) = ridge(B(:,k),A,0.05,0); 
end
X4 = X4_uncut(2:785,:);
X5 = X5_uncut(2:785,:);
lassoPlot(XL3,FitInfo,'PlotType','CV')
legend('show')

%% LOAD TEST
tcount = 10000;
[timgs, tlabels] = readMNIST('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', tcount, 0);
pixel = size(reshape(timgs(:,:,1),[],1),1);
Bt = zeros(tcount,10); At = zeros(tcount,pixel);

for c = 1:tcount
    Bt(c,tlabels(c)+1) = 1;
    At(c,:) = reshape(timgs(:,:,c),1,[]);
end

%% P PROJECTION - pick Xn to project
P = zeros(28,28);
for k = 1:10
    P = P + reshape(X3(:,k),28,28);
end
imagesc(P)
colorbar

%% THRESHOLD PLOT
subplot(1,2,1)
imagesc(P)
colorbar
title('no threshold')

subplot(1,2,2)
thres = 0.0003;
indices = find(abs(P)<thres);
P(indices) = 0;
imagesc(P)
colorbar
caxis([-0.002,0.002])
title('threshold: abs(P)>0.0003')
X3_thres = X3;
X3_thres(indices,:)=0;

%% PLOT B
subplot(1,2,1)
imagesc(Bt(1:30,:))
colorbar
subplot(1,2,2)
Bpred = max1(A*X3); % change Xn
imagesc(Bpred(1:30,:))
colorbar

%% ERROR

Err1 = nnz(sum(abs(B-max1(A*X1)),2))/count
Err2 = nnz(sum(abs(B-max1(A*X2)),2))/count
Err3 = nnz(sum(abs(B-max1(A*X3)),2))/count
Err3_thres = nnz(sum(abs(B-max1(A*X3_thres)),2))/count
Err4 = nnz(sum(abs(B-max1(A*X4)),2))/count
Err5 = nnz(sum(abs(B-max1(A*X5)),2))/count

tErr1 = nnz(sum(abs(Bt-max1(At*X1)),2))/tcount
tErr2 = nnz(sum(abs(Bt-max1(At*X2)),2))/tcount
tErr3 = nnz(sum(abs(Bt-max1(At*X3)),2))/tcount
tErr3_thres = nnz(sum(abs(Bt-max1(At*X3_thres)),2))/tcount
tErr4 = nnz(sum(abs(Bt-max1(At*X4)),2))/tcount
tErr5 = nnz(sum(abs(Bt-max1(At*X5)),2))/tcount

%% BAR PLOT
x = categorical({'Lasso with threshold', 'SVD + Lasso', 'Averaging + SVD + lasso'});
x = reordercats(x,{'Lasso with threshold', 'SVD + Lasso', 'Averaging + SVD + lasso'});
y = [Err3_thres, tErr3_thres; 0.19623, 0.1926; 0.1466, 0.1374];
b = bar(x,y,1);
ylim([0 0.75])
ylabel("Error")
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
legend('Training Data','Test Data')


function Bpred = max1(input)
    [val,loc] = max(input, [], 2);
    Bpred = zeros(size(input));
    index = 1;
    for k = loc'
        Bpred(index,k) = 1;
        index = index +1;
    end
end
