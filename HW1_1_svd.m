close all; clear all; clc;
count = 60000; offset = 0;
[imgs, labels] = readMNIST('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', count, offset);
pixel = size(reshape(imgs(:,:,1),[],1),1);
B = zeros(count,10); A = zeros(count,pixel);

for c = 1:count
    B(c,labels(c)+1) = 1;
    A(c,:) = reshape(imgs(:,:,c),1,[]);
end

tcount = 10000;
[timgs, tlabels] = readMNIST('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', tcount, 0);
pixel = size(reshape(timgs(:,:,1),[],1),1);
Bt = zeros(tcount,10); At = zeros(tcount,pixel);

for c = 1:tcount
    Bt(c,tlabels(c)+1) = 1;
    At(c,:) = reshape(timgs(:,:,c),1,[]);
end

%% SUBTRACT MEAN
A_mean = mean(A,1);
A = A - repmat(A_mean,count,1);
tA_mean = mean(At,1);
At = At - repmat(tA_mean,tcount,1);

%% SVD
[U, S, V] = svd(A','econ');

%% EV PLOT
semilogy(diag(S),'b','LineWidth',2)
ylim([10,1e6])
xlabel('eigenvalues')

%% PLOT PROJECTIONS
P = zeros(28,28);
for k = 1:9
    subplot(3,3,k)
    P = reshape(U(:,k),28,28);
    imagesc(P)
colorbar
end

%% RANK AND PLOT OF MAPPING
A_svd = A*U;
dim = 100;
imagesc(A_svd(1:50,1:100))

%% LASSO REGRESSION
for k = 1:10
[XL1, FitInfo] = lasso(A_svd(:,1:dim),B(:,k),'CV',10);
Y(:,k) = XL1(:,FitInfo.IndexMinMSE);
end

%% ERROR
X_svd = U(:,1:dim)*Y;
P = zeros(28,28);
for k = 1:10
    P = P + reshape(X_svd(:,k),28,28);
end
imagesc(P)
colorbar

ErrSVD = nnz(sum(abs(B-max1(A*X_svd)),2))/count
tErrSVD = nnz(sum(abs(Bt-max1(At*X_svd)),2))/tcount

function Bpred = max1(input)
    [val,loc] = max(input, [], 2);
    Bpred = zeros(size(input));
    index = 1;
    for k = loc'
        Bpred(index,k) = 1;
        index = index +1;
    end
end