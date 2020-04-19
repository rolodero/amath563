close all; clear all; clc;
count = 60000; offset = 0;
[imgs, labels] = readMNIST('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', count, offset);
pixel = size(reshape(imgs(:,:,1),[],1),1);
A = {[]}; B = {[]}; X = {[]};
for k=2:10
    A{k} = [];
end
for c = 1:count
    A{labels(c)+1} = [A{labels(c)+1}; reshape(imgs(:,:,c),1,[])];
end
for k=1:10
    sz = size(A{k});
    B{k} = zeros(sz(1),10);
    B{k}(:,k) = ones(sz(1),1);
end

%%
tcount = 10000;
[timgs, tlabels] = readMNIST('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', tcount, 0);
pixel = size(reshape(timgs(:,:,1),[],1),1);
At = {[]}; Bt = {[]};
for k=2:10
    At{k} = [];
end
for c = 1:tcount
    At{tlabels(c)+1} = [At{tlabels(c)+1}; reshape(timgs(:,:,c),1,[])];
end
for k=1:10
    sz = size(At{k});
    Bt{k} = zeros(sz(1),10);
    Bt{k}(:,k) = ones(sz(1),1);
end

%% SVD FOR EACH DIGIT
[U, S, V] = svd(A{3}','econ');
P = zeros(28,28);
for k = 1:9
    subplot(3,3,k)
    P = reshape(U(:,k),28,28);
    imagesc(P)
colorbar
end

%% PLOTS AND ERROR CALC
dim = 10;
for k = 1:10
[U, S, V] = svd(A{k}','econ');

Acurr = A{k}*U;
Bcurr = B{k}(:,k);
Y = Acurr(:,1:dim)\Bcurr;
X = U(:,1:dim)*Y;
P = zeros(28,28);
    P = P + reshape(X(:,1),28,28);
subplot(2,5,k)
imagesc(P)
colorbar
tErr(k) = nnz(sum(abs(Bt{k}(:,k)-At{k}*X),2))/tcount;
end
%% ERROR PLOT
x = categorical({'0', '1', '2','3', '4', '5','6', '7', '8','9'});
x = reordercats(x,{'0', '1', '2','3', '4', '5','6', '7', '8','9'});
b = bar(x,tErr,0.8);
ylim([0 0.2])
ylabel("Error")
xlabel("Digits")
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')


function Bpred = max1(input)
    [val,loc] = max(input, [], 2);
    Bpred = zeros(size(input));
    index = 1;
    for k = loc'
        Bpred(index,k) = 1;
        index = index +1;
    end
end