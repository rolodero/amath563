close all; clear all; clc;
%% IMPORT DATA
people = 39;
Y_cropped = cell([1 people]);
for k = 1:people
    if (k < 10)
        num = "0" + num2str(k);
    else
        num = "" + num2str(k);
    end
    D = 'CroppedYale\yaleB'+num;
    S = dir(fullfile(D,'*.pgm')); % pattern to match filenames.
    for s = 1:numel(S)
        F = fullfile(D,S(s).name);
        I = imread(F);
        %imshow(I)
        Y_cropped{k} = [Y_cropped{k} double(reshape(I,[],1))];
        %S(s).data = I; % optional, save data.
    end
end
YC = [];
subject = [];
for k = 1:39
    subject = [subject k.*ones(1,size(Y_cropped{k},2))];
    YC = [YC Y_cropped{k}];
end
YC = double(YC); % Image size: 192 168
% Normalize
YC = YC - repmat(mean(YC,1),size(YC,1),1);
%%
YU = [];
D = 'yalefaces';
    S = dir(fullfile(D,'subject*.*')); % pattern to match filenames.
    for s = 1:numel(S)
        F = fullfile(D,S(s).name);
        I = imread(F);
        size(I)
       % imshow(I)
        YU = [YU reshape(I,[],1)];
        %S(s).data = I; % optional, save data.
    end
YU = double(YU); % Image size: 243 320
YU = YU - repmat(mean(YU,1),size(YU,1),1);
%% SVD
[U_c, S_c, V_c] = svd(YC(:,2:end),'econ');
[U_u, S_u, V_u] = svd(YU(:,2:end),'econ');

%% First 9 eigenfaces YC + YU
for k = 1:4
    subplot(2,4,k), imagesc(reshape(U_c(:,k),192,168))
    colormap(gray);
    subplot(2,4,k+4), imagesc(reshape(U_u(:,k),243,320))
    colormap(gray);
end

%% Singular Value Spectrum YC + YU
subplot(1,2,1),semilogy(diag(S_c)/sum(diag(S_c)),'-*','LineWidth',1.5)
xlabel('Singular Value #')
ylabel('Normalized Value')
xlim([0,2400])

subplot(1,2,2),semilogy(diag(S_u)/sum(diag(S_u)),'-*','LineWidth',1.5)
xlabel('Singular Value #')
ylabel('Normalized Value')
xlim([0,155])

%% Rank r truncation YC
avgFace = mean(YC(:,2:end),2);
testFaceMS = YC(:,1) - avgFace;
index = 1;
for r=[10 50 100 300 500]
reconFace = avgFace + (U_c(:,1:r)*(U_c(:,1:r)')*testFaceMS);
subplot(1,5,index),imagesc(reshape(reconFace,192,168))
title(["r = "; num2str(r)])
colormap(gray)
index = index +1;
end

%% First 9 eigenfaces YU
for k = 1:9
    subplot(3,3,k), imagesc(reshape(U_u(:,k),243,320))
    colormap(gray);
end
%% Singular Value Spectrum YU
semilogy(diag(S_u)/sum(diag(S_u)),'-*','LineWidth',1.5)
xlabel('Singular Value #')
ylabel('Normalized Value')
xlim([0,155])

%%  ===== CLASSIFIER =====
% Redo SVD with subtracting mean

[U, S, V] = svd(YC,'econ');
%%
for k = 1:9
    subplot(3,3,k), imagesc(reshape(U(:,k),192,168))
    colormap(gray);
end

%% Projection in feature spaces:
for k = 1:1%39
    if (k ~= 14) 
        proj = (U(:,5:7)')*(Y_cropped{k});
        plot3(proj(1,:),proj(2,:),proj(3,:),'.','MarkerSize',10);
        %indx = find(subject == k);
        %plot3(V(indx,1),V(indx,2),V(indx,3),'.','MarkerSize',15);
        legend; hold on;
        grid on;
    end
end

%% CROSS VALIDATE SET CREATIONS
ntrain = 50; % ntest = rest

KNN_error = []; LDA_error = []; NB_error = []; SVM_error = []; TREE_error = [];
for kfolds = 1:5

Ytrain = [];
trainLabel = [];
Ytest = [];
testLabel = [];
for k = 1:39
    if (k ~= 14)
        p = randperm(size(Y_cropped{k},2));
        trainLabel = [trainLabel k.*ones(1,ntrain)];
        Ytrain = [Ytrain Y_cropped{k}(:,p(1:ntrain))];
        testLabel = [testLabel k.*ones(1,size(Y_cropped{k},2)-ntrain)];
        Ytest = [Ytest Y_cropped{k}(:,p(ntrain+1:end))];
    end
end

% SVD

[U, S, V] = svd(Ytrain,'econ');
% KNN
r_knn = 140;
K = 1;
Idx = knnsearch(V(:,1:r_knn),(S(1:r_knn,1:r_knn)\(U(:,1:r_knn)'*Ytest))','K',K);
knnlabel = [];
% for id=Idx    
%     knnlabel = [knnlabel trainLabel(id)];
% end
for id=1:size(Idx,1)
    row = Idx(id,:);
    lables = zeros(1,K);
    for k=1:K
        lables(k) = trainLabel(row(k));
    end
    [M,F] = mode(lables);
    if F == 1
        knnlabel = [knnlabel lables(1)];
    else
        knnlabel = [knnlabel M];
    end
end
knn_error = nnz(knnlabel-testLabel)/size(testLabel,2)

% LINEAR DISCRIMINANT ANALYSIS
r_lda = 1000;
ldalabel = classify((S(1:r_lda,1:r_lda)\(U(:,1:r_lda)'*Ytest))',V(:,1:r_lda),trainLabel');
lda_error = nnz(ldalabel'-testLabel)/size(testLabel,2)

% NAIVE BAYES
r_nb = 200;
nb = fitcnb(V(:,1:r_nb),trainLabel');
nblabel = nb.predict((S(1:r_nb,1:r_nb)\(U(:,1:r_nb)'*Ytest))');
nb_error = nnz(nblabel'-testLabel)/size(testLabel,2)

% SUPPORT VECTOR MACHINES
r_svm = 600;
svm = fitcecoc(V(:,1:r_svm),trainLabel');
svmlabel = predict(svm,(S(1:r_svm,1:r_svm)\(U(:,1:r_svm)'*Ytest))');
svm_error = nnz(svmlabel'-testLabel)/size(testLabel,2)

% CLASSIFICATION TREES

r_tree = 800;
tree = fitctree(V(:,1:r_tree),trainLabel');
treelabel = predict(tree,(S(1:r_tree,1:r_tree)\(U(:,1:r_tree)'*Ytest))');
tree_error = nnz(treelabel'-testLabel)/size(testLabel,2)


KNN_error = [KNN_error knn_error];
LDA_error = [LDA_error lda_error]; 
NB_error = [NB_error nb_error];
SVM_error = [SVM_error svm_error]; 
TREE_error = [TREE_error tree_error];
end

%% Error Plot

x = categorical({'KNN', 'LDA', 'Bayes', 'SVM', 'Tree'});
x = reordercats(x,{'KNN', 'LDA', 'Bayes', 'SVM', 'Tree'});
y = [KNN_error; LDA_error; NB_error; SVM_error; TREE_error];
b = bar(x,y,1);
ylim([0 0.75])
ylabel("Error")
xtips1 = b(3).XEndPoints + 0.08;
ytips1 = b(3).YEndPoints + 0.08;
%labels1 = string(b(3).YData);
labels1 = string(mean(y,2));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize',15)
set(gca,'fontsize',15)

