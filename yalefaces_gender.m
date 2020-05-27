close all; clear all; clc;
%% IMPORT DATA
people = 39;
id_female = [5 15 22 27 28 32 34 37];
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
        %Y_cropped{k} = [Y_cropped{k} double(reshape(I,[],1))]; % NO EGDES
        Y_cropped{k} = [Y_cropped{k} double(reshape(double(I) + 60.*edge(I,'canny'),[],1))]; %40
        %S(s).data = I; % optional, save data.
    end
end
YC = []; % 1 male, 2 female
male = [];
female = [];
for k = 1:39
    if k ~= 14 
        if ismember(k,id_female)
            female = [female Y_cropped{k}];
        else
            male = [male Y_cropped{k}];
        end  
    end
end
YC = [male female]; % Image size: 192 168
YC = YC - repmat(mean(YC,1),size(YC,1),1); % Normalize
%male = YC(:,1:1903);
%female = YC(:,1904:end);
%%
% %imagesc(reshape(YC(:,100),192,168))
% [U_c, S_c, V_c] = svd(YC(:,2:end),'econ');
% %%
% for k = 1:8
%     subplot(2,4,k), imagesc(reshape(U_c(:,k),192,168))
%     colormap(gray);
% end
%% CROSS VALIDATE SET CREATIONS
KNN_error = []; LDA_error = []; NB_error = []; SVM_error = []; TREE_error = [];
for kfolds = 1:4
    Ytrain = [];
    trainLabel = [];
    Ytest = [];
    testLabel = [];

    m = randperm(size(male,2));
    f = randperm(size(female,2));
    trainLabel = [ones(1,1522) 2*ones(1,409)];
    Ytrain = [male(:,m(1:1522)) female(:,f(1:409))];
    testLabel = [ones(1,381) 2*ones(1,102)];
    Ytest = [male(:,m(1523:end)) female(:,f(410:end))];

    % SVD
    [U, S, V] = svd(Ytrain,'econ');
    % %% Projection in feature spaces:
    % 
    % %proj = (U(:,5:7)')*(Y_cropped{k});
    % %plot3(proj(1,:),proj(2,:),proj(3,:),'.','MarkerSize',10);
    % plot3(V(1:1522,7),V(1:1522,8),V(1:1522,9),'.','MarkerSize',15); hold on;
    % plot3(V(1523:end,7),V(1523:end,8),V(1523:end,9),'.','MarkerSize',15);
    % grid on;
    % legend; 
    %
    % KNN
    r_knn = 100;
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
    r_svm = 1000;
    svm = fitcsvm(V(:,1:r_svm),trainLabel');
    svmlabel = predict(svm,(S(1:r_svm,1:r_svm)\(U(:,1:r_svm)'*Ytest))');
    svm_error = nnz(svmlabel'-testLabel)/size(testLabel,2)

    % CLASSIFICATION TREES

    r_tree = 900;
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
ylim([0 0.5])
ylabel("Error")
xtips1 = b(5).XEndPoints + 0.04;
ytips1 = b(5).YEndPoints + 0.04;
%labels1 = string(b(3).YData);
labels1 = string(mean(y,2));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize',15)
set(gca,'fontsize',15)


