close all; clear; clc;
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
YC = YC - repmat(mean(YC,1),size(YC,1),1);
gender(1,1:1903) = 1;
gender(1,1904:2414) = 2;
%%
[U, S, V] = svd(YC,'econ');
%% K-MEANS
KM_Labels = []; GMM_Labels = [];
for m = 1:5
k = 2;
km_labels = kmeans(V(:,1:32),k);

% GMM
%GMModel = fitgmdist(V(:,1:32),k);
%gmm_labels = cluster(GMModel,V(:,1:32));
KM_Labels = [KM_Labels, km_labels - 1]; %GMM_Labels = [GMM_Labels, gmm_labels - 1];
end
KM_Labels = mean(KM_Labels,2);
GMM_Labels = mean(GMM_Labels,2);
KM_Lablels_perSub = zeros(1,64);
index = 1;
for k=1:39
    KM_Lablels_perSub(1:size(Y_cropped{k},2)) = KM_Lablels_perSub(1:size(Y_cropped{k},2)) + ...
            KM_Labels(index:(index - 1 + size(Y_cropped{k},2)))';
    index = index + size(Y_cropped{k},2);
end
bar(KM_Lablels_perSub);
xlabel('Image')
%ylabel('Category')
ylabel('Occurrence')
title('K-Means')
%%

for k = 1:64
    subplot(8,8,k), imagesc(reshape(YC(:,k),192,168))
    colormap(gray);
end

%%
subplot(1,2,1),bar(KM_Labels); ylim([0,2.5]); hold on;
g = bar(gender); hold off;
alpha(g,0.5)
xlabel('Image')
ylabel('Category')
title('K-Means')
subplot(1,2,2), bar(GMM_Labels); ylim([0,2.5]); hold on;
gm = bar(gender); hold off;
alpha(gm,0.5)
xlabel('Image')
ylabel('Category')
title('Gaussian Mixture Model')
