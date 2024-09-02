function [Xtrain, Xtest, Ytrain, Ytest] = holdouts_train_test(DATA, testingsamplesize)

dp = cvpartition(DATA.Class, 'HoldOut', testingsamplesize);

idxtrain = training(dp);
DATAtrain = DATA(idxtrain,:);

idxtest = test(dp);
DATAtest = DATA(idxtest,:);

Class = DATAtrain{:,end};
X = DATAtrain(Class==1, :);
Y = DATAtrain(Class==0, :);

X(:,end)=[];
Y(:,end)=[];

Class = DATAtest{:,end};
Xtest = DATAtest(Class==1, :);
Ytest = DATAtest(Class==0, :);

Xtest(:,end)=[];
Ytest(:,end)=[];

Xtrain=table2array(X);
Ytrain=table2array(Y);
Xtest=table2array(Xtest);
Ytest=table2array(Ytest);

end