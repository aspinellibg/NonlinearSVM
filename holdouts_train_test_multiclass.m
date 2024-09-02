function [DATAtrain, DATAtest] = holdouts_train_test_multiclass(DATA, testingsamplesize)

dp = cvpartition(DATA.CLASS, 'HoldOut', testingsamplesize);

idxtrain = training(dp);
DATAtrain = DATA(idxtrain,:);
DATAtrain = table2array(DATAtrain);

idxtest = test(dp);
DATAtest = DATA(idxtest,:);
DATAtest = table2array(DATAtest);


end