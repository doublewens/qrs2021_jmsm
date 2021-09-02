function re = conJMSM(source,target,options)
Xs = source(:,1:end-1);
Xs = zscore(Xs);
Ys = source(:,end);

Xt = target(:,1:end-1);
Xt = zscore(Xt);
Yt = target(:,end);

[pXs,pXt] = JMSM(Xs,Xt,options);

%% train the classifier
% lr
model = train(Ys,sparse(pXs),'-s 0 -c 1 -B -1 -q'); % num * fec
[~, ~, prob_estimates] = predict(Yt,sparse(pXt),model,'-b 1');
score = prob_estimates(:,model.Label==1);

try
    re = performance( Yt, score);
catch
    re.F1=nan; re.AUC=nan; re.MCC=nan;
end
end

