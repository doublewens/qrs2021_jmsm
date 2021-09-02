function perf = performance( actual_label, probPos)
probPos(probPos>1) = 1;
predicted_label = double(probPos>=0.5); %

cf=confusionmat(actual_label,predicted_label); % Matlab build-in function

if numel(cf)==1
    if unique(actual_label)==1 && unqiue(predicted_label)==1 % only positive sampels and all they are classified correctly
        TP = cf; TN = 0; FP = 0; FN = 0; % 
    else % only negative sampels and all they are classified correctly
        TP = 0; TN = cf; FP = 0; FN = 0;
    end
else
    TP=cf(2,2);
    TN=cf(1,1);
    FP=cf(1,2);
    FN=cf(2,1);
end

%% various performance
PD=TP/(TP+FN);
Precision=TP/(TP+FP);
F1=2*Precision*PD/(Precision+PD);
[~,~,~,AUC]=perfcurve(actual_label, probPos, '1');% 
MCC = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));

perf.F1=F1; 
perf.AUC=AUC; 
perf.MCC=MCC;
end