% A demo of EQ->ant-1.7 with JMSM
clc
clear
import weka.filters.*;
import weka.*;
addpath('.\utility\');
addpath('.\classifier\liblinear\');

% Set saving path of emperiment results
filePath = ['.\output\exp-',datestr(datetime,'yyyy-mm-dd'),'\'];
if ~exist(filePath,'dir') %
    mkdir(filePath);
end

% Load source and target projects
dataPath = '.\datasets\';
source_project = 'EQ';
target_project = 'ant-1.7';
source = loadArff([dataPath,source_project,'.arff']);
target = loadArff([dataPath,target_project,'.arff']);

% Experiment Settings
runtimes = 20;    % the number of runnings
percent = 0.9;    % the percentage of training data in source data

% Initialization
F1=zeros(runtimes,1);AUC=zeros(runtimes,1);MCC=zeros(runtimes,1);
sourcesCopy = source;

% parameters of JMSM
options.T = 5;
options.mu = 1;
options.ep = 10^(-6);

for i=1:runtimes
    % Split the source data
    rand('seed',i);
    idx = randperm(size(sourcesCopy,1),round(percent*size(sourcesCopy,1)));
    trainData = sourcesCopy(idx,:);
    while numel(unique(trainData(:,end)))==1
        idx = randperm(size(sourcesCopy,1),round(percent*size(sourcesCopy,1)));
        trainData = sourcesCopy(idx,:);
    end
    source = trainData;   
    re = conJMSM(source, target, options);
    F1(i,:)=re.F1; 
    AUC(i,:)=re.AUC;
    MCC(i,:)=re.MCC;    
end
combination_result = [F1,AUC,MCC];
save([filePath,'\',source_project,'_to_',target_project,'.mat'],'combination_result')