function data = loadArff( path )
file1 = java.io.File(path);  % create a Java File object (arff file is just a text file)
loader = weka.core.converters.ArffLoader;  % create an ArffLoader object
loader.setFile(file1);  % using ArffLoader to load data in file .arff
insts = loader.getDataSet; % get an Instances object
insts.setClassIndex(insts.numAttributes()-1); %  set the index of class label
[data,~,~,~,~] = weka2matlab(insts,[]); %{XXX,YYY}-->{0,1}
data = [data(:, 1:end-1), double(data(:, end)>0)]; % If defects(i) > 0, then defects(i) = 1, otherwise defects(i) = 0.
end