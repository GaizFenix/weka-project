import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class arff2bow {
    public static void main(String[] args) {
        try {
            // Load data from cleaned CSV
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(args[0])); // Input file in first program argument
            loader.setFieldSeparator(",");  // Change if needed
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); // class attribute is last
            System.out.println(data.numInstances() + " " + data.numAttributes() + " " + data.numClasses());

            // Save in .arff format
            ArffSaver as = new ArffSaver();
            as.setFile(new File(args[1]));
            as.setInstances(data);
            as.writeBatch();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
