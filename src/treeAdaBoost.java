import java.io.File;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class treeAdaBoost {
    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("ERROR! Correct usage: java treeAdaBoost <input_train.arff> <input_dev.arff> <input_dictionary.txt>");
            System.exit(1);
        }

        // Record start time
        long startTime = System.currentTimeMillis();
        System.out.println("Program started at: " + new java.util.Date(startTime));

        String inTrainPath = args[0];
        String inDevPath = args[1];
        String inDictionaryPath = args[2];
        // String outStatsPath = args[3];

        try {
            // Load train and dev data
            DataSource dsTrain = new DataSource(inTrainPath);
            Instances train = dsTrain.getDataSet();
            train.setClassIndex(0);

            DataSource dsDev = new DataSource(inDevPath);
            Instances dev = dsDev.getDataSet();
            dev.setClassIndex(dev.numAttributes() - 1);

            // Adequate dev to train data
            FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
            filter.setDictionaryFile(new File(inDictionaryPath));
            // filter.setIDFTransform(true);
            // filter.setTFTransform(true);
            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setInputFormat(dev);

            Instances newDev = Filter.useFilter(dev, filter);

            SparseToNonSparse filter2 = new SparseToNonSparse();
            filter2.setInputFormat(newDev);

            newDev = Filter.useFilter(newDev, filter2);

            // J48
            J48 j48 = new J48();
            j48.setConfidenceFactor(0.25f);
            j48.setMinNumObj(2);

            // AdaBoost with J48 as base mdl
            AdaBoostM1 ada = new AdaBoostM1();
            ada.setWeightThreshold(90);
            ada.setNumIterations(15);
            ada.setClassifier(j48);
            ada.setDebug(false);

            // Build model
            ada.buildClassifier(train);

            // Evaluate model
            Evaluation eval = new Evaluation(newDev);
            eval.evaluateModel(ada, newDev);
        
            System.out.println("\n=== Dev set evaluation ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
            System.out.println("Number of iterations: " + ada.getNumIterations());
            System.out.println("Weight threshold: " + ada.getWeightThreshold());
            
            // Record end time
            long endTime = System.currentTimeMillis();
            System.out.println("Program finished at: " + new java.util.Date(endTime));
            
            // Calculate and print elapsed time
            long elapsedTime = endTime - startTime;
            System.out.println("Elapsed time (seconds): " + (elapsedTime / 1000.0));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
