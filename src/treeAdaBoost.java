import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

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

        if (args.length != 4) {
            System.err.println("ERROR! Correct usage: java treeAdaBoost <input_train.arff> <input_dev.arff> <input_dictionary.txt> <output_stats.txt>");
            System.exit(1);
        }

        String inTrainPath = args[0];
        String inDevPath = args[1];
        String inDictionaryPath = args[2];
        String outStatsPath = args[3];

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
            filter.setIDFTransform(true);
            filter.setTFTransform(true);
            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setInputFormat(dev);

            Instances newDev = Filter.useFilter(dev, filter);

            SparseToNonSparse filter2 = new SparseToNonSparse();
            filter2.setInputFormat(newDev);

            newDev = Filter.useFilter(newDev, filter2);

            // Optimize hyperapameters (AdaBoost + J48)
            double[] confidenceFactors = {0.05, 0.1, 0.15, 0.25};
            int[] minNumInstances = {2, 5, 10};

            double bestAccuracy = 0.0;
            double bestC = 0.0;
            int bestM = 0;
            int iter = 1;

            for (double c : confidenceFactors) {
                for (int m : minNumInstances) {
                    System.out.println("Iteration " + iter + " -> Confidence factor: " + c + " and min num instances: " + m);
                    // Setup J48 classifier
                    J48 tree = new J48();
                    tree.setConfidenceFactor((float) c);
                    tree.setMinNumObj(m);

                    // Setup AdaBoost
                    AdaBoostM1 classifier = new AdaBoostM1();
                    classifier.setClassifier(tree);
                    classifier.setUseResampling(false);
                    classifier.buildClassifier(train);

                    // Train and test the model
                    Evaluation eval = new Evaluation(train);
                    eval.evaluateModel(classifier, newDev);

                    // Metrics and hyperparameter comparison
                    if (eval.pctCorrect() > bestAccuracy) {
                        bestAccuracy = eval.pctCorrect();
                        bestC = c;
                        bestM = m;
                    }
                }
            }

            // Train final model and get stats
            J48 tree = new J48();
            tree.setConfidenceFactor((float) bestC);
            tree.setMinNumObj(bestM);

            AdaBoostM1 classifier = new AdaBoostM1();
            classifier.setClassifier(tree);
            classifier.setUseResampling(false);
            classifier.buildClassifier(train);

            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(classifier, newDev);

            // Save stats
            String stats = "===== AdaBoost + J48 with best hyperparameters =====\n";
            stats += "Confidence Factor: " + bestC + "\n";
            stats += "Min Num Instances: " + bestM + "\n";
            stats += eval.toSummaryString() + "\n";
            stats += eval.toClassDetailsString() + "\n";
            stats += eval.toMatrixString() + "\n";

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outStatsPath))) {
                writer.write(stats);
                System.out.println("File saved to: " + outStatsPath);
            } catch (IOException e) {
                System.err.println("Error writing to file: " + outStatsPath);
                e.printStackTrace();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
