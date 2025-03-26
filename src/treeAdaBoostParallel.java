import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class treeAdaBoostParallel {

    // A simple class to store hyperparameter results.
    static class HyperParamResult {
        int numIterations;
        int weightThreshold;
        double accuracy;

        HyperParamResult(int numIterations, int weightThreshold, double accuracy) {
            this.numIterations = numIterations;
            this.weightThreshold = weightThreshold;
            this.accuracy = accuracy;
        }
    }

    public static void main(String[] args) {

        if (args.length != 4) {
            System.err.println("ERROR! Correct usage: java treeAdaBoostParallel <input_train.arff> <input_dev.arff> <input_dictionary.txt> <output_stats.txt>");
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

            // Adequate dev to train data using dictionary-based filter
            FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
            filter.setDictionaryFile(new File(inDictionaryPath));
            filter.setIDFTransform(true);
            filter.setTFTransform(true);
            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setInputFormat(dev);

            Instances initialDev = Filter.useFilter(dev, filter);

            SparseToNonSparse filter2 = new SparseToNonSparse();
            filter2.setInputFormat(initialDev);
            final Instances newDev = Filter.useFilter(initialDev, filter2);

            // Hyperparameters to search
            int[] iterationsArray = IntStream.rangeClosed(1, 5).toArray(); // {1, 2, 3, 4, 5}
            int[] weightsArray = {80, 85, 90, 95, 100};

            // Create list of all hyperparameter combinations
            List<int[]> hyperParamPairs = Arrays.stream(iterationsArray)
                    .boxed()
                    .flatMap(iter -> 
                        Arrays.stream(weightsArray)
                              .mapToObj(weight -> new int[]{iter, weight})
                    )
                    .collect(Collectors.toList());

            // Evaluate each hyperparameter pair in parallel
            HyperParamResult bestResult = hyperParamPairs.parallelStream()
                    .map(pair -> {
                        int numIter = pair[0];
                        int weight = pair[1];
                        System.out.printf("Evaluating: Iterations = %d, Weight Threshold = %d%n", numIter, weight);

                        try {
                            // Setup J48 classifier
                            J48 tree = new J48();
                            // Optionally set additional tree parameters here

                            // Setup AdaBoost with the current hyperparameters
                            AdaBoostM1 classifier = new AdaBoostM1();
                            classifier.setClassifier(tree);
                            classifier.setUseResampling(false);
                            classifier.setNumIterations(numIter);
                            classifier.setWeightThreshold(weight);
                            classifier.buildClassifier(train);

                            // Evaluate classifier on the development set
                            Evaluation eval = new Evaluation(train);
                            eval.evaluateModel(classifier, newDev);
                            double accuracy = eval.pctCorrect();

                            return new HyperParamResult(numIter, weight, accuracy);
                        } catch (Exception e) {
                            e.printStackTrace();
                            return new HyperParamResult(numIter, weight, 0.0);
                        }
                    })
                    .max(Comparator.comparingDouble(result -> result.accuracy))
                    .orElse(null);

            // Check if a best result was found
            if (bestResult == null) {
                System.err.println("No valid hyperparameter evaluation result obtained.");
                return;
            }

            // Use best hyperparameters to train the final model
            System.out.println("Best Hyperparameters Found:");
            System.out.println("Iterations: " + bestResult.numIterations);
            System.out.println("Weight Threshold: " + bestResult.weightThreshold);
            System.out.println("Accuracy: " + bestResult.accuracy);

            // Train final model with best hyperparameters
            J48 finalTree = new J48();
            AdaBoostM1 finalClassifier = new AdaBoostM1();
            finalClassifier.setClassifier(finalTree);
            finalClassifier.setUseResampling(false);
            finalClassifier.setNumIterations(bestResult.numIterations);
            finalClassifier.setWeightThreshold(bestResult.weightThreshold);
            finalClassifier.buildClassifier(train);

            Evaluation finalEval = new Evaluation(train);
            finalEval.evaluateModel(finalClassifier, newDev);

            // Save evaluation statistics to file
            String stats = "===== AdaBoost + J48 with best hyperparameters =====\n";
            stats += "Iterations: " + bestResult.numIterations + "\n";
            stats += "Weight Threshold: " + bestResult.weightThreshold + "\n";
            stats += finalEval.toSummaryString() + "\n";
            stats += finalEval.toClassDetailsString() + "\n";
            stats += finalEval.toMatrixString() + "\n";

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
