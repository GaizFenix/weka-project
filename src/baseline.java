import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import java.io.FileWriter;
import java.io.PrintWriter;
import weka.core.Utils;

public class baseline {
    public static void main(String[] args) {
        try {

            if (args.length != 2) {
                System.err.println("ERROR! Correct usage: java baseline <input_clean.arff> <output_file.txt>");
                System.exit(1);
            }

            String inputArff = args[0]; 
            String outputFile = args[1]; 
    
            DataSource source = new DataSource(inputArff);
            Instances data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
    
            double[] accuracies = new double[25];
            double[] precisions = new double[25];
            double[] recalls = new double[25];
            double[] f1Scores = new double[25];
            double[] tprs = new double[25]; // True Positive Rate
            double[] fprs = new double[25]; // False Positive Rate
            double[] tnrs = new double[25]; // True Negative Rate
            double[] fnrs = new double[25]; // False Negative Rate
            double[][] confusionMatrix = null;
            double meanROC = 0;
            double meanPRC = 0;
            double meanKappa = 0;

            try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {

                for (int seed = 1; seed <= 25; seed++) {
                    J48 classifier = new J48();
                    classifier.setSeed(seed);
                    classifier.buildClassifier(data);
        
                    Evaluation eval = new Evaluation(data);
                    eval.crossValidateModel(classifier, data, 10, new java.util.Random(seed));
                    accuracies[seed - 1] = eval.pctCorrect();
                    precisions[seed - 1] = eval.weightedPrecision();
                    recalls[seed - 1] = eval.weightedRecall();
                    f1Scores[seed - 1] = eval.weightedFMeasure();

                    meanROC += eval.weightedAreaUnderROC();
                    meanPRC += eval.weightedAreaUnderPRC();
                    meanKappa += eval.kappa();

                    // Métricas adicionales por semilla
                    tprs[seed - 1] = eval.truePositiveRate(0); // TPR para la clase 0
                    fprs[seed - 1] = eval.falsePositiveRate(0); // FPR para la clase 0
                    tnrs[seed - 1] = 1 - eval.falsePositiveRate(0); // TNR = 1 - FPR
                    fnrs[seed - 1] = 1 - eval.truePositiveRate(0); // FNR = 1 - TPR
                    
                    if (seed == 25) {
                        confusionMatrix = eval.confusionMatrix();
                    }
                }
    
                double meanAccuracy = calculateMean(accuracies);
                double stdDevAccuracy = calculateStdDev(accuracies, meanAccuracy);
                double meanPrecision = calculateMean(precisions);
                double meanRecall = calculateMean(recalls);
                double meanF1Score = calculateMean(f1Scores);
                meanROC /= 25;
                meanPRC /= 25;
                meanKappa /= 25;
                double meanTPR = calculateMean(tprs);
                double meanFPR = calculateMean(fprs);
                double meanTNR = calculateMean(tnrs);
                double meanFNR = calculateMean(fnrs);
    
                // Sección de métricas generales
                writer.println("===== General Metrics =====");
                writer.println("Mean Accuracy: " + meanAccuracy);
                writer.println("Standard Deviation (Accuracy): " + stdDevAccuracy);
                writer.println("Mean Precision: " + meanPrecision);
                writer.println("Mean Recall: " + meanRecall);
                writer.println("Mean F1-Score: " + meanF1Score);
                writer.println("Mean ROC Area: " + meanROC);
                writer.println("Mean PRC Area: " + meanPRC);
                writer.println("Mean Kappa Statistic: " + meanKappa);
                writer.println("Mean TPR (True Positive Rate): " + meanTPR);
                writer.println("Mean FPR (False Positive Rate): " + meanFPR);
                writer.println("Mean TNR (True Negative Rate): " + meanTNR);
                writer.println("Mean FNR (False Negative Rate): " + meanFNR);

                if (confusionMatrix != null) {
                    writer.println("Confusion Matrix (last seed):");
                    for (double[] row : confusionMatrix) {
                        writer.println(Utils.arrayToString(row));
                    }
                }

                // Sección de métricas por semilla
                writer.println("\n===== Metrics by Seed =====");
                writer.println("Accuracies for each seed:");
                for (int i = 0; i < accuracies.length; i++) {
                    writer.println("Seed " + (i + 1) + ": " + accuracies[i]);
                }

                writer.println("Precisions for each seed:");
                for (int i = 0; i < precisions.length; i++) {
                    writer.println("Seed " + (i + 1) + ": " + precisions[i]);
                }

                writer.println("Recalls for each seed:");
                for (int i = 0; i < recalls.length; i++) {
                    writer.println("Seed " + (i + 1) + ": " + recalls[i]);
                }

                writer.println("F1-Scores for each seed:");
                for (int i = 0; i < f1Scores.length; i++) {
                    writer.println("Seed " + (i + 1) + ": " + f1Scores[i]);
                }

                writer.println("\nTPRs (True Positive Rate) for each seed:");
                for (int i = 0; i < tprs.length; i++) {
                    writer.println("Seed " + (i + 1) + ": " + tprs[i]);
                }

                writer.println("\nFPRs (False Positive Rate) for each seed:");
                for (int i = 0; i < fprs.length; i++) {
                    writer.println("Seed " + (i + 1) + ": " + fprs[i]);
                }

                writer.println("\nTNRs (True Negative Rate) for each seed:");
                for (int i = 0; i < tnrs.length; i++) {
                    writer.println("Seed " + (i + 1) + ": " + tnrs[i]);
                }

                writer.println("\nFNRs (False Negative Rate) for each seed:");
                for (int i = 0; i < fnrs.length; i++) {
                    writer.println("Seed " + (i + 1) + ": " + fnrs[i]);
                }
                

            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    //Calcula la media
    private static double calculateMean(double[] values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }
    //Calcula la desviacion estandar
    private static double calculateStdDev(double[] values, double mean) {
        double variance = 0;
        for (double value : values) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= values.length;
        return Math.sqrt(variance);
    }
}
