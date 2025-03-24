import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
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
            double[] rocs = new double[25]; // ROC Area
            double[] prcs = new double[25]; // PRC Area
            double[] kappas = new double[25]; // Kappa Statistic
            double[][] confusionMatrix = null;

            try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {

                for (int seed = 1; seed <= 25; seed++) {
                    // Configurar el filtro Resample para el stratified
                    Resample resample = new Resample();
                    resample.setRandomSeed(seed);
                    resample.setSampleSizePercent(70); // 70% para entrenamiento
                    resample.setNoReplacement(true); // Sin reemplazo
                    resample.setBiasToUniformClass(1.0); // Mantener la distribucion de las clases
                    resample.setInvertSelection(false); // Seleccionar el conjunto de entrenamiento
                    resample.setInputFormat(data);

                    // Crear el conjunto de entrenamiento(70%)
                    Instances train = Filter.useFilter(data, resample);

                    // Crear el conjunto de prueba (30%)
                    resample.setInvertSelection(true); // Seleccionar el conjunto de prueba
                    Instances test = Filter.useFilter(data, resample);

                    // Entrenar el clasificador
                    J48 classifier = new J48();
                    classifier.buildClassifier(train);

                    // Evaluar el modelo
                    Evaluation eval = new Evaluation(test);
                    eval.evaluateModel(classifier, test);

                    accuracies[seed - 1] = eval.pctCorrect();
                    precisions[seed - 1] = eval.weightedPrecision();
                    recalls[seed - 1] = eval.weightedRecall();
                    f1Scores[seed - 1] = eval.weightedFMeasure();
                    rocs[seed - 1] = eval.weightedAreaUnderROC();
                    prcs[seed - 1] = eval.weightedAreaUnderPRC();
                    kappas[seed - 1] = eval.kappa();

                    // Métricas adicionales por semilla
                    tprs[seed - 1] = eval.truePositiveRate(0); // TPR para la clase 0
                    fprs[seed - 1] = eval.falsePositiveRate(0); // FPR para la clase 0
                    tnrs[seed - 1] = 1 - eval.falsePositiveRate(0); // TNR = 1 - FPR
                    fnrs[seed - 1] = 1 - eval.truePositiveRate(0); // FNR = 1 - TPR

                    if (seed == 25) {
                        confusionMatrix = eval.confusionMatrix();
                    }
                }

                // Calcular métricas generales
                double meanAccuracy = calculateMean(accuracies);
                double stdDevAccuracy = calculateStdDev(accuracies, meanAccuracy);
                double meanPrecision = calculateMean(precisions);
                double meanRecall = calculateMean(recalls);
                double meanF1Score = calculateMean(f1Scores);
                double meanROC = calculateMean(rocs);
                double meanPRC = calculateMean(prcs);
                double meanKappa = calculateMean(kappas);
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

                writer.println("\nPrecisions for each seed:");
                for (int i = 0; i < precisions.length; i++) {
                    writer.println("Seed " + (i + 1) + ": " + precisions[i]);
                }

                writer.println("\nRecalls for each seed:");
                for (int i = 0; i < recalls.length; i++) {
                    writer.println("Seed " + (i + 1) + ": " + recalls[i]);
                }

                writer.println("\nF1-Scores for each seed:");
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
    //Media
    private static double calculateMean(double[] values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }
    //Desviación estándar
    private static double calculateStdDev(double[] values, double mean) {
        double variance = 0;
        for (double value : values) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= values.length;
        return Math.sqrt(variance);
    }
}
