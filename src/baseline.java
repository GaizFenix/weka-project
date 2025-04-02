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

            int numSeeds = 20;
            int[] seeds = {5, 162, 182, 197, 267, 345, 378, 388, 497, 625, 638, 668, 685, 704, 756, 766, 770, 812, 937, 956};
    
            DataSource source = new DataSource(inputArff);
            Instances data = source.getDataSet();
            data.setClassIndex(0);
    
            double[] accuracies = new double[numSeeds];
            double[] precisions = new double[numSeeds];
            double[] recalls = new double[numSeeds];
            double[] f1Scores = new double[numSeeds];
            double[] tprs = new double[numSeeds]; // True Positive Rate
            double[] fprs = new double[numSeeds]; // False Positive Rate
            double[] tnrs = new double[numSeeds]; // True Negative Rate
            double[] fnrs = new double[numSeeds]; // False Negative Rate
            double[] rocs = new double[numSeeds]; // ROC Area
            double[] prcs = new double[numSeeds]; // PRC Area
            double[] kappas = new double[numSeeds]; // Kappa Statistic
            double[][] confusionMatrix = null;

            try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {

                for (int it = 1; it <= numSeeds; it++) {

                    long startTime = System.currentTimeMillis();
                    // Imprimir el tiempo de inicio
                    System.out.println("Start time: " + startTime);

                    // Imprimir la semilla actual
                    System.out.println("Seed " + it + ": " + seeds[it - 1]);
                    System.out.println("Iteration " + it + " of " + numSeeds);

                    // Configurar el filtro Resample para el stratified
                    Resample resample = new Resample();
                    resample.setRandomSeed(seeds[it - 1]); // Semilla para la aleatoriedad
                    resample.setSampleSizePercent(70); // 70% para entrenamiento
                    resample.setNoReplacement(false); // Sin reemplazo
                    resample.setBiasToUniformClass(0.3); // Mantener la distribucion de las clases
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

                    accuracies[it-1] = eval.pctCorrect();
                    precisions[it - 1] = eval.weightedPrecision();
                    recalls[it - 1] = eval.weightedRecall();
                    f1Scores[it - 1] = eval.weightedFMeasure();
                    rocs[it - 1] = eval.weightedAreaUnderROC();
                    prcs[it - 1] = eval.weightedAreaUnderPRC();
                    kappas[it - 1] = eval.kappa();

                    // Métricas adicionales por semilla
                    tprs[it - 1] = eval.truePositiveRate(0); // TPR para la clase 0
                    fprs[it - 1] = eval.falsePositiveRate(0); // FPR para la clase 0
                    tnrs[it - 1] = 1 - eval.falsePositiveRate(0); // TNR = 1 - FPR
                    fnrs[it - 1] = 1 - eval.truePositiveRate(0); // FNR = 1 - TPR

                    // Imprimir el tiempo de finalización
                    long endTime = System.currentTimeMillis();
                    System.out.println("End time: " + endTime);
                    // Calcular el tiempo total de ejecución
                    long totalTime = endTime - startTime;
                    System.out.println("Total time: " + totalTime/1000 + " seconds");

                    if (it == numSeeds) {
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
                writer.println("Standard Deviation (Precision): " + calculateStdDev(precisions, meanPrecision));
                writer.println("Mean Recall: " + meanRecall);
                writer.println("Standard Deviation (Recall): " + calculateStdDev(recalls, meanRecall));
                writer.println("Mean F1-Score: " + meanF1Score);
                writer.println("Standard Deviation (F1-Score): " + calculateStdDev(f1Scores, meanF1Score));
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
