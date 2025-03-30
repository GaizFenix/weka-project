import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;

public class Arff2BowForStats {

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Uso: java Arff2BowForStats <input.csv> <output.arff>");
            System.exit(1);
        }

        String inputFile = args[0];
        String outputFile = args[1];

        try {
            // 1. Cargar el CSV
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(inputFile));
            Instances data = loader.getDataSet();

            System.out.println("=== Estructura inicial del dataset ===");
            System.out.println(data.toSummaryString());

            // 2. Procesar texto a Bag-of-Words (si hay atributos de texto)
            boolean hasStringAttrs = false;
            for (int i = 0; i < data.numAttributes(); i++) {
                if (data.attribute(i).isString()) {
                    hasStringAttrs = true;
                    break;
                }
            }

            if (hasStringAttrs) {
                System.out.println("\nProcesando atributos de texto a Bag-of-Words...");
                StringToWordVector filter = new StringToWordVector();
                filter.setInputFormat(data);
                data = Filter.useFilter(data, filter);
                
                System.out.println("\n=== Estructura después de StringToWordVector ===");
                System.out.println(data.toSummaryString());
            }

            // 3. Guardar como ARFF
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(outputFile));
            saver.writeBatch();

            System.out.println("\nConversión completada. Archivo guardado en: " + outputFile);
            System.out.println("Número total de atributos: " + data.numAttributes());
            System.out.println("Número total de instancias: " + data.numInstances());

        } catch (Exception e) {
            System.err.println("Error durante el procesamiento:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}