import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class ARFFAnalyzer {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Uso: java ARFFAnalyzer <archivo.arff> <salida.txt>");
            return;
        }

        String arffFile = args[0];
        String outputFile = args[1];

        try {
            // Cargar el conjunto de datos
            DataSource source = new DataSource(arffFile);
            Instances data = source.getDataSet();

            // Inicializar contadores
            int nominalCount = 0;
            int numericCount = 0;
            int stringCount = 0;
            int booleanCount = 0;
            int totalAttributes = data.numAttributes();
            int totalInstances = data.numInstances();

            // Contar atributos por tipo
            for (int i = 0; i < totalAttributes; i++) {
                Attribute attr = data.attribute(i);
                if (attr.isNominal()) {
                    
                    nominalCount++;
                } else if (attr.isNumeric()) {
                    numericCount++;
                } else if (attr.isString()) {
                    stringCount++;
                } else if (attr.isRelationValued()) {
                    // No contamos como tipo básico
                } else if (attr.isDate()) {
                    // Tratamos las fechas como numéricas para este análisis
                    numericCount++;
                } else {
                    // Asumimos boolean si no es ninguno de los anteriores
                    booleanCount++;
                }
            }

            // Contar instancias por clase (si hay atributo clase)
            Map<String, Integer> classCounts = new HashMap<>();
            if (data.classIndex() >= 0) {
                Attribute classAttr = data.classAttribute();
                for (int i = 0; i < totalInstances; i++) {
                    String classValue = data.instance(i).stringValue(classAttr);
                    classCounts.put(classValue, classCounts.getOrDefault(classValue, 0) + 1);
                }
            }

            // Analizar palabras en atributos string
            Map<String, Integer> wordFrequencies = new HashMap<>();
            for (int i = 0; i < totalAttributes; i++) {
                Attribute attr = data.attribute(i);
                if (attr.isString()) {
                    for (int j = 0; j < totalInstances; j++) {
                        String text = data.instance(j).stringValue(attr);
                        if (text != null && !text.isEmpty()) {
                            processText(text, wordFrequencies);
                        }
                    }
                }
            }

            // Obtener top 10 palabras
            List<Map.Entry<String, Integer>> topWords = getTopWords(wordFrequencies, 10);

            // Escribir resultados en archivo
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));

            writer.write("=== Informe de Análisis del Dataset ===\n\n");
            writer.write("Atributos:\n");
            writer.write(String.format("Nominal Attributes: %d\n", nominalCount));
            writer.write(String.format("Numeric Attributes: %d\n", numericCount));
            writer.write(String.format("String Attributes: %d\n", stringCount));
            writer.write(String.format("Boolean Attributes: %d\n", booleanCount));
            writer.write(String.format("Attributes total: %d\n\n", totalAttributes));

            writer.write("Instancias:\n");
            if (data.classIndex() >= 0) {
                for (Map.Entry<String, Integer> entry : classCounts.entrySet()) {
                    writer.write(String.format("Instancias %s: %d\n", entry.getKey(), entry.getValue()));
                }
            }
            writer.write(String.format("Instancias total: %d\n\n", totalInstances));

            writer.write("Top 10 palabras más frecuentes:\n");
            for (int i = 0; i < topWords.size(); i++) {
                Map.Entry<String, Integer> entry = topWords.get(i);
                writer.write(String.format("%d. %s: %d apariciones\n", i+1, entry.getKey(), entry.getValue()));
            }

            writer.close();
            System.out.println("Análisis completado. Resultados guardados en: " + outputFile);

        } catch (Exception e) {
            System.err.println("Error al analizar el archivo ARFF: " + e.getMessage());
            }
        }

    private static void processText(String text, Map<String, Integer> wordFrequencies) {
        // Expresión regular para encontrar palabras (incluye acentos y ñ)
        Pattern pattern = Pattern.compile("[\\p{L}]+");
        Matcher matcher = pattern.matcher(text.toLowerCase());

        while (matcher.find()) {
            String word = matcher.group();
            if (word.length() > 2) { // Ignorar palabras muy cortas
                wordFrequencies.put(word, wordFrequencies.getOrDefault(word, 0) + 1);
            }
        }
    }

    private static List<Map.Entry<String, Integer>> getTopWords(Map<String, Integer> wordFrequencies, int topN) {
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(wordFrequencies.entrySet());
        
        // Ordenar por frecuencia descendente
        Collections.sort(entries, new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> a, Map.Entry<String, Integer> b) {
                return b.getValue().compareTo(a.getValue());
            }
        });

        // Devolver las topN palabras (o menos si no hay suficientes)
        return entries.subList(0, Math.min(topN, entries.size()));
    }
}
    

