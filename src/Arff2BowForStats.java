import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class Arff2BowForStats {
    public static void main(String[] args) {

        if (args.length != 2) {
            System.err.println("ERROR! Correct usage: java arff2bowSimple <input_raw.csv> <output.arff>");
            System.exit(1);
        }

        String outCsvTempPath = "data/aux/temp_not_biased.csv";
        String inCsvRawFilePath = args[0];
        String outArffPath = args[1];
        String tempDictionaryPath = "data/aux/dictionary_not_biased.txt";
        String finalDictionaryPath = "data/aux/dictionary_not_biased_final.txt";

        try {
            preprocessCSV(inCsvRawFilePath, outCsvTempPath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Load the file
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(outCsvTempPath));
            loader.setFieldSeparator(",");
            loader.setNoHeaderRowPresent(false);
            loader.setStringAttributes("first");
            loader.setNominalAttributes("last");

            // Load into Instances object
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); // Class index is last

            // Filter applying (StringToWordVector and SparseToNonSparse)
            StringToWordVector filter = new StringToWordVector();
            filter.setAttributeIndices("first");
            filter.setDictionaryFileToSaveTo(new File(tempDictionaryPath));
            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setInputFormat(data);

            Instances newData = Filter.useFilter(data, filter);

            SparseToNonSparse filter2 = new SparseToNonSparse();
            filter2.setInputFormat(newData);

            newData = Filter.useFilter(newData, filter2);

            // Attribute importance
            AttributeSelection attrSel = new AttributeSelection();
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
            Ranker ranker = new Ranker();

            attrSel.setEvaluator(evaluator);
            attrSel.setSearch(ranker);
            attrSel.SelectAttributes(newData);

            // Build a map from attribute name to its InfoGain score
            Map<String, Double> importanceMap = new HashMap<>();
            double[][] rankedAttributes = attrSel.rankedAttributes();
            for (int i = 0; i < rankedAttributes.length; i++) {
                int attrIndex = (int) rankedAttributes[i][0];
                double score = rankedAttributes[i][1];
                String attrName = newData.attribute(attrIndex).name();
                importanceMap.put(attrName, score);
            }

            // Now, filter the dictionary file to only keep attributes with importance > 0.0.
            // The dictionary file is in the format: attributeName,index
            File dictFile = new File(tempDictionaryPath);
            File newDictFile = new File(finalDictionaryPath);

            BufferedReader dictReader = new BufferedReader(new FileReader(dictFile));
            BufferedWriter dictWriter = new BufferedWriter(new FileWriter(newDictFile));

            String line;
            while ((line = dictReader.readLine()) != null) {
                // Each line is like "a,2848" - split on comma
                String[] parts = line.split(",");
                if (parts.length >= 2) {
                    String attributeName = parts[0].trim();
                    // Look up the importance; if missing assume importance 0.0
                    Double imp = importanceMap.get(attributeName);
                    if (imp != null && imp > 0.0) {
                        dictWriter.write(line);
                        dictWriter.newLine();
                    }
                }
            }
            dictReader.close();
            dictWriter.close();

            System.out.println("\nFiltered dictionary saved to: " + newDictFile.getAbsolutePath());

            // Filter the dataset again using the new dictionary
            int classIndex = newData.classIndex();
            ArrayList<Integer> removeIndicesList = new ArrayList<>();

            // Iterate over the importance map and collect indices of attributes with 0.0 importance.
            // (Skip the class attribute.)
            for (Map.Entry<String, Double> entry : importanceMap.entrySet()) {
                String attrName = entry.getKey();
                double importance = entry.getValue();
                int idx = newData.attribute(attrName).index();
                if (idx == classIndex) {
                    continue; // never remove the class attribute
                }
                if (importance == 0.0) {
                    removeIndicesList.add(idx);
                }
            }

            // If there are attributes to remove, build the Remove filter.
            if (!removeIndicesList.isEmpty()) {
                // Sort the indices in ascending order.
                Collections.sort(removeIndicesList);
                
                // Build a comma-separated list of 1-based indices (Remove filter uses 1-based indexing)
                StringBuilder removeIndices = new StringBuilder();
                for (int idx : removeIndicesList) {
                    removeIndices.append(idx + 1).append(",");
                }
                // Remove trailing comma (after the last index)
                removeIndices.deleteCharAt(removeIndices.length() - 1);
                
                // Create and apply the Remove filter
                Remove removeFilter = new Remove();
                removeFilter.setAttributeIndices(removeIndices.toString());
                removeFilter.setInputFormat(newData);
                Instances filteredData = Filter.useFilter(newData, removeFilter);
                
                // Overwrite newData with the filtered version
                newData = filteredData;
            } else {
                System.out.println("No attributes with 0.0 importance were found to remove.");
            }

            // Save the processed data as ARFF
            ArffSaver as = new ArffSaver();
            as.setFile(new File(outArffPath));
            as.setInstances(newData);
            as.writeBatch();

            System.out.println("Processed data (BoW) ARFF saved to: " + outArffPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void preprocessCSV(String inputPath, String outputPath) throws IOException {
        Map<String, String> diseaseMap = getDiseaseChapterMap();

        try (BufferedReader reader = new BufferedReader(new FileReader(inputPath));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {

            String line;

            // Write header
            if ((line = reader.readLine()) != null) {
                writer.write(line.trim());
                writer.newLine();
            }

            // Process data lines
            while ((line = reader.readLine()) != null) {
                System.out.println("DEBUG: Processing line: " + line);

                // Split the line into columns, handling commas inside quotes
                String[] values = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
                System.out.println("DEBUG: Number of columns: " + values.length);

                // Validate the number of columns
                if (values.length < 7) {
                    System.err.println("WARNING: Line has fewer than 7 columns. Skipping: " + line);
                    continue;
                } else if (values.length > 7) {
                    System.err.println("WARNING: Line has more than 7 columns. Attempting to fix: " + line);
                    values = fixExtraColumns(values);
                }

                // Normalize class
                String rawClass = values[6].trim().replaceAll("^['\"]|['\"]$", "").toLowerCase();
                String mappedClass = diseaseMap.getOrDefault(rawClass, rawClass); // Keep unmapped class if not found

                // Replace only the class column with the mapped class
                values[6] = mappedClass;

                // Write the processed line to the output file
                writer.write(String.join(",", values));
                writer.newLine();
            }
        }
    }

    private static Map<String, String> getDiseaseChapterMap() {
        Map<String, String> map = new HashMap<>();

        // Lowercase keys ONLY
        map.put("diarrhea/dysentery", "Certain infectious and Parasitic Diseases");
        map.put("other infectious diseases", "Certain infectious and Parasitic Diseases");
        map.put("aids", "Certain infectious and Parasitic Diseases");
        map.put("sepsis", "Certain infectious and Parasitic Diseases");
        map.put("meningitis", "Certain infectious and Parasitic Diseases");
        map.put("meningitis/sepsis", "Certain infectious and Parasitic Diseases");
        map.put("malaria", "Certain infectious and Parasitic Diseases");
        map.put("encephalitis", "Certain infectious and Parasitic Diseases");
        map.put("measles", "Certain infectious and Parasitic Diseases");
        map.put("hemorrhagic fever", "Certain infectious and Parasitic Diseases");
        map.put("tb", "Certain infectious and Parasitic Diseases");
        map.put("other infectious diseases", "Certain infectious and Parasitic Diseases");

        map.put("leukemia/lymphomas", "Neoplasms");
        map.put("colorectal cancer", "Neoplasms");
        map.put("lung cancer", "Neoplasms");
        map.put("cervical cancer", "Neoplasms");
        map.put("breast cancer", "Neoplasms");
        map.put("stomach cancer", "Neoplasms");
        map.put("prostate cancer", "Neoplasms");
        map.put("esophageal cancer", "Neoplasms");
        map.put("other cancers", "Neoplasms");

        map.put("diabetes", "Endocrine or Nutritional and Metabolic Diseases");
        map.put("other non-communicable diseases", "Endocrine or Nutritional and Metabolic Diseases");

        map.put("epilepsy", "Diseases of the Nervous System");

        map.put("stroke", "Diseases of the circulatory system");
        map.put("acute myocardial infarction", "Diseases of the circulatory system");
        map.put("other cardiovascular diseases", "Diseases of the circulatory system");

        map.put("pneumonia", "Diseases of Respiratory System");
        map.put("asthma", "Diseases of Respiratory System");
        map.put("copd", "Diseases of Respiratory System");

        map.put("cirrhosis", "Diseases of the Digestive System");
        map.put("other digestive diseases", "Diseases of the Digestive System");

        map.put("renal failure", "Diseases of the Genitourinary System");

        map.put("preterm delivery", "Pregnancy or childbirth and the puerperium");
        map.put("stillbirth", "Pregnancy or childbirth and the puerperium");
        map.put("maternal", "Pregnancy or childbirth and the puerperium");
        map.put("birth asphyxia", "Pregnancy or childbirth and the puerperium");
        map.put("other defined causes of child deaths", "Pregnancy or childbirth and the puerperium");

        map.put("congenital malformations", "Congenital Malformations");
        map.put("congenital malformation", "Congenital Malformations");

        map.put("bite of venomous animal", "Injury or Poisoning and External Causes");
        map.put("poisonings", "Injury or Poisoning and External Causes");

        map.put("road traffic", "External Causes of Morbidity and Mortality");
        map.put("falls", "External Causes of Morbidity and Mortality");
        map.put("homicide", "External Causes of Morbidity and Mortality");
        map.put("fires", "External Causes of Morbidity and Mortality");
        map.put("drowning", "External Causes of Morbidity and Mortality");
        map.put("suicide", "External Causes of Morbidity and Mortality");
        map.put("violent death", "External Causes of Morbidity and Mortality");
        map.put("other injuries", "External Causes of Morbidity and Mortality");

        return map;
    }

    private static String[] fixExtraColumns(String[] values) {
        System.out.println("WARNING: Fixing extra columns in line: " + String.join(",", values));
        // Si hay más de 7 columnas, combinamos las columnas adicionales en la penúltima columna
        if (values.length > 7) {
            String[] fixedValues = new String[7];
            System.arraycopy(values, 0, fixedValues, 0, 5); // Copiar las primeras 5 columnas

            // Combinar las columnas adicionales en la penúltima columna
            StringBuilder combinedColumn = new StringBuilder(values[5]);
            for (int i = 6; i < values.length - 1; i++) {
                combinedColumn.append(",").append(values[i]);
            }
            fixedValues[5] = combinedColumn.toString();

            // Mantener la última columna (clase) intacta
            fixedValues[6] = values[values.length - 1];

            return fixedValues;
        }

        // Si no hay más de 7 columnas, devolver los valores originales
        return values;
    }
}