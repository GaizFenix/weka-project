import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class testInstancesData {
    public static void main(String[] args) {
        
        if (args.length != 2) {
            System.err.println("ERROR! Correct usage: java testInstancesData <input_test.csv> <input_mdl>");
            System.exit(1);
        }

        String outCsvTempPath = "data/aux/temp_for_test.csv";
        String inCsvRawFilePath = args[0];
        String inMdlPath = args[1];
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

            // Code for when text and class are only attrs used
            loader.setStringAttributes("first");
            loader.setNominalAttributes("last");

            // Load into Instances object
            Instances test = loader.getDataSet();
            int classIndex = test.numAttributes() - 1; // Last attribute is class
            test.setClassIndex(-1); // Class index unset

            // Delete class attribute
            test.deleteAttributeAt(classIndex);

            // Create new class attribute with train values
            ArrayList<String> classValues = new ArrayList<>(Arrays.asList(
                "Certain infectious and Parasitic Diseases",
                "Neoplasms",
                "Endocrine or Nutritional and Metabolic Diseases",
                    "Diseases of the Nervous System",
                "Diseases of the circulatory system",
                "Diseases of Respiratory System",
                "Diseases of the Digestive System",
                "Diseases of the Genitourinary System",
                "Pregnancy or childbirth and the puerperium",
                "Congenital Malformations",
                "Injury or Poisoning and External Causes",
                "External Causes of Morbidity and Mortality"
            ));

            Attribute classAttr = new Attribute("class", classValues);

            // Add the class attribute to the dataset
            test.insertAttributeAt(classAttr, test.numAttributes());
            test.setClassIndex(test.numAttributes() - 1); // Set the class index to the new attribute

            // Set missing class values for all instances
            for (int i = 0; i < test.numInstances(); i++) {
                test.instance(i).setMissing(test.classIndex());
            }

            // Filter applying (StringToWordVector and SparseToNonSparse)
            FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
            filter.setDictionaryFile(new File(finalDictionaryPath));
            filter.setAttributeIndices("first");
            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setInputFormat(test);

            Instances newTest = Filter.useFilter(test, filter);

            SparseToNonSparse filter2 = new SparseToNonSparse();
            filter2.setInputFormat(newTest);

            newTest = Filter.useFilter(newTest, filter2);

            System.out.println("Test set attrs:");
            for (int i = 0; i < newTest.numAttributes(); i++) {
                System.out.printf("%d: %s%n", i, newTest.attribute(i).name());
            }

            // Load mdl
            Classifier classifier = (Classifier) SerializationHelper.read(inMdlPath);

            // Classify test instances
            Map<String,Integer> counts = new HashMap<>();
            for (int i = 0; i < newTest.numInstances(); i++) {
                double clsLabel = classifier.classifyInstance(newTest.instance(i));
                String className = newTest.classAttribute().value((int) clsLabel);
                // System.out.printf("Instance %d Classified as: %.2f%n", i, clsLabel);

                // Count occurrences
                counts.put(className, counts.getOrDefault(className, 0) + 1);
            }

            // Print counts
            System.out.println("Class counts:");
            for (Map.Entry<String, Integer> entry : counts.entrySet()) {
                System.out.printf("%s: %d%n", entry.getKey(), entry.getValue());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void preprocessCSV(String inputPath, String outputPath) throws IOException {
        // Map<String, String> diseaseMap = getDiseaseChapterMap();
    
        try (BufferedReader reader = new BufferedReader(new FileReader(inputPath));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
    
            String line;
    
            // Write header - FOR USING ONLY TEXT AND CLASS
            if ((line = reader.readLine()) != null) {
                String[] headers = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
                if (headers.length >= 7) {
                    writer.write(headers[5].trim() + "," + headers[6].trim());
                    writer.newLine();
                } else {
                    System.err.println("Header row has fewer than 7 columns — skipping.");
                    return;
                }
            }
    
            // Process data lines
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
    
                if (values.length < 7) continue;
    
                String feature = "\"" + values[5].trim().replace("\"", "\"\"") + "\"";

                /*
                String rawClass = values[6].trim();
                rawClass = rawClass.replaceAll("^['\"]|['\"]$", "").trim().toLowerCase();
    
                String mappedClass = diseaseMap.get(rawClass);
                if (mappedClass == null) {
                    System.err.println("WARNING: Unmapped class '" + rawClass + "' — skipping line.");
                    continue;
                }

                System.out.println("Mapped class: " + mappedClass);
                */
    
                writer.write(feature + ",?");
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
}