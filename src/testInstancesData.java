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
    
                writer.write(feature + ",?");
                writer.newLine();
            }
        }
    }
}