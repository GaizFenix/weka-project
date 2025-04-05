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

public class classify {
    public static void main(String[] args) {
        
        if (args.length != 3) {
            System.err.println("ERROR! Correct usage: java classify <raw_test.csv> <input_mdl> <output_results.txt");
            System.exit(1);
        }

        // Ensure the tmp directory exists
        File tmpDir = new File(".tmp");
        if (!tmpDir.exists()) {
            if (tmpDir.mkdirs()) {
                System.out.println("Temporary directory '.tmp' created successfully.");
            } else {
                System.err.println("Failed to create temporary directory '.tmp'.");
                System.exit(1); // Exit if the directory cannot be created
            }
        }

        String outCsvTempPath = ".tmp/tmp_test.csv";
        String inCsvRawFilePath = args[0];
        String inMdlPath = args[1];
        String outResultsPath = args[2];
        String finalDictionaryPath = ".tmp/final_dictionary.txt";

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

            // Load mdl
            Classifier classifier = (Classifier) SerializationHelper.read(inMdlPath);

            // Classify test instances
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outResultsPath))) {
                Map<String, Integer> counts = new HashMap<>();
                for (int i = 0; i < newTest.numInstances(); i++) {
                    double clsLabel = classifier.classifyInstance(newTest.instance(i));
                    String className = newTest.classAttribute().value((int) clsLabel);
            
                    // Write classification result to file
                    writer.write(String.format("Instance %d -> Classified as: %s%n", i, className));
            
                    // Count occurrences
                    counts.put(className, counts.getOrDefault(className, 0) + 1);
                }
            
                // Write counts to file
                writer.write("\nCLASS COUNTS:\n");
                for (Map.Entry<String, Integer> entry : counts.entrySet()) {
                    writer.write(String.format("%s: %d%n", entry.getKey(), entry.getValue()));
                }
            
                System.out.println("Classification results written to: " + outResultsPath);
            } catch (Exception e) {
                System.err.println("Error writing classification results to file: " + e.getMessage());
                e.printStackTrace();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void preprocessCSV(String inputPath, String outputPath) throws IOException {    
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