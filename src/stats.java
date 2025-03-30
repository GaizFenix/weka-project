import java.io.*;
import java.util.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class stats {

    public static void main(String[] args) {
       

        String inputFile = "/workspaces/weka-project/data/raw/cleaned_PHMRC_VAI_redacted_free_text.test_blind.csv";
        String outputFile = "\\workspaces\\weka-project\\data\\raw\\resultStats.txt";

        try {
            // Load CSV file
            DataSource source = new DataSource(inputFile);
            Instances data = source.getDataSet();

            // Analyze dataset
            int numInstances = data.numInstances();
            int numAttributes = data.numAttributes();
            int classIndex = data.classIndex() == -1 ? numAttributes - 1 : data.classIndex();
            data.setClassIndex(classIndex);

            Map<String, Integer> classCounts = new HashMap<>();
            for (int i = 0; i < numInstances; i++) {
                String classValue = data.instance(i).stringValue(classIndex);
                classCounts.put(classValue, classCounts.getOrDefault(classValue, 0) + 1);
            }

            // Write results to output file
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {
                writer.write("Number of instances: " + numInstances + "\n");
                writer.write("Number of attributes: " + numAttributes + "\n");
                writer.write("Class attribute: " + data.attribute(classIndex).name() + "\n");
                writer.write("Classes and densities:\n");
                for (Map.Entry<String, Integer> entry : classCounts.entrySet()) {
                    writer.write("  " + entry.getKey() + ": " + entry.getValue() + " (" +
                            String.format("%.2f", (entry.getValue() / (double) numInstances) * 100) + "%)\n");
                }
            }

            System.out.println("Analysis complete. Results saved to " + outputFile);

        } catch (Exception e) {
            System.err.println("Error processing file: " + e.getMessage());
        }
    }
}
    

