import java.io.*;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class arff2bow {
    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Erabilpena: java arff2bow <dirty_csv> <cleaned_csv> <output_arff>");
            System.out.println("Helburuak: weka aplikazioaren funtzionamendurako datuen egokitzea burutzea.");
            System.out.println("Aurre-: - Input csv-a existitzen da eta erabilgarri dago.\n        - Weka libreria gehituta egon behar du.");
            System.out.println("Post-: - Output ARFF artxiboa inputeko CSV-aren datu berdinak ditu Wekarako formatu egokiarekin.");
            return;
        }
        //proba
        try {
            // Step 1: Clean the dirty CSV
            cleanCSV(args[0], args[1]);

            // Step 2: Load data from cleaned CSV
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(args[1])); // Cleaned CSV file
            loader.setFieldSeparator(",");  // Change if needed
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); // class attribute is last
            System.out.println(data.numInstances() + " " + data.numAttributes() + " " + data.numClasses());

            // Step 3: Save in .arff format
            ArffSaver as = new ArffSaver();
            as.setFile(new File(args[2])); // Output ARFF file
            as.setInstances(data);
            as.writeBatch();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Cleans a CSV file by wrapping all values in quotes and writes to a new file.
     * @param inputPath Path to the dirty CSV file.
     * @param outputPath Path to the cleaned CSV file.
     * @throws IOException If an I/O error occurs.
     */
    private static void cleanCSV(String inputPath, String outputPath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(inputPath));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {

            String line;
            while ((line = reader.readLine()) != null) {
                //String[] values = line.split(","); // Split by comma
                String[] values = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
                for (int i = 0; i < values.length; i++) {
                    values[i] = values[i].trim(); 
                    values[i] = "\"" + values[i].replace("\"", "\"\"") + "\"";
                    //values[i] = "\"" + values[i].trim() + "\""; // Wrap each value in quotes
                }
                writer.write(String.join(",", values)); // Join values with commas
                writer.newLine();
            }
        }
    }
}
