import weka.core.Instances;
import weka.core.converters.ArffLoader;
import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

public class ARFFAnalyzer {

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Usage: java ARFFAnalyzer <input-arff-file> <output-txt-file>");
            return;
        }

        String arffFilePath = args[0];
        String outputFilePath = args[1];

        // Load ARFF file
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(arffFilePath));
        Instances data = loader.getDataSet();

        // Analyze statistics
        int numInstances = data.numInstances();
        int numAttributes = data.numAttributes();
        int numNominalAttributes = 0;
        int numNumericAttributes = 0;

        for (int i = 0; i < numAttributes; i++) {
            if (data.attribute(i).isNominal()) {
                numNominalAttributes++;
            } else if (data.attribute(i).isNumeric()) {
                numNumericAttributes++;
            }
        }

        // Count word frequencies
        Map<String, Integer> wordCounts = new HashMap<>();
        Pattern wordPattern = Pattern.compile("^[a-zA-Z]+$"); // Only count alphabetic words

        for (int i = 0; i < data.numInstances(); i++) {
            String instanceString = data.instance(i).toString();
            String[] words = instanceString.split("\\s+|,"); // Split by whitespace or commas
            for (String word : words) {
                word = word.toLowerCase().trim(); // Normalize to lowercase
                if (wordPattern.matcher(word).matches()) { // Only count valid words
                    wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
                }
            }
        }

        // Get top 25 words
        List<Map.Entry<String, Integer>> sortedWordCounts = new ArrayList<>(wordCounts.entrySet());
        sortedWordCounts.sort((a, b) -> b.getValue().compareTo(a.getValue()));

        List<Map.Entry<String, Integer>> top25Words = sortedWordCounts.subList(0, Math.min(25, sortedWordCounts.size()));

        // Count values for attr_sex
        Map<String, Integer> sexCounts = new HashMap<>();
        int sexIndex = data.attribute("attr_sex").index();
        for (int i = 0; i < data.numInstances(); i++) {
            String sexValue = data.instance(i).stringValue(sexIndex);
            sexCounts.put(sexValue, sexCounts.getOrDefault(sexValue, 0) + 1);
        }

        // Count values for attr_site
        Map<String, Integer> siteCounts = new HashMap<>();
        int siteIndex = data.attribute("attr_site").index();
        for (int i = 0; i < data.numInstances(); i++) {
            String siteValue = data.instance(i).stringValue(siteIndex);
            siteCounts.put(siteValue, siteCounts.getOrDefault(siteValue, 0) + 1);
        }

        // Count values for attr_age
        Map<Integer, Integer> ageCounts = new HashMap<>();
        int ageIndex = data.attribute("attr_age").index();
        for (int i = 0; i < data.numInstances(); i++) {
            int ageValue = (int) data.instance(i).value(ageIndex); // Convert age to int
            ageCounts.put(ageValue, ageCounts.getOrDefault(ageValue, 0) + 1);
        }

        // Count occurrences of #
        int hashIndex = data.attribute("#").index();
        int hashCount = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            hashCount += data.instance(i).value(hashIndex);
        }

        // Write results to output file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
            writer.write("Number of instances: " + numInstances + "\n");
            writer.write("Number of attributes: " + numAttributes + "\n");
            writer.write("Number of nominal attributes: " + numNominalAttributes + "\n");
            writer.write("Number of numeric attributes: " + numNumericAttributes + "\n");
            writer.write("Top 25 words:\n");
            for (Map.Entry<String, Integer> entry : top25Words) {
                writer.write(entry.getKey() + ": " + entry.getValue() + "\n");
            }

            writer.write("\nCounts for attr_sex:\n");
            for (Map.Entry<String, Integer> entry : sexCounts.entrySet()) {
                writer.write(entry.getKey() + ": " + entry.getValue() + "\n");
            }

            writer.write("\nCounts for attr_site:\n");
            for (Map.Entry<String, Integer> entry : siteCounts.entrySet()) {
                writer.write(entry.getKey() + ": " + entry.getValue() + "\n");
            }

            writer.write("\nCounts for attr_age (Table):\n");
            writer.write(String.format("%-10s | %-10s\n", "Age", "Count"));
            writer.write("----------------------------\n");
            for (Map.Entry<Integer, Integer> entry : ageCounts.entrySet()) {
                writer.write(String.format("%-10d | %-10d\n", entry.getKey(), entry.getValue()));
            }

            writer.write("\nTotal count for #: " + hashCount + "\n");
        } catch (IOException e) {
            System.err.println("Error writing to output file: " + e.getMessage());
        }

        System.out.println("Analysis complete. Results written to " + outputFilePath);
    }
}