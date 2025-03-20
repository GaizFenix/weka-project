import java.io.*;

public class CSVCleaner {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java CSVCleaner <input_csv> <output_csv>");
            return;
        }

        String inputFile = args[0];
        String outputFile = args[1];

        try (BufferedReader reader = new BufferedReader(new FileReader(inputFile));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {

            String line;
            while ((line = reader.readLine()) != null) {
                // Clean the line
                String cleanedLine = fixCSVLine(line);
                writer.write(cleanedLine);
                writer.newLine();
            }

            System.out.println("Fixed CSV saved to: " + outputFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String fixCSVLine(String line) {
        StringBuilder fixedLine = new StringBuilder();
        boolean inQuotes = false;
        char prevChar = '\0';

        for (char c : line.toCharArray()) {
            if (c == '"') {
                inQuotes = !inQuotes; // Toggle quote state
            } else if (c == ',' && !inQuotes) {
                fixedLine.append("\",\""); // Ensure commas are correctly enclosed
                continue;
            }

            // Add current character to the fixed line
            fixedLine.append(c);
            prevChar = c;
        }

        // Ensure the whole line is wrapped in double quotes if needed
        if (!line.startsWith("\"")) {
            fixedLine.insert(0, "\"");
        }
        if (!line.endsWith("\"")) {
            fixedLine.append("\"");
        }

        return fixedLine.toString();
    }
}
