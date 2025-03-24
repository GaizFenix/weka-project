import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class arff2bowNotBiased {
    public static void main(String[] args) {

        if (args.length != 3) {
            System.err.println("ERROR! Correct usage: java arff2bowSimple <input_raw.csv> <output_train.arff> <output_dev.arff>");
            System.exit(1);
        }

        String outCsvTempPath = "data/aux/temp_not_biased.csv";
        String inCsvRawFilePath = args[0];
        String outTrainArffPath = args[1];
        String outDevArffPath = args[2];

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

            // Resample to create train and dev sets
            Resample r = new Resample();
            r.setRandomSeed(18);
            r.setSampleSizePercent(70); // 70% for training
            r.setNoReplacement(true); // Without replacement
            r.setInvertSelection(false); // Select training set
            r.setInputFormat(data);

            // Create training set (70%)
            Instances train = Filter.useFilter(data, r);

            // Create dev set (30%)
            r.setInvertSelection(true); // Select dev set
            r.setInputFormat(data);
            Instances dev = Filter.useFilter(data, r);

            // Filter applying (StringToWordVector and SparseToNonSparse)
            StringToWordVector filter = new StringToWordVector();
            filter.setAttributeIndices("first");
            filter.setDictionaryFileToSaveTo(new File("data/aux/dictionary_not_biased.txt"));
            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setInputFormat(train);

            Instances newTrain = Filter.useFilter(train, filter);

            SparseToNonSparse filter2 = new SparseToNonSparse();
            filter2.setInputFormat(newTrain);

            newTrain = Filter.useFilter(newTrain, filter2);

            // Save train BoW as ARFF
            ArffSaver as = new ArffSaver();
            as.setFile(new File(outTrainArffPath));
            as.setInstances(newTrain);
            as.writeBatch();

            System.out.println("Train (BoW) ARFF saved to: " + outTrainArffPath);

            // Save dev raw as ARFF
            as.setFile(new File(outDevArffPath));
            as.setInstances(dev);
            as.writeBatch();

            System.out.println("\nDev (raw) ARFF saved to: " + outDevArffPath);

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
    
                String rawClass = values[6].trim();
                rawClass = rawClass.replaceAll("^['\"]|['\"]$", "").trim().toLowerCase();
    
                String mappedClass = diseaseMap.get(rawClass);
                if (mappedClass == null) {
                    System.err.println("WARNING: Unmapped class '" + rawClass + "' — skipping line.");
                    continue;
                }
    
                writer.write(feature + "," + mappedClass);
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