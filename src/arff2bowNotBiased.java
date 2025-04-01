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
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
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
        String tempDictionaryPath = "data/aux/dictionary_not_biased.txt";
        String finalDictionaryPath = "data/aux/dictionary_not_biased_final.txt";
        // String attrIntArray = "data/aux/attr_int_array.txt";

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
            // loader.setNumericAttributes("1, 3");
            // loader.setNominalAttributes("2, 4, 5, 7");
            // loader.setStringAttributes("6");

            // Code for when text and class are only attrs used
            loader.setStringAttributes("first");
            loader.setNominalAttributes("last");

            // Load into Instances object
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); // Class index is last

            // Rename attributes
            // data.renameAttribute(0, "attr_newid");
            // data.renameAttribute(1, "attr_module");
            // data.renameAttribute(2, "attr_age");
            // data.renameAttribute(3, "attr_sex");
            // data.renameAttribute(4, "attr_site");
            // data.renameAttribute(5, "open_response");
            // data.renameAttribute(6, "gs_text34");

            // Resample to create train and dev sets
            Resample r = new Resample();
            r.setRandomSeed(18);
            r.setSampleSizePercent(75); // 75% for training
            r.setNoReplacement(false); // With replacement
            r.setInvertSelection(false); // Select training set
            r.setBiasToUniformClass(0.3);
            r.setInputFormat(data);

            // Create training set (70%)
            Instances train = Filter.useFilter(data, r);

            // Create dev set (30%)
            r.setInvertSelection(true); // Select dev set
            r.setNoReplacement(true); // Without replacement
            r.setBiasToUniformClass(0.0);
            r.setInputFormat(data);
            Instances dev = Filter.useFilter(data, r);

            System.out.println("\nTrain set size: " + train.size());
            System.out.println("Dev set size: " + dev.size());

            // Filter applying (StringToWordVector and SparseToNonSparse)
            StringToWordVector filter = new StringToWordVector();
            filter.setAttributeIndices("first");
            filter.setDictionaryFileToSaveTo(new File(tempDictionaryPath));
            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setInputFormat(train);

            Instances newTrain = Filter.useFilter(train, filter);

            SparseToNonSparse filter2 = new SparseToNonSparse();
            filter2.setInputFormat(newTrain);

            newTrain = Filter.useFilter(newTrain, filter2);

            // Attribute importance
            // File outputFile = new File("data/aux/attr_importance.txt");
            // BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));

            /*
            AttributeSelection attrSel = new AttributeSelection();
            CfsSubsetEval evaluator = new CfsSubsetEval();
            BestFirst ranker = new BestFirst();

            attrSel.setEvaluator(evaluator);
            attrSel.setSearch(ranker);
            attrSel.SelectAttributes(newTrain);

            int[] attrIndexes = attrSel.selectedAttributes();
            System.out.println(attrSel.selectedAttributes().length + " attributes selected.");
            */

            // ATTRIBUTE SELECTION USING INFOGAIN
            AttributeSelection attrSel = new AttributeSelection();
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
            Ranker ranker = new Ranker();

            attrSel.setEvaluator(evaluator);
            attrSel.setSearch(ranker);
            attrSel.SelectAttributes(newTrain);

            // Build a map from attribute name to its InfoGain score
            Map<String, Double> importanceMap = new HashMap<>();
            double[][] rankedAttributes = attrSel.rankedAttributes();
            for (int i = 0; i < rankedAttributes.length; i++) {
                int attrIndex = (int) rankedAttributes[i][0];
                double score = rankedAttributes[i][1];
                String attrName = newTrain.attribute(attrIndex).name();
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

            // Filter the train set again using the new dictionary
            int classIndex = newTrain.classIndex();
            ArrayList<Integer> removeIndicesList = new ArrayList<>();

            // Iterate over the importance map and collect indices of attributes with 0.0 importance.
            // (Skip the class attribute.)
            for (Map.Entry<String, Double> entry : importanceMap.entrySet()) {
                String attrName = entry.getKey();
                double importance = entry.getValue();
                int idx = newTrain.attribute(attrName).index();
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
                removeFilter.setInputFormat(newTrain);
                Instances filteredTrain = Filter.useFilter(newTrain, removeFilter);
                
                // Overwrite newTrain with the filtered version
                newTrain = filteredTrain;
            } else {
                System.out.println("No attributes with 0.0 importance were found to remove.");
            }

            /* Create writer for the final dictionary
            BufferedWriter dictWriter = new BufferedWriter(new FileWriter("data/aux/cfs_dictionary.txt"));

            // The filtered Instances (after attribute selection)
            newTrain = attrSel.reduceDimensionality(newTrain);

            for (int i = 0; i < newTrain.numAttributes(); i++) {
                if (i == newTrain.classIndex()) continue; // skip class attribute
                String word = newTrain.attribute(i).name();
                dictWriter.write(word + "," + i); // or use a running index if preferred
                dictWriter.newLine();
            }

            dictWriter.close();
            */


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

            System.out.println("Dev (raw) ARFF saved to: " + outDevArffPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void preprocessCSV(String inputPath, String outputPath) throws IOException {
        Map<String, String> diseaseMap = getDiseaseChapterMap();
    
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

            /* Write header - WHEN USING ALL ATTRIBUTES
            if ((line = reader.readLine()) != null) {
                writer.write(line.trim());
                writer.newLine();
            }
            */
    
            // Process data lines
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
    
                if (values.length < 7) continue;
    
                String feature = "\"" + values[5].trim().replace("\"", "\"\"") + "\"";
                // values[5] = "\"" + values[5].trim().replace("\"", "\"\"") + "\"";
    
                String rawClass = values[6].trim();
                rawClass = rawClass.replaceAll("^['\"]|['\"]$", "").trim().toLowerCase();
    
                String mappedClass = diseaseMap.get(rawClass);
                if (mappedClass == null) {
                    System.err.println("WARNING: Unmapped class '" + rawClass + "' — skipping line.");
                    continue;
                }

                // values[6] = "\"" + mappedClass.replace("\"", "\"\"") + "\""; // quote and escape mapped class
    
                writer.write(feature + "," + mappedClass);
                // writer.write(String.join(",", values));
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