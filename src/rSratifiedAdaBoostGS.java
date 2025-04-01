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
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.GridSearch;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
//import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class rSratifiedAdaBoostGS {
    public static void main(String[] args) throws IOException {

        if (args.length != 3) {
            System.err.println("ERROR! Correct usage: java rStratifiedAdaBoostGS <input_raw.csv> <output_train.arff> <output_dev.arff>");
            System.exit(1);
        }

        int bestSeed = -1;
        double bestAccuracy = 0.0;
        String outCsvTempPath = "data/aux/temp_not_biased.csv";
        int[] seeds = {5, 162, 182, 197, 267, 345, 378, 388, 497, 625, 638, 668, 685, 704, 756, 766, 770, 812, 937, 956};
        double[] everyAcc = new double[seeds.length];
        double[] everyPrecision = new double[seeds.length];
        double[] everyRecall = new double[seeds.length];
        double[] everyF1 = new double[seeds.length];
        double[] everyROC = new double[seeds.length];
        double[] everyPRC = new double[seeds.length];
        Integer itMax = seeds.length;
        String inCsvRawFilePath = args[0];
        //String outTrainArffPath = args[1];
        //String outDevArffPath = args[2];
        String tempDictionaryPath = "data/aux/dictionary_not_biased.txt";
        String finalDictionaryPath = "data/aux/dictionary_not_biased_final.txt";
        

        try {
            preprocessCSV(inCsvRawFilePath, outCsvTempPath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Load CSV file
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(outCsvTempPath));
        loader.setFieldSeparator(",");
        loader.setNoHeaderRowPresent(false);
        loader.setStringAttributes("first");
        loader.setNominalAttributes("last");

        // Load into Instances object
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1); // Class index is last

        // Load the file
        try {
            for (int it = 1; it <= itMax; it ++) {
                int seed = seeds[it - 1];
                System.out.println("\nProcessing with random seed: " + seed);

                // Resample to create train and dev sets
                Resample r = new Resample();
                r.setRandomSeed(seed);
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

                //System.out.println("\nTrain set size: " + train.size());
                //System.out.println("Dev set size: " + dev.size());

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
                File dictFile = new File(tempDictionaryPath);
                File newDictFile = new File(finalDictionaryPath);

                BufferedReader dictReader = new BufferedReader(new FileReader(dictFile));
                BufferedWriter dictWriter = new BufferedWriter(new FileWriter(newDictFile));

                String line;
                while ((line = dictReader.readLine()) != null) {
                    String[] parts = line.split(",");
                    if (parts.length >= 2) {
                    String attributeName = parts[0].trim();
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

                for (Map.Entry<String, Double> entry : importanceMap.entrySet()) {
                    String attrName = entry.getKey();
                    double importance = entry.getValue();
                    int idx = newTrain.attribute(attrName).index();
                    if (idx == classIndex) {
                    continue;
                    }
                    if (importance == 0.0) {
                    removeIndicesList.add(idx);
                    }
                }

                if (!removeIndicesList.isEmpty()) {
                    Collections.sort(removeIndicesList);

                    StringBuilder removeIndices = new StringBuilder();
                    for (int idx : removeIndicesList) {
                    removeIndices.append(idx + 1).append(",");
                    }
                    removeIndices.deleteCharAt(removeIndices.length() - 1);

                    Remove removeFilter = new Remove();
                    removeFilter.setAttributeIndices(removeIndices.toString());
                    removeFilter.setInputFormat(newTrain);
                    Instances filteredTrain = Filter.useFilter(newTrain, removeFilter);

                    newTrain = filteredTrain;
                } else {
                    System.out.println("No attributes with 0.0 importance were found to remove.");
                }

                // Save train BoW as ARFF
                /*ArffSaver as = new ArffSaver();
                as.setFile(new File(outTrainArffPath.replace(".arff", "_seed" + seed + ".arff")));
                as.setInstances(newTrain);
                as.writeBatch();

                System.out.println("Train (BoW) ARFF saved to: " + outTrainArffPath.replace(".arff", "_seed" + seed + ".arff"));

                // Save dev raw as ARFF
                /*as.setFile(new File(outDevArffPath.replace(".arff", "_seed" + seed + ".arff")));
                as.setInstances(dev);
                as.writeBatch();

                System.out.println("Dev (raw) ARFF saved to: " + outDevArffPath.replace(".arff", "_seed" + seed + ".arff"));*/

                // Record start time
                long startTime = System.currentTimeMillis();
                System.out.println("Program started at: " + new java.util.Date(startTime));

                newTrain.setClassIndex(0); // Set class index to last attribute
                dev.setClassIndex(dev.numAttributes() - 1); // Set class index to last attribute
            
                // Adequate dev to train data
                FixedDictionaryStringToWordVector filter3 = new FixedDictionaryStringToWordVector();
                filter3.setDictionaryFile(newDictFile);
                // filter.setIDFTransform(true);
                // filter.setTFTransform(true);
                filter3.setLowerCaseTokens(true);
                filter3.setOutputWordCounts(true);
                filter3.setInputFormat(dev);

                Instances newDev = Filter.useFilter(dev, filter3);

                SparseToNonSparse filter4 = new SparseToNonSparse();
                filter4.setInputFormat(newDev);

                newDev = Filter.useFilter(newDev, filter4);

                // J48
                J48 j48 = new J48();
                j48.setConfidenceFactor(0.25f);
                j48.setMinNumObj(2);

                // AdaBoost with J48 as base mdl
                AdaBoostM1 ada = new AdaBoostM1();
                // ada.setWeightThreshold(80);
                // ada.setNumIterations(15);
                ada.setClassifier(j48);

                // GridSearch
                GridSearch gs = new GridSearch();
                gs.setClassifier(ada);
                gs.setEvaluation(new SelectedTag(GridSearch.EVALUATION_ACC, GridSearch.TAGS_EVALUATION));

                // Set X property
                gs.setXProperty("weightThreshold");
                gs.setXMin(50);
                gs.setXMax(100);
                gs.setXStep(10);
                gs.setXExpression("I");

                // Set Y property
                gs.setYProperty("numIterations");
                gs.setYMin(10);
                gs.setYMax(15);
                gs.setYStep(5);
                gs.setYExpression("I");

                // Paralelize execution
                gs.setNumExecutionSlots(Runtime.getRuntime().availableProcessors());

                // Build model
                gs.buildClassifier(newTrain);

                // Evaluate model
                Evaluation eval = new Evaluation(newDev);
                eval.evaluateModel(gs, newDev);

                // --- Result printing ---
                System.out.println("=== Best parameters found ===");
                // Returned values correspond to the optimized hyperparameters:
                // [0] -> AdaBoost weightThreshold, [1] -> AdaBoost numIterations
                // [0] -> J48 confidenceFactor, [1] -> J48 minNumObj
                Classifier bestCls = gs.getBestClassifier();

                if (bestCls instanceof AdaBoostM1) {
                    AdaBoostM1 bestAda = (AdaBoostM1) bestCls;
                    System.out.println("weightThreshold: " + bestAda.getWeightThreshold());
                    System.out.println("numIterations: " + bestAda.getNumIterations());
                } else if (bestCls instanceof J48) {
                    J48 bestJ48 = (J48) bestCls;
                    System.out.println("confidenceFactor: " + bestJ48.getConfidenceFactor());
                    System.out.println("minNumObj: " + bestJ48.getMinNumObj());
                }
            
                System.out.println("\n=== Dev set evaluation ===");
                System.out.println(eval.toSummaryString());
                System.out.println(eval.toClassDetailsString());
                System.out.println(eval.toMatrixString());
                //System.out.println(eval.toClassDetailsString());
                //System.out.println(eval.toMatrixString());

                // Record end time
                long endTime = System.currentTimeMillis();
                System.out.println("Program finished at: " + new java.util.Date(endTime));
                
                // Calculate and print elapsed time
                long elapsedTime = endTime - startTime;
                double accuracy = eval.pctCorrect();
                System.out.println("Elapsed time (seconds): " + (elapsedTime / 1000.0));
                System.out.println("Accuracy for seed " + seed + ": " + accuracy);

                
                // Save acc for average
                everyAcc[it - 1] = accuracy;
                everyPrecision[it - 1] = eval.weightedPrecision();
                everyRecall[it - 1] = eval.weightedRecall();
                everyF1[it - 1] = eval.weightedFMeasure();
                everyROC[it - 1] = eval.weightedAreaUnderROC();
                everyPRC[it - 1] = eval.weightedAreaUnderPRC();

                // Save best acc
                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    bestSeed = seed;
                }

            }

            // Calculate mean and standard deviation
            double meanAcc = calculateMean(everyAcc);
            // Calculate mean precision
            double meanPrecision = calculateMean(everyPrecision);
            // Calculate mean recall
            double meanRecall = calculateMean(everyRecall);
            // Calculate mean F1
            double meanF1 = calculateMean(everyF1);
            // Calculate mean ROC
            double meanROC = calculateMean(everyROC);
            // Calculate mean PRC
            double meanPRC = calculateMean(everyPRC);
            // Print results
            System.out.println("\n=== General Metrics ===");
            System.out.println("Mean accuracy: " + meanAcc);
            System.out.println("Standard ACC deviation: " + calculateStdDev(everyAcc, meanAcc));
            System.out.println("Mean precision: " + meanPrecision);
            System.out.println("Standard precision deviation: " + calculateStdDev(everyPrecision, meanPrecision));
            System.out.println("Mean recall: " + meanRecall);
            System.out.println("Standard recall deviation: " + calculateStdDev(everyRecall, meanRecall));
            System.out.println("Mean F1: " + meanF1);
            System.out.println("Standard F1 deviation: " + calculateStdDev(everyF1, meanF1));
            System.out.println("Mean ROC: " + meanROC);
            System.out.println("Standard ROC deviation: " + calculateStdDev(everyROC, meanROC));
            System.out.println("Mean PRC: " + meanPRC);
            System.out.println("Standard PRC deviation: " + calculateStdDev(everyPRC, meanPRC));
            System.out.println("Best accuracy: " + bestAccuracy);
            System.out.println("Best seed: " + bestSeed);
        
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
    
    //Desviación estándar
    private static double calculateStdDev(double[] values, double mean) {
        double variance = 0;
        for (double value : values) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= values.length;
        return Math.sqrt(variance);
    }

    private static double calculateMean(double[] values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
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