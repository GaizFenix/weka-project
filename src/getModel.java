import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;
import weka.classifiers.meta.GridSearch;

public class getModel {
    public static void main(String[] args) {

        if (args.length != 4) {
            System.err.println("ERROR! Correct usage: java getModel <input_train.arff> <input_dev.arff> <output_quality_stats.txt> <output_mdl>");
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

        // Record start time
        long startTime = System.currentTimeMillis();
        System.out.println("Program started at: " + new java.util.Date(startTime));
        System.out.println("Optimizing model, please wait...");

        String inTrainPath = args[0];
        String inDevPath = args[1];
        String outStatsPath = args[2];
        String outMdlPath = args[3];
        String inDictionaryPath = ".tmp/final_dictionary.txt";

        try {
            // Load train and dev data
            DataSource dsTrain = new DataSource(inTrainPath);
            Instances train = dsTrain.getDataSet();
            train.setClassIndex(0);

            DataSource dsDev = new DataSource(inDevPath);
            Instances dev = dsDev.getDataSet();
            dev.setClassIndex(dev.numAttributes() - 1);

            // Adequate dev to train data
            FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
            filter.setDictionaryFile(new File(inDictionaryPath));
            filter.setLowerCaseTokens(true);
            filter.setOutputWordCounts(true);
            filter.setInputFormat(dev);

            Instances newDev = Filter.useFilter(dev, filter);

            SparseToNonSparse filter2 = new SparseToNonSparse();
            filter2.setInputFormat(newDev);

            newDev = Filter.useFilter(newDev, filter2);

            // J48
            J48 j48 = new J48();
            j48.setConfidenceFactor(0.25f);
            j48.setMinNumObj(2);

            // AdaBoost with J48 as base mdl
            AdaBoostM1 ada = new AdaBoostM1();
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
            gs.buildClassifier(train);

            // Evaluate model
            Evaluation eval = new Evaluation(newDev);
            eval.evaluateModel(gs, newDev);

            // --- Result printing ---
            System.out.println("=== Best parameters found ===");
            // Returned values correspond to the optimized hyperparameters:
            // [0] -> AdaBoost weightThreshold, [1] -> AdaBoost numIterations
            // [0] -> J48 confidenceFactor, [1] -> J48 minNumObj
            Classifier bestCls = gs.getBestClassifier();

            AdaBoostM1 bestAda = (AdaBoostM1) bestCls;
            System.out.println("weightThreshold: " + bestAda.getWeightThreshold());
            System.out.println("numIterations: " + bestAda.getNumIterations());
        
            // Record end time
            long endTime = System.currentTimeMillis();
            System.out.println("Program finished at: " + new java.util.Date(endTime));
            
            // Calculate and print elapsed time
            long elapsedTime = endTime - startTime;
            System.out.println("Elapsed time (seconds): " + (elapsedTime / 1000.0));

            // Write evaluation results to the output file
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outStatsPath))) {
                writer.write("=== Dev set evaluation ===\n");
                writer.write(eval.toSummaryString() + "\n");
                writer.write(eval.toClassDetailsString() + "\n");
                writer.write(eval.toMatrixString() + "\n");
                writer.write("Elapsed time (seconds): " + (elapsedTime / 1000.0) + "\n");
                System.out.println("Evaluation results written to: " + outStatsPath);
            } catch (Exception e) {
                System.err.println("Error writing evaluation results to file: " + e.getMessage());
                e.printStackTrace();
            }

            // Train final model and save it
            Instances data = new Instances(train);
            for (Instance i : newDev) {
                data.add(i);
            }

            // Train model with data from previous tryouts
            J48 tree = new J48();
            tree.setConfidenceFactor(0.25f);
            tree.setMinNumObj(2);

            // AdaBoost with J48 as base mdl
            AdaBoostM1 mdl = new AdaBoostM1();
            mdl.setWeightThreshold(bestAda.getWeightThreshold());
            mdl.setNumIterations(bestAda.getNumIterations());
            mdl.setClassifier(j48);

            System.out.println("Building model, please wait...");

            mdl.buildClassifier(data);

            System.out.println("Model built, ending program execution...");

            // Save mdl
            SerializationHelper.write(outMdlPath, mdl);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}