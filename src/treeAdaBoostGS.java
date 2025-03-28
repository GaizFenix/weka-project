import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;
import weka.classifiers.meta.GridSearch;

public class treeAdaBoostGS {
    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("ERROR! Correct usage: java treeAdaBoost <input_train.arff> <input_dev.arff> <input_dictionary.txt>");
            System.exit(1);
        }

        // Record start time
        long startTime = System.currentTimeMillis();
        System.out.println("Program started at: " + new java.util.Date(startTime));

        String inTrainPath = args[0];
        String inDevPath = args[1];
        String inDictionaryPath = args[2];
        // String outStatsPath = args[3];

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
            // filter.setIDFTransform(true);
            // filter.setTFTransform(true);
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
            // ada.setWeightThreshold(80);
            // ada.setNumIterations(15);
            ada.setClassifier(j48);

            // GridSearch
            GridSearch gs = new GridSearch();
            gs.setClassifier(ada);
            gs.setEvaluation(new SelectedTag(GridSearch.EVALUATION_ACC, GridSearch.TAGS_EVALUATION));

            // GridSearch for J48
            /*
            gs.setXProperty("confidenceFactor");
            gs.setXMin(0.1);
            gs.setXMax(0.5);
            gs.setXStep(0.1);
            gs.setXExpression("I");

            gs.setYProperty("minNumObj");
            gs.setYMin(2);
            gs.setYMax(10);
            gs.setYStep(2);
            gs.setYExpression("I");
            */

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

            // Record end time
            long endTime = System.currentTimeMillis();
            System.out.println("Program finished at: " + new java.util.Date(endTime));
            
            // Calculate and print elapsed time
            long elapsedTime = endTime - startTime;
            System.out.println("Elapsed time (seconds): " + (elapsedTime / 1000.0));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
