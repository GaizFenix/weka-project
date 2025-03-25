import java.io.File;

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
        if (args.length != 4) {
            System.err.println("ERROR! Correct usage: java treeAdaBoost <input_train.arff> <input_dev.arff> <input_dictionary.txt> <output_stats.txt>");
            System.exit(1);
        }

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
            filter.setIDFTransform(true);
            filter.setTFTransform(true);
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
            ada.setNumIterations(10);

            // GridSearch
            GridSearch gs = new GridSearch();
            gs.setClassifier(j48);
            gs.setEvaluation(new SelectedTag(GridSearch.EVALUATION_HOLDOUT, GridSearch.TAGS_EVALUATION));

            // Set X property
            gs.setXProperty("classifier.confidenceFactor");
            gs.setXMin(0.05);
            gs.setXMax(0.25);
            gs.setXStep(0.05);
            gs.setXExpression("C");

            // Set Y property
            gs.setYProperty("classifier.minNumObj");
            gs.setYMin(2);
            gs.setYMax(10);
            gs.setYStep(2);
            gs.setYExpression("M");

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
            // [0] -> J48 confidenceFactor, [1] -> J48 minNumObj
            double[] bestParams = gs.getValues();
            System.out.println("confidenceFactor: " + bestParams[0]);
            System.out.println("minNumObj: " + bestParams[1]);

            System.out.println("\n=== Dev set evaluation ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
