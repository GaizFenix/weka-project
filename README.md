# Weka Project

## Overview
This project implements various machine learning algorithms using the Weka framework. It focuses on processing data for verbal autopsy analysis, utilizing techniques such as AdaBoost and Grid Search for model optimization.

## Key Components
- **Data Processing**: The project includes scripts for converting CSV data to ARFF format and applying filters to prepare the data for analysis.
- **Machine Learning Models**: Implements AdaBoost with J48 as the base classifier, along with Grid Search for hyperparameter tuning.
- **Evaluation**: The models are evaluated using various metrics, and results are printed to the console.

## File Structure
```
.
├── .gitignore
├── design.pdf
├── LICENSE
├── README.md
├── TODO.txt
├── data
│   ├── arff
│   ├── aux
│   └── raw
└── src
    ├── Arff2BowForStats.java
    ├── installGridSearchLib.java
    ├── treeAdaBoostGS.java
    └── ...
```

## Usage
To run the AdaBoost model, use the following command:
```
java treeAdaBoostGS <input_train.arff> <input_dev.arff> <input_dictionary.txt>
```

### Example
```bash
java treeAdaBoostGS data/arff/data_train.arff data/arff/data_dev_raw.arff data/aux/dictionary_4_stats_final.txt
```

## References
- [Verbal Autopsy](https://www.healthdata.org/verbal-autopsy)
- [WHO Verbal Autopsy Standards](https://www.who.int/standards/classifications/other-classifications/verbal-autopsy-standards-ascertaining-and-attributing-causes-of-death-tool)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

