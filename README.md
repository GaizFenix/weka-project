# Weka Project

## Overview
This repository contains a project that demonstrates the execution of various machine learning tasks using Weka. Below, you will find detailed instructions for preprocessing, inference, and classification.

---

## Table of Content

1. [Key Components](#key-components)
2. [Project structrure](#project-structure)
3. [Execution Example](#execution-examples)
   - [Preprocessing](#preprocessing)
   - [Inference](#inference)
   - [Classification](#classification)
3. [Requirements](#requirements)
4. [Refereneces](#references)
5. [License](#license)

---

## Key Components
- **Data Processing**: The project includes scripts for converting CSV data to ARFF format and applying filters to prepare the data for analysis.
- **Machine Learning Models**: Implements AdaBoost with J48 as the base classifier, along with Grid Search for hyperparameter tuning.
- **Evaluation**: The models are evaluated using various metrics, and results are printed to the console.


## Project Structure
```
.
├── design.pdf
├── LICENSE
├── README.md
├── data
│   ├── arff
│   ├── aux
│   └── raw
└── src
    ├── preprocess.java
    ├── getModel.java
    ├── classify.java
    └── ...
```

## Execution Example

### Preprocessing

- **Input**: `<raw.train.csv>`
- **Output**: `<train.arff>` and `<dev.arff>`

Command with example parameters (change depending what you use):

```bash
java -jar preprocess.jar data/raw/cleaned_PHMRC_VAI_redacted_free_text.train.csv data/arff/data_train_bow.arff data/arff/data_dev_raw.arff
```

### Inference

- **Input**: `<train.arff>` and `<dev.arff>`
- **Output**: `<kalitatearen estimazioa.txt>` eta `<model>`

Command with example parameters (change depending what you use):

```bash
java -jar getModel.jar data/arff/data_train_bow.arff data/arff/data_dev_raw.arff data/AdaModel.mdl
```

### Classification

- **Input**: `<model>` and `<raw_test_blind.csv>`
- **Output**: `<sailkapena.txt>`

Command with example parameters (change depending what you use):

```bash
java -jar classify.jar data/AdaModel.mdl data/raw/cleaned_PHMRC_VAI_redacted_free_text.train.csv data/iragarpenEmaitzak.txt
```

## Requirements

- Java Runtime Environment (JRE)
- Weka libraries
- Input files in .csv or .arff format


## References
- [Verbal Autopsy](https://www.healthdata.org/verbal-autopsy)
- [WHO Verbal Autopsy Standards](https://www.who.int/standards/classifications/other-classifications/verbal-autopsy-standards-ascertaining-and-attributing-causes-of-death-tool)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

