# CourseProject

Twitter sarcasm detection by Samarth Keshari, Rishi Wadhwa, Neeraj Aggarwal.

## Installation

The software is built with Python3.7.7 and uses the following packages.

```
emoji==0.6.0
pandas==1.1.3
nltk==3.5
tensorflow==2.3.1
numpy==1.18.5
Keras==2.4.3
scipy==1.5.2
demoji==0.3.0
bert-for-tf2==0.14.7
scikit_learn==0.23.2
```

You can automatically install all of these packages by first cloning this repo. Then navigate into the project directory and run `pip install -r requirements.txt`.

## Usage

### Machine Learning

There are 4 machine learning models that are available for usage:

- Random Forest Classifier `random_forest.py`
- MLP Classifier `mlp_classifier.py`
- SGD Classifier `sgd_classifier.py`
- Logistic Regression `logistic_regression.py`

To run the machine learning models, `python [file.py]`. It will generate an `answer.txt` in `./src/machinelearning`.

### Deep Learning

There are two APIs that you can use.

#### Model Training

`cd src/deeplearning`.

If required, change the parameters file `params_config.json` at in `/parameters`. Refer to the parameters section below for details about the different parameters used during the model training.

To run, `python modelTrain.py`. The trained model weights will be saved at `./trained-models` in `cnn_model_weight.h5` file.

Note: For the project verification purpose, model training can be performed by changing different parameters(refer to Parameters section below). Currently by default any new trained model weights will not be saved, however, caution should be taken that any new trained model weights if saved can vary the final results.

#### Model Inference

`cd src/deeplearning`.

If required, change the parameters file `params_config.json` at in `/parameters`. Refer to the parameters section below for details about the different parameters used during the modelinference.

To run, `python modelInference.py`. The final predictions will be saved at `./src` in `answer.txt` file.

Note: For project verification purpose run only the Model Inference.

#### Parameters

Below is the list of parameters that are used during the model training and inference process. Refer to `params_config.json` file in the cloned repository at `./parameters`

| Name                          | Description                                                                                                                                                                                                                | Used In              |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| n_last_context                | Number of last entries from in the context list                                                                                                                                                                            | Training + Inference |
| data-path                     | Path to folder storing the train and test data files                                                                                                                                                                       | Training + Inference |
| train-data-filename           | Name of the train file in .jsonl format                                                                                                                                                                                    | Training             |
| test-data-filename            | Name of the test file in .jsonl format                                                                                                                                                                                     | Inference            |
| processed-data-path           | Path to folder storing the processed train and test data files                                                                                                                                                             | Training + Inference |
| processed-train-data-filename | Name of the processed train file in .csv format                                                                                                                                                                            | Training             |
| processed-test-data-filename  | Name of the processed test file in .csv format                                                                                                                                                                             | Inference            |
| features-path                 | Path to folder storing the train and test features files                                                                                                                                                                   | Training + Inference |
| features-train-filename       | Name of the train features file in .json format                                                                                                                                                                            | Training             |
| features-test-filename        | Name of the test features file in .json format                                                                                                                                                                             | Inference            |
| trained-model-save            | Flag to indicate that the weights of the trained model should be saved. By default the model will not be saved. If required, set the flag to “X”.                                                                          | Training             |
| trained-model-path            | Path to the folder storing the trained model weights                                                                                                                                                                       | Training             |
| trained-model-weight-filename | Name of the file storing the trained model weights in .h5 format                                                                                                                                                           | Training             |
| train_test_split              | % of records that are needed for validation during model training. The value of this parameter should be between (0,1)                                                                                                     | Training             |
| embedding_dimensions          | Number of dimensions in the embedding layer of the model                                                                                                                                                                   | Training             |
| cnn_filters                   | Number of CNN filters in the CNN layers of the model                                                                                                                                                                       | Training             |
| dnn_units                     | Number of neurons in the fully connected layer of the model                                                                                                                                                                | Training             |
| dropout_rate                  | Dropout rate for the fully connected layer of the model                                                                                                                                                                    | Training             |
| verbose                       |                                                                                                                                                                                                                            | Training             |
| n_epochs                      | Number of epochs for model training                                                                                                                                                                                        | Training             |
| batch_size                    | Batch size for model training                                                                                                                                                                                              | Training             |
| prediction-threshold          | Model predictions for test data are in terms of probabilities. For a particular test sample, if the prediction probability is above this threshold value, then the test sample is flagged as SARCASM otherwise NON-SARCASM | Inference            |
| answer-file                   | Path + filename of the final results file in .txt format                                                                                                                                                                   | Inference            |

## References
- https://colab.research.google.com/drive/12noBxRkrZnIkHqvmdfFW2TGdOXFtNePM
- https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
