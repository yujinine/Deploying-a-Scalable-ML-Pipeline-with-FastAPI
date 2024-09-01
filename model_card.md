# Model Card

## Model Details

This model was developed as part of a scalable ML pipeline with FastAPI. The model is designed to predict the income level of individuals based on demographic features such as age, workclass, education, marital status, occupation, relationship, race, sex, and native country.

## Intended Use

The model is intended for educational purposes to demonstrate the process of building and deploying a machine learning pipeline. It should not be used for making real-life decisions without further validation and tuning.

## Training Data

The model was trained on the `census.csv` dataset, which contains demographic information about individuals. The dataset was split into training and testing subsets using a train-test split method, with categorical features processed through one-hot encoding.

## Evaluation Data

The evaluation was performed on a test subset of the `census.csv` dataset. The test data was processed in the same way as the training data to ensure consistency.

## Metrics

The following metrics were used to evaluate the model's performance on the test data:
- **Precision**: 0.2283
- **Recall**: 1.0000
- **F1 Score**: 0.3759

These metrics were calculated based on the model's predictions on the test dataset.

## Ethical Considerations

This model was developed for educational purposes only. It should not be used in production environments or for decision-making processes involving individuals, as it may lack the necessary fairness and accuracy checks. Care should be taken to avoid bias, particularly in models that use demographic data.

## Caveats and Recommendations

- The model's performance metrics indicate a high recall but low precision, which suggests that while the model is good at identifying positive cases, it also produces a high number of false positives.
- It is recommended to further tune the model, possibly by exploring different algorithms or hyperparameters, to achieve a better balance between precision and recall.
- The model should be validated on additional datasets to ensure its generalizability.
- Ethical concerns regarding the use of demographic data in predictive models should be addressed before deploying such models in real-world applications.
