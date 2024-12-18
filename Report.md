# Analysis Report

## Introduction

The data is related with direct marketing campaigns of a banking institution. 
The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed. 

**Task:** Build a predictive model that determines the likelihood of a client subscribing to a term deposit based on the features provided in the dataset. 

**Reason for choosing the 'bank.csv' dataset:**
It will be easier on my laptop’s current memory and allow me to work with the data more efficiently.


## Methods

Here's an elaboration on the methods used for data processing:
1. Data Normalization/Scaling
Data normalization/scaling was performed using:
Standardization: Numerical features were standardized to have a mean of 0 and a standard deviation of 1.
2. Encoding Categorical Variables
Categorical variables were encoded using:
One-Hot Encoding: Categorical features were converted into binary vectors using this method.
3. Feature Engineering
New features were created using:
Domain Knowledge: New features were created based on domain knowledge to capture relevant information.

These data processing methods helped to:
Handle missing values and outliers
Normalize/scale data to prevent feature dominance
Encode categorical variables for modeling
Create new features to capture relevant information

## Results

1. Boxplots were used to check for outlier in numerical features and below is the summary:

#### Summary of Outliers

**Age:** Outliers include individuals older than 73. \
**Balance:** Outliers include negative balances and high positive balances above 3596.\
**Day:** No outliers. All values lie within the range. \
**Duration:** Outliers include call durations greater than 666 seconds (11 minutes).\
**Campaign:** Outliers include clients who were contacted more than 6 times. \
**Pdays:** Outliers include all positive values for pdays. -1 means client wasn't previously contacted. Positive pdays indicates days since last contacted.\
**Previous:** Outliers are all values greater than 0, indicating prior contacts.

2. Bar chart was used to visualize the target feature and it was found there was imbalance. \
The 'No' was 3193 and the 'Yes' was 423. \

The SMOTE (Synthetic Minority Over-sampling Technique) method which works by creating synthetic samples of the minority \ class was used. This method was chosen because:\
**Reduces overfitting:** By increasing the number of instances in the minority class, SMOTE helps reduce overfitting.\
**Improves model performance:** SMOTE can improve the performance of models on the minority class.\
**Easy to implement:** SMOTE is a simple method to implement. 


## Discussion

 Interpret the results, discuss any limitations, and provide recommendations for future work.
1. Performance of the algorithms used to train the machine learning model

**Key Metrics Overview:**
**Accuracy:** Proportion of correctly classified samples over all samples.
**Precision:** Of the positive predictions made, how many are actually positive? High precision means fewer false positives.
**Recall (Sensitivity):** Of all actual positives, how many were correctly identified? High recall means fewer false negatives.
**F1-Score:** Harmonic mean of precision and recall, giving equal importance to both.
**F2-Score:** Weighted measure that emphasizes recall over precision.

|Model|	Accuracy |	Precision	|Recall |	F1_score|	F2_score|
|--- | ---|
|Logistic Regression	|0.844199|	0.393035	|0.806122|	0.528428|	0.666105|
|Support Vector Machine	|0.851934	|0.391566|	0.663265|	0.492424|	0.582437|
|Random Forest|	0.891713	|0.500000	|0.418367	|0.455556	|0.432489|

**Observations:**
**Logistic Regression:** Performs well in recall (80.6%), meaning it identifies a large portion of actual subscribers. However, it suffers from a low precision (39.3%), meaning many "no" predictions are mistakenly labeled "yes."\
**SVM:** Offers a slightly better accuracy (85.2%) but trades recall (66.3%) for precision (39.2%). It is a more balanced approach but less effective than Logistic Regression at identifying potential subscribers.\
**Random Forest:** Has the best accuracy (89.2%) and highest precision (50%), making it great for minimizing false positives. However, it struggles with recall (41.8%), meaning it misses many potential subscribers.

#### Note: 
1. The task is to build a predictive model that determines the likelihood of a client subscribing to a term deposit based on the features provided in the dataset.\
Logistic Regression will be the best choice since the primary goal is to maximize recall, ensuring one captures as many potential subscribers as possible.

2. After performing hyperparmeter tuning on the Logistic Regression model, the result was same as the initial results for the Recall which was: 80.6%. \
This can be because:
-- The model might be overfitting or underfitting the training data, which can result in poor recall scores.\
-- The hyperparameter tuning process might not have explored a sufficient range of hyperparameters to improve the recall score. 

## Conclusion

This analysis aimed to build a predictive model that determines the likelihood of a client subscribing to a term deposit based on the features provided in the dataset. The key findings and takeaways from this analysis are:

1. The dataset exhibited class imbalance, with a significant majority of clients not subscribing to term deposits. To address this, the SMOTE technique was used to oversample the minority class.
2. Three machine learning models (Logistic Regression, Support Vector Machine, and Random Forest) were trained and evaluated on the dataset. The results showed that Logistic Regression performed best in terms of recall, making it the most suitable model for this task.
3. Hyperparameter tuning was performed on the Logistic Regression model, but unfortunately, it did not improve the recall score. This could be due to overfitting or underfitting, or insufficient exploration of the hyperparameter search space.

Based on these findings, the following recommendations are made:
1. Future work could focus on exploring other machine learning models and techniques that can better handle class imbalance and improve recall scores.
2. Collecting more data or using data augmentation techniques could help improve the model's performance.
3. Further hyperparameter tuning and exploration of different hyperparameter search spaces could help improve the model's performance.

Overall, this analysis provides a solid foundation for building predictive models that can help banking institutions identify potential clients who are likely to subscribe to term deposits.