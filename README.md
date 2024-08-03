# Greendestination

# Problem Statement




![Project Objective](https://github.com/user-attachments/assets/b99b7407-5f45-44e0-93ee-28bc89758971)





* Employee attrition refers to an employee’s voluntary or involuntary resignation from a workforce. Organizations spend many resources in hiring talented employees and training them. Every employee is critical to a company’s success. Our goal was to predict employee attrition and identify the factors contributing to an employee leaving a workforce.
* We trained various classification models on our dataset and assessed their performance using different metrics such as accuracy, precision, recall and F1 Score. We also analyzed the dataset to identify key factors contributing to an employee leaving a workforce. Our project will assist organizations in gaining fresh insights into what drives attrition and thus enhance retention rate.


# Methodolody


![Flowchart](https://github.com/user-attachments/assets/9954fa42-6395-4a22-b368-29e44316c3dc)



# Machine Learning Models

* We trained and evaluated 9 supervised machine learning classification models.

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- AdaBoost
- Support Vector Machine
- Linear Discriminant Analysis
- Multilayer Perceptron
- K-Nearest Neighbors

** Further, to get the best performance, hyperparameter tuning was carried out using RandomSearchCV and GridSearchCV. K-fold cross-validation with 5 folds was also performed on the training set. To handle model interpretability, appropriate graphs and figures were used.Accuracy for the attrition decision is a biased metric, and hence we evaluated the model on all the following classification metrics: accuracy, precision, recall and F1 Score.

# Libraries Used:
1) Numpy
2) Pandas
3) Seaborn
4) Matplotlib
5) Scikit-learn

# Conclusion:
* Wetrained various supervised classification models (LR, NB, DT, RF, AdaBoost, SVM, LDA, MLP and KNN) and summarised their results in this project. As observed from EDA and our previous analysis, each model performed significantly worse on the unprocessed dataset, due to its imbalanced nature. The best performance was obtained in Random Forest Model with PCA and Oversampling withaccuracy of 99.2%, precision of 98.6%, recall of 99.8% and f1 score of 99.2%. Other models such as SVC and MLP also performed equally well with accuracies and F1 scores consistently more than 90%. Oversampling with PCA had better performances across models except LR and NB with tree based models having highest metric scores. In accordance to EDA, MonthlyIncome, Age, OverTime, Total WorkingYears played major roles in the attrition decision and Gender did not impact attrition.

