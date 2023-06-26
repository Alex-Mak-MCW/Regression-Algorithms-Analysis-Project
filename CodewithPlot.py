#!/usr/bin/env python3

# import libraries and dependencies
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, RidgeClassifier, Lasso, LassoCV, ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score, f1_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV

# For confusion matrix and heatmap
import matplotlib.pyplot as plt
import seaborn as sns

# (OPTION) Load dataset direct from sklearn
# from sklearn.datasets import load_breast_cancer

# read the breast cancer dataset
df = pd.read_csv('data.csv')

# X are the factors that caused the breast cancer/tumor
X=df.iloc[:,2:31].values
# Y is a categorical variable that shows whether the tumor is benign (safe), or malicious (cancer)
Y=df.iloc[:,1].values

# Encode categorical data (Y here) to 0 and 1 using label encoder
labelEncoder_Y=LabelEncoder()
Y=labelEncoder_Y.fit_transform(Y)

# 80(training)-10(valdiation)-10(test) split of the training and testing data

# Split data into 2: 80% train, 20% hold (validation, test)
X_train, X_hold, Y_train, Y_hold = train_test_split(X, Y, test_size=0.2, random_state=0)

# Split the hold into validation and test (10% each)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_hold, Y_hold, test_size=0.5, random_state=0)

# Standardize the features using StandardScaler for all 3 models (training, valdiation, testing)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_valid=sc.fit_transform(X_valid)
X_test=sc.fit_transform(X_test)

# ----------------------------------------------------------
# Linear Regression PART (Baseline)
# Fit a linearregression model using sklearn
linear_model=LinearRegression()

# Train the training model
linear_model.fit(X_train, Y_train)

# Validation:
linear_Y_valid_pred=linear_model.predict(X_valid)

# handle outlier
for i in range(len(linear_Y_valid_pred)):
    if linear_Y_valid_pred[i]>1.5:
        linear_Y_valid_pred[i]=1.45

linear_Y_valid_pred=linear_Y_valid_pred.round()

# Test the training model using the test dataset
linear_Y_test_pred=linear_model.predict(X_test)

# handle outlier
for i in range(len(linear_Y_test_pred)):
    if linear_Y_test_pred[i]>1.5:
        linear_Y_test_pred[i]=1.45
linear_Y_test_pred=linear_Y_test_pred.round()

print("\nFor Linear Regression (Baseline 1):")
# print("Valdiation Accuracy: ",accuracy_score(linear_Y_valid_pred,Y_test))
print("Test Accuracy: ",accuracy_score(linear_Y_test_pred,Y_test))

# Confusion Matrix display (Linear Regression)
linear_conf_matrix = confusion_matrix(Y_test, linear_Y_test_pred)

fig, ax = plt.subplots(figsize=(8,6), dpi=100)
sns.set(font_scale = 1.1)

# Display Confusion Matrix through Heatmap
ax = sns.heatmap(linear_conf_matrix, annot=True, fmt='d', )
# set x-axis label and ticks. 
ax.set_xlabel("Predicted Diagnosis", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(['Negative', 'Positive'])

# set y-axis label and ticks
ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(['Negative', 'Positive'])

ax.set_title("Confusion Matrix for the Linear Regression Model", fontsize=14, pad=20)
# plt.show()
plt.savefig("LinearRegressionCM.jpg")

# ----------------------------------------------------------
# LOGISTIC Regression PART
# Fit a logistic regression model using sklearn
logistic_model=LogisticRegression(max_iter=1000)

# Train the model
logistic_model.fit(X_train, Y_train)

print("\nFor Logistic Regression (Baseline 2):")
# Test the training model using the validation dataset

logistic_Y_valid_pred=logistic_model.predict(X_valid)

# Test the training model using the test dataset
logistic_Y_test_pred=logistic_model.predict(X_test)

# For testing before and after hyperparameter tuning
"""
print("Before hyperparameter tuning: ")
print("Valdiation Accuracy: ",accuracy_score(logistic_Y_valid_pred,Y_test))
print("Test Accuracy: ",accuracy_score(logistic_Y_test_pred,Y_test))
"""

# Hyperparameter tuning
hyper=[{'C':[1/1e-15,1/1e-10,1/1e-8,1/1e-3,1/1e-2,1,1/5,1/10,1/20,1/30,1/35,1/40,1/45,1/50,1/55,1/100]}]
grid_search = GridSearchCV(estimator = logistic_model,
                           param_grid = hyper,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search.fit(X_valid, Y_valid)
best_accuracy_log = grid_search.best_score_
best_parameters = grid_search.best_params_

best_log_stepsize=best_parameters["C"]

# Train the model with the best stepsize
tuned_logistic_model=LogisticRegression( C=best_log_stepsize, max_iter=1000)

tuned_logistic_model.fit(X_train, Y_train)

# Test the training model using the test dataset
logistic_Y_test_pred=tuned_logistic_model.predict(X_test)

print("After hyperparameter tuning: ")
# print("Valdiation Accuracy:",best_accuracy_log)
print("Test Accuracy:",accuracy_score(logistic_Y_test_pred,Y_test))

# Confusion Matrix display (Logistic Regression)
logistic_conf_matrix = confusion_matrix(Y_test, logistic_Y_test_pred)

fig, ax = plt.subplots(figsize=(8,6), dpi=100)
sns.set(font_scale = 1.1)

# Display Confusion Matrix through Heatmap
ax = sns.heatmap(logistic_conf_matrix, annot=True, fmt='d', )
# set x-axis label and ticks. 
ax.set_xlabel("Predicted Diagnosis", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(['Negative', 'Positive'])

# set y-axis label and ticks
ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(['Negative', 'Positive'])

ax.set_title("Confusion Matrix for the Logistic Regression Model", fontsize=14, pad=20)
# plt.show()
plt.savefig("LogisticRegressionCM.jpg")

# ----------------------------------------------------------
# Lasso Regression PART
# Fit a lasso regression model using sklearn
lasso_model=LogisticRegression(penalty='l1',solver='liblinear', max_iter=1000, tol=0.1)

# Train the model
lasso_model.fit(X_train, Y_train)

print("\nFor Lasso Regression (Approach 1):")
# Test the training model using the validation dataset

lasso_Y_valid_pred=lasso_model.predict(X_valid)

# Test the training model using the test dataset
lasso_Y_test_pred=lasso_model.predict(X_test)

# For testing before and after hyperparameter tuning
"""
print("Before hyperparameter tuning: ")
print("Valdiation Accuracy: ",accuracy_score(lasso_Y_valid_pred,Y_test))
print("Test Accuracy: ",accuracy_score(lasso_Y_test_pred,Y_test))
"""

# Hyperparameter tuning
hyper=[{'C':[1/1e-15,1/1e-10,1/1e-8,1/1e-3,1/1e-2,1,1/5,1/10,1/20,1/30,1/35,1/40,1/45,1/50,1/55,1/100]}]
grid_search = GridSearchCV(estimator = lasso_model,
                           param_grid = hyper,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search.fit(X_valid, Y_valid)
best_accuracy_log = grid_search.best_score_
best_parameters = grid_search.best_params_

best_stepsize2=best_parameters["C"]

# Train the model with the best stepsize
tuned_lasso_model=LogisticRegression(penalty='l1',solver='liblinear', C=best_stepsize2, max_iter=1000, tol=0.01)

tuned_lasso_model.fit(X_train, Y_train)
# Test the training model using the test dataset
lasso_Y_test_pred=tuned_lasso_model.predict(X_test)

print("After hyperparameter tuning: ")
# print("Valdiation Accuracy:",best_accuracy_log)
print("Test Accuracy:",accuracy_score(lasso_Y_test_pred,Y_test))

# Confusion Matrix display (Lasso Regression)
lasso_conf_matrix = confusion_matrix(Y_test, lasso_Y_test_pred)

fig, ax = plt.subplots(figsize=(8,6), dpi=100)
sns.set(font_scale = 1.1)

# Display Confusion Matrix through Heatmap
ax = sns.heatmap(lasso_conf_matrix, annot=True, fmt='d', )
# set x-axis label and ticks. 
ax.set_xlabel("Predicted Diagnosis", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(['Negative', 'Positive'])

# set y-axis label and ticks
ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(['Negative', 'Positive'])

ax.set_title("Confusion Matrix for the Lasso Regression Model", fontsize=14, pad=20)
# plt.show()
plt.savefig("LassoRegressionCM.jpg")

# ----------------------------------------------------------
# Ridge Regression PART
# Fit a ridge regression model using sklearn
ridge_model=RidgeClassifier(max_iter=1000)

# Train the model
ridge_model.fit(X_train, Y_train)

print("\nFor Ridge Regression (Approach 2):")

# Test the training model using the validation dataset
ridge_Y_valid_pred=ridge_model.predict(X_valid)

# Test the training model using the test dataset
ridge_Y_test_pred=ridge_model.predict(X_test)

# For testing before and after hyperparameter tuning
"""
print("Before hyperparameter tuning: ")
print("Valdiation Accuracy: ",accuracy_score(ridge_Y_valid_pred,Y_test))
print("Test Accuracy: ",accuracy_score(ridge_Y_test_pred,Y_test))
"""

# Hyperparameter tuning
parameters = [{'max_iter': [5, 10, 50, 100]}]
hyper=[{'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}]
grid_search = GridSearchCV(estimator = ridge_model,
                           param_grid = hyper,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search.fit(X_valid, Y_valid)
best_accuracy_log = grid_search.best_score_
best_parameters = grid_search.best_params_

best_stepsize=best_parameters["alpha"]

# Train the model with the best stepsize
tuned_ridge_model=RidgeClassifier(alpha=best_stepsize, max_iter=1000)

tuned_ridge_model.fit(X_train, Y_train)
# Test the training model using the test dataset
ridge_Y_test_pred=tuned_ridge_model.predict(X_test)

print("After hyperparameter tuning: ")
# print("Valdiation Accuracy:",best_accuracy_log)
print("Test Accuracy:",accuracy_score(ridge_Y_test_pred,Y_test))

# Confusion Matrix display (Ridge Regression)
ridge_conf_matrix = confusion_matrix(Y_test, ridge_Y_test_pred)

fig, ax = plt.subplots(figsize=(8,6), dpi=100)
sns.set(font_scale = 1.1)

# Display Confusion Matrix through Heatmap
ax = sns.heatmap(ridge_conf_matrix, annot=True, fmt='d', )
# set x-axis label and ticks. 
ax.set_xlabel("Predicted Diagnosis", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(['Negative', 'Positive'])

# set y-axis label and ticks
ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(['Negative', 'Positive'])

ax.set_title("Confusion Matrix for the Ridge Regression Model", fontsize=14, pad=20)
# plt.show()
plt.savefig("RidgeRegressionCM.jpg")

# ----------------------------------------------------------
# Elastic Net Regression PART
# Fit a ridge regression model using sklearn
net_model=SGDClassifier(penalty='elasticnet', max_iter=1000)

# Train the model
net_model.fit(X_train, Y_train)

print("\nFor Elastic Net Regression (Approach 3):")

# Test the training model using the validation dataset
net_Y_valid_pred=net_model.predict(X_valid)

# Test the training model using the test dataset
net_Y_test_pred=net_model.predict(X_test)

# For testing before and after hyperparameter tuning
"""
print("Before hyperparameter tuning: ")
print("Valdiation Accuracy: ",accuracy_score(net_Y_valid_pred,Y_test))
print("Test Accuracy: ",accuracy_score(net_Y_test_pred,Y_test))
"""

# Hyperparameter tuning
hyper=[{'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}]
grid_search = GridSearchCV(estimator = net_model,
                           param_grid = hyper,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search.fit(X_valid, Y_valid)
best_accuracy_log = grid_search.best_score_
best_parameters = grid_search.best_params_

best_stepsize3=best_parameters["alpha"]

# Train the model with the best stepsize
tuned_net_model=SGDClassifier(penalty='elasticnet', alpha=best_stepsize3, max_iter=1000)
# tuned_lasso_model=LassoCV(alpha=best_stepsize2)

tuned_net_model.fit(X_train, Y_train)
# Test the training model using the test dataset
net_Y_test_pred=tuned_net_model.predict(X_test)

print("After hyperparameter tuning: ")
# print("Valdiation Accuracy:",best_accuracy_log)
print("Test Accuracy:",accuracy_score(net_Y_test_pred,Y_test))

# Confusion Matrix display (Elastic Net Regression)
net_conf_matrix = confusion_matrix(Y_test, net_Y_test_pred)

fig, ax = plt.subplots(figsize=(8,6), dpi=100)
sns.set(font_scale = 1.1)

# Display Confusion Matrix through Heatmap
ax = sns.heatmap(net_conf_matrix, annot=True, fmt='d', )
# set x-axis label and ticks. 
ax.set_xlabel("Predicted Diagnosis", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(['Negative', 'Positive'])

# set y-axis label and ticks
ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(['Negative', 'Positive'])

ax.set_title("Confusion Matrix for the Elastic Net Regression Model", fontsize=14, pad=20)
# plt.show()
plt.savefig("ElasticNetRegressionCM.jpg")
