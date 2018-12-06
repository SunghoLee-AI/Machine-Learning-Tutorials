# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:50:09 2018

@author: SL
"""

#------------------------------------------------------------------------------
# Inputs:   1.Read in HDD (Heart Disease Dataset)
#------------------------------------------------------------------------------
# Outputs / Data Analysis Process:
#           1.Data Exploration and Pre-Processing
#           2.Construct Classification Models - Training & Testing
#           3.Evaluate Performance of the Developed Classification Models By Using CM (Confusion Matrix), ROC curves and Few Other Metrics Too
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
#
from sklearn.metrics import *
#
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR
from copy import deepcopy
#------------------------------------------------------------------------------

plt.close()

#------------------------------------------------------------------------------
# Auxiliary Functions
""" Auxiliary functions """
def normalize(X): # max-min normalizing
    N, m = X.shape
    Y = np.zeros([N, m])
    
    for i in range(m):
        mX = min(X[:,i])
        Y[:,i] = (X[:,i] - mX) / (max(X[:,i]) - mX)
    
    return Y

def split_dataset(data, r): # split a dataset in matrix format, using a given ratio for the testing set
	N = len(data)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

	n = int(round(N*r)) # number of elements in testing sample
	nt = N - n # number of elements in training sample
	ind = -np.ones(n,int) # indexes for testing sample
	R = np.random.randint(N) # some random index from the whole dataset
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) # ensure that the random index hasn't been used before
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) # remaining indexes	
	X = data[ind_,:-1] # training features
	XX = data[ind,:-1] # testing features
	Y = data[ind_,-1] # training targets
	YY = data[ind,-1] # testing targests
	return X, XX, Y, YY
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# ORIGIN OF DATA
# Data Set: http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# DOWNLOAD DATA
# DATA SO CALLED 'HDD (Heart Disease Dataset)'
HDD = pd.read_csv(url, header=None)
HDD.head()
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# ADD HEADERS
HDD.columns = ["AGE","SEX","CP","TRESTBPS","CHOL","FBS","RESTECG","THALACH","EXANG","OLDPEAK","SLOPE","CA","THAL","NUM"]
HDD.head()
#------------------------------------------------------------------------------


print("\n\n\n\n  --------------------Milestone 3 Assignment - By Sungho (Shawn) Lee-------------------------\n")
print("Introduction/Overview:")
print("I chose Heart Disease Dataset (HDD) for the sample dataset for this project")
print("Dataset Source: http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/")
print("It comes with 303 observations (i.e. patients) with 14 attributes for each observation.\n")
print("\nAccount for aberrant data (missing and outlier values):")
print("Based on my diagnostic analysis, it confirmed that there are missing values in two columns: CA, THAL")


#------------------------------------------------------------------------------
# COERCE TO NUMERIC AND IMPUTE MEDIANS FOR COLUMNs: CA, THAL
# for CA column
HDD.loc[:, "CA"] = pd.to_numeric(HDD.loc[:, "CA"], errors='coerce')
HasNan = np.isnan(HDD.loc[:,"CA"])
HDD.loc[HasNan, "CA"] = np.nanmedian(HDD.loc[:,"CA"])
# for THAL column
HDD.loc[:, "THAL"] = pd.to_numeric(HDD.loc[:, "THAL"], errors='coerce')
HasNan = np.isnan(HDD.loc[:,"THAL"])
HDD.loc[HasNan, "THAL"] = np.nanmedian(HDD.loc[:,"THAL"])
#------------------------------------------------------------------------------


print("Based on my data exploration, it confirmed that there aren't really obvious outliers.")


#------------------------------------------------------------------------------
# Binning NUM into 0 (if value=0) and 1 (for value >0), to convert it into "Binary" Classification problem.
HDD.loc[HDD.loc[:,"NUM"]>0,"NUM"]=1
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# converting dataframe into numpy array
dataset=np.array(HDD.values)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Preprocessing the data for classification
""" Preprocessing the data for classification """
r = 0.2 # ratio of test data over all data (this can be changed to any number between 0.0 and 1.0 (not inclusive)
dataset = dataset[1:,:] # get rid of headers
all_inputs = normalize(dataset[:,:-1]) # inputs (features)
normalized_data = deepcopy(dataset)
normalized_data[:,:-1] = all_inputs
X, XX, Y, YY = split_dataset(normalized_data, r)
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# CLASSIFICATION MODELS


# MODEL1 START----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

""" CLASSIFICATION MODELS """
# k Nearest Neighbors classifier
print('\n\n\n\n                  ------ CLASSIFICATION MODEL1 ------                  ')
print ('                    (K nearest neighbors classifier)                   ')
k = 5 # number of neighbors
distance_metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
knn.fit(X, Y)
#
TargetLabels=YY
PredictedLabels=knn.predict(XX)
PredictedProb=knn.predict_proba(XX)[:,1]
print ('actual class values:')
print (TargetLabels)
print ("predictions for test set:")
print (PredictedLabels)
print ("predicted Probability for test set:")
print (PredictedProb)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Confusion Matrix (comes from "sklearn")
CM = confusion_matrix(TargetLabels, PredictedLabels)
print ("\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TargetLabels, PredictedLabels)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(TargetLabels, PredictedLabels)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(TargetLabels, PredictedLabels)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(TargetLabels, PredictedLabels)
print ("\nF1 score:", np.round(F1, 2))
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color
#
fpr, tpr, th = roc_curve(TargetLabels, PredictedProb) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Plotting
plt.figure(1)
plt.title('ROC - MODEL 1, 2, 3')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='MODEL1 (K nearest neighbors classifier)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend()
#plt.legend(loc=LL)
plt.show()
#------------------------------------------------------------------------------

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(TargetLabels, PredictedProb), 2), "\n")

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
# MODEL1 END------------------------------------------------------------------------------------------



# MODEL2 START----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

""" CLASSIFICATION MODELS """
# Decision Tree Classifier
print('\n\n\n\n                  ------ CLASSIFICATION MODEL2 ------                  ')
print ('                      (Decision Tree Classifier)                   ')
dt = DecisionTreeClassifier()
dt.fit(X, Y)
#
TargetLabels=YY
PredictedLabels=dt.predict(XX)
PredictedProb=dt.predict_proba(XX)[:,1]
print ('actual class values:')
print (TargetLabels)
print ("predictions for test set:")
print (PredictedLabels)
print ("predicted Probability for test set:")
print (PredictedProb)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Confusion Matrix (comes from "sklearn")
CM = confusion_matrix(TargetLabels, PredictedLabels)
print ("\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TargetLabels, PredictedLabels)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(TargetLabels, PredictedLabels)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(TargetLabels, PredictedLabels)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(TargetLabels, PredictedLabels)
print ("\nF1 score:", np.round(F1, 2))
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# ROC analysis
LC = 'darkred' # Line Color
#
fpr, tpr, th = roc_curve(TargetLabels, PredictedProb) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Plotting
plt.figure(1)
plt.plot(fpr, tpr, color=LC,lw=LW, label='MODEL2 (Decision Tree Classifier)' % AUC)
plt.legend()
#plt.legend(loc=LL)
plt.show()
#------------------------------------------------------------------------------

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(TargetLabels, PredictedProb), 2), "\n")

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
# MODEL2 END------------------------------------------------------------------------------------------



# MODEL3 START----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

""" CLASSIFICATION MODELS """
# Logistic regression classifier
print('\n\n\n\n                  ------ CLASSIFICATION MODEL3 ------                  ')
print ('                    (Logistic Regression Classifier)                   ')
C_parameter = 50. / len(X) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter
#####################
lr = LogisticRegression(C=C_parameter, multi_class=class_parameter, penalty=penalty_parameter, solver=solver_parameter, tol=tolerance_parameter)
lr.fit(X, Y)
#
TargetLabels=YY
PredictedLabels=lr.predict(XX)
PredictedProb=lr.predict_proba(XX)[:,1]
print ('actual class values:')
print (TargetLabels)
print ("predictions for test set:")
print (PredictedLabels)
print ("predicted Probability for test set:")
print (PredictedProb)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Confusion Matrix (comes from "sklearn")
CM = confusion_matrix(TargetLabels, PredictedLabels)
print ("\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TargetLabels, PredictedLabels)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(TargetLabels, PredictedLabels)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(TargetLabels, PredictedLabels)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(TargetLabels, PredictedLabels)
print ("\nF1 score:", np.round(F1, 2))
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# ROC analysis
LC = 'darkblue' # Line Color
#
fpr, tpr, th = roc_curve(TargetLabels, PredictedProb) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Plotting
plt.figure(1)
plt.plot(fpr, tpr, color=LC,lw=LW, label='MODEL3 (Logistic Regression Classifier)' % AUC)
plt.legend()
#plt.legend(loc=LL)
plt.show()
#------------------------------------------------------------------------------

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(TargetLabels, PredictedProb), 2), "\n")

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
# MODEL3 END------------------------------------------------------------------------------------------



# RESULTS & CONCLUSIONS----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

print('\n\n\n\n                  ------ MODELS EVALUATION RESULTS & CONCLUSIONS ------                  ')
print('\n1. I picked three classification models for this exercise: Model1: K Nearest Neighbors (KNN), Model2: Decision Tree Classifier, Model3: Logistic Regression Classifier. Logistic regression classifier is known as one of the simplest and the most useful and robust classifier as taught in this class by Dr. Henle and I wanted to test it against other two complex classifiers. I picked KNN model too because it seems quite simple, but also it is a very robust classifier for a binary classification problem like this. I also picked Decision Tree Classifier because again I am dealing with a binary classification problem and I expect Decision Tree Classifier may have a lower performance compared against the KNN which is considered as a more complex model than Decision Tree model. I wanted to confirm its relative performance comparisons.')
print('\n2. As shown in ROCs, Model2 ROC is typically under the other two ROCs from Model1 and Model3. This means that Model1 and 3 always have a higher True Positive Rate (TPR) given the same False Positive Rate (FPR). In other words, for the same price level (let us say the false positive is the price we are paying to achieve a certain sensitivity level in predicting the heart disease), the Model1 and 3 have a much better performance in predicting the heart disease. Therefore, there is no reason to pick Model2 over Model1 and 3 for the present classification problem.')
print('\n3. Also as shown in ROCs, I think the overall performance of Model3 seems slightly better than Model1 because ROC of Model3 is typically (but not always though) above ROC of Model1. However, it is hard to make a definitive answer to which model is superior than the other because it will depend on the requirement from the user/client to receive this model from me. For example, if a maximum acceptable false positive rate (FPR) is given as a requirement, our goal would be to pick a whichever model that gives us the highest true positive rate (TPR) at this maximum acceptable FPR level at ROC curve (i.e. a vertical line drawn over somewhere at ROC curve). But again, Model3 seems to have typically a higher chance of the prediction being true.')
print('\n[PLEASE NOTE: The exact numeric numbers in the following performance evaluation notes below (#4, #5, #6) is merely based on a specific group of testing data that was chosen by executing this python code (20% of the total dataset randomly chosen at each time of execution)]')
print('\n4. Based on the evaluation of Model1 performance for a default threshold (50% probability), the precision rate turns out to be 0.78 which means it has 78% probability that the prediction is true for a given patient (a binary classification in this case: whether the patient will have a heart disease or not). And the recall rate is 0.75 which is a little lower than precision rate. But at the same time, the false positive rate is around 0.14. It is hard to judge whether this set of TPR and FPR is an optimal set. But, based on ROC, this particular set of TPR and FPR seems to be located at the far top-left corner of the ROC curve, so I think 50% threshold is a generally acceptable optimal threshold to be used in Model1 for this classification problem. However, if we want to penalize the false negative much more for example, we would want to increase the threshold (higher than 50%) although we would have to take the higher cost (the TPR will be decreased as well).')
print('\n5. In terms of the accuracy rate which is a very different way of evaluating the classifier compared to TPR & FPR quantities, Model3 still has the highest accuracy rate compared to Model1 and Model2, for the case of 50% probability threshold selected.')
print('\n6. Again for the case of 50% probability threshold selected, for both precision and recall rates, Model3 has the highest rate compared to Model1 and 2. This explains why the F1 score (which is the average of these two metrics) is the highest at Model3 as well. These were all confirmed by comparing the precision, recall and F1 scores that are computed in this code.')

print("\n\n\n  --------------------                   THE END                    -------------------------\n")
print("  --------------------                  THANK YOU                   -------------------------\n\n")


#---------------------------------------------------    END     ---------------------------------------------------

