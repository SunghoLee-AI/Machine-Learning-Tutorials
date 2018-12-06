# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:50:09 2018

@author: SL
"""

#------------------------------------------------------------------------------
# Inputs:   1.Read in HDD (Heart Disease Dataset)
#------------------------------------------------------------------------------
# Outputs:  1.Data Exploration and Pre-Processing
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
#------------------------------------------------------------------------------

#plt.close("all")

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

print("\n\n\n\n  --------------------Milestone 2 Assignment - By Sungho (Shawn) Lee-------------------------\n")
print("Introduction/Overview:")
print("I chose Heart Disease Dataset (HDD) for the sample dataset for this project")
print("Dataset Source: http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/")
print("It comes with 303 observations (i.e. patients) with 14 attributes for each observation.\n")


#------------------------------------------------------------------------------
print("\n1.Account for aberrant data (missing and outlier values)-----------------------------")
print("Based on my diagnostic analysis, it confirmed that there are missing values in two columns: CA, THAL")
#np.isnan(pd.to_numeric(HDD.loc[:, "NUM"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "THAL"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "CA"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "SLOPE"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "OLDPEAK"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "EXANG"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "THALACH"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "RESTECG"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "FBS"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "CHOL"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "TRESTBPS"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "CP"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "SEX"], errors='coerce')).sum()
#np.isnan(pd.to_numeric(HDD.loc[:, "AGE"], errors='coerce')).sum()
#
# COERCE TO NUMERIC AND IMPUTE MEDIANS FOR COLUMNs: CA, THAL
# for CA column
HDD.loc[:, "CA"] = pd.to_numeric(HDD.loc[:, "CA"], errors='coerce')
HasNan = np.isnan(HDD.loc[:,"CA"])
HDD.loc[HasNan, "CA"] = np.nanmedian(HDD.loc[:,"CA"])
# for THAL column
HDD.loc[:, "THAL"] = pd.to_numeric(HDD.loc[:, "THAL"], errors='coerce')
HasNan = np.isnan(HDD.loc[:,"THAL"])
HDD.loc[HasNan, "THAL"] = np.nanmedian(HDD.loc[:,"THAL"])

#
print("Based on my data exploration, it confirmed that there aren't really obvious outliers.")
#LimitHi = np.median(HDD.loc[:, "AGE"]) + 3*np.std(HDD.loc[:, "AGE"])
#LimitLo = np.median(HDD.loc[:, "AGE"]) - 3*np.std(HDD.loc[:, "AGE"])
#Index_outlier = (HDD.loc[:, "AGE"]<LimitLo) | (HDD.loc[:, "AGE"]>LimitHi) | (HDD.loc[:, "AGE"]<0)
#Index_legitimate = ~Index_outlier
#HDD.loc[Index_outlier, "AGE"] = np.mean(HDD.loc[Index_legitimate, "AGE"])


#------------------------------------------------------------------------------
print("\n2.Normalize numeric values (at least 1 column).--------------------------------------")
print("Values at 4 columns (AGE, TRESTBPS, CHOL, THALACH) are Z-normalized")
# To normalize columns using Z-normalization
HDD_Norm = HDD.copy()
# for AGE column
mu1 = np.mean(HDD.loc[:, "AGE"])
sigma1 = np.std(HDD.loc[:, "AGE"])
HDD_Norm.loc[:, "AGE"] = (HDD.loc[:, "AGE"] - mu1)/sigma1
# for TRESTBPS column
mu1 = np.mean(HDD.loc[:, "TRESTBPS"])
sigma1 = np.std(HDD.loc[:, "TRESTBPS"])
HDD_Norm.loc[:, "TRESTBPS"] = (HDD.loc[:, "TRESTBPS"] - mu1)/sigma1
# for CHOL column
mu1 = np.mean(HDD.loc[:, "CHOL"])
sigma1 = np.std(HDD.loc[:, "CHOL"])
HDD_Norm.loc[:, "CHOL"] = (HDD.loc[:, "CHOL"] - mu1)/sigma1
# for THALACH column
mu1 = np.mean(HDD.loc[:, "THALACH"])
sigma1 = np.std(HDD.loc[:, "THALACH"])
HDD_Norm.loc[:, "THALACH"] = (HDD.loc[:, "THALACH"] - mu1)/sigma1


#------------------------------------------------------------------------------
print("\n3.Bin numeric variables (at least 1 column).-----------------------------------------")
# Binning for values at column CHOL
# Equal-width Binning using numpy
NumberOfBins = 3
BinWidth = (HDD_Norm.loc[:, "CHOL"].max() - HDD_Norm.loc[:, "CHOL"].min())/NumberOfBins
MinBin1 = float('-inf')
MaxBin1 = HDD_Norm.loc[:, "CHOL"].min() + 1 * BinWidth
MaxBin2 = HDD_Norm.loc[:, "CHOL"].min() + 2 * BinWidth
MaxBin3 = float('inf')
print("Values at column CHOL are binned into 3 equal-width bins as defined below:)")
print("\n Bin 1 is from ", MinBin1, " to ", MaxBin1)
print(" Bin 2 is greater than ", MaxBin1, " up to ", MaxBin2)
print(" Bin 3 is greater than ", MaxBin2, " up to ", MaxBin3)
#
Binned_CHOL = np.array([" "]*len(HDD_Norm.loc[:, "CHOL"])) # Empty starting point for equal-width-binned array
Binned_CHOL[(MinBin1 < HDD_Norm.loc[:, "CHOL"]) & (HDD_Norm.loc[:, "CHOL"] <= MaxBin1)] = "L" # Low
Binned_CHOL[(MaxBin1 < HDD_Norm.loc[:, "CHOL"]) & (HDD_Norm.loc[:, "CHOL"] <= MaxBin2)] = "M" # Med
Binned_CHOL[(MaxBin2 < HDD_Norm.loc[:, "CHOL"]) & (HDD_Norm.loc[:, "CHOL"] < MaxBin3)] = "H" # High
#
print("Values at column CHOL are binned into 4 equal-width bins: ")
print(Binned_CHOL)


#------------------------------------------------------------------------------
print("\n4.Consolidate categorical data (at least 1 column).----------------------------------")
# Consolidagte categorical data
print("In this dataset, there is no categorical data. So, I just had to pick a data column for this exercise.")
print("Values at Column CA are consolidated based on the following criteria:")
print("Benign Condition: z-normalized CA<1.0")
print("Severe Condition: z-normalized CA>=1.0")
#
Ind_Benign=(HDD_Norm.loc[:, "CHOL"] < 1.0)
HDD_Norm.loc[Ind_Benign, "CHOL"] = "Benign"
HDD_Norm.loc[~Ind_Benign, "CHOL"] = "Severe"


#------------------------------------------------------------------------------
print("\n5.Remove obsolete columns.-----------------------------------------------------------")
# Remove obsolete columns
print("I picked column EXANG just for the purpose of doing this exercise (to remove obsolete columns)")
HDD_Norm = HDD_Norm.drop("EXANG", axis=1)


#------------------------------------------------------------------------------
# WRITE A LOCAL COPY OF THE FILE
# index=Flase --> means I don't want the index names (row and collum headers) stored when I create this new data
HDD_Norm.to_csv('SUNGHO(SHAWN)_LEE-M02-Dataset.csv', sep=",", index=False)
# Where is the file located?
import os
os.getcwd()
os.listdir()
# Check the file displays the same dataframe as before
HDD_Norm=pd.read_csv('SUNGHO(SHAWN)_LEE-M02-Dataset.csv')
HDD_Norm.head()
#------------------------------------------------------------------------------

print("\n\n\n  --------------------                   THE END                    -------------------------\n")
print("  --------------------                  THANK YOU                   -------------------------\n\n")

#---------------------------------------------------    ENd     ---------------------------------------------------

