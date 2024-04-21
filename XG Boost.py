# %% [markdown]
# XG Boost

# %% [markdown]
# Importing Libraries
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Importing Data Set

# %%
df=pd.read_csv("Social_Network_Ads.csv")
df
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x,y)

# %% [markdown]
# splitting The Data Set Into training and testing Sets
# 

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
print(x_train,x_test,y_train,y_test)

# %% [markdown]
# Training Xg Boost On the training set

# %%
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)

# %% [markdown]
# Making The confusion matrix

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns
y_pred=classifier.predict(x_test)
cm= confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm,annot=True,fmt="g")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion matrix")
plt.show()
plt.savefig("ConfusionMatrix.png")

# %% [markdown]
# Accuracy Score

# %%
accuracy_score(y_test,y_pred)

# %% [markdown]
# Classification Report

# %%
print(classification_report(y_test,y_pred))

# %% [markdown]
# Applying K fold Cross Validation

# %%
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
print("accuracies:", accuracies.mean())
print("Standard deviation: ",accuracies.std())


