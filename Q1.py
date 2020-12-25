# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:34:42 2020

@author: KhalidOM
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




# Function that replace every string --> to its hach code , to use it in LinearRegression 
def replaceToHach(column1):

    column2 = list(column1)
    for i, value in  enumerate(column2):
        column2[i] = abs(hash(value)) 
        #column2[i] = abs(hash(value)) - abs(hash(value)) * 0.9999999999999999
    
        
        
    return column2
            







#data sample 1000 record , (Read the file)
used_cars = pd.read_csv("used_cars.csv")

# Drop any rows with null values
used_cars = used_cars.drop(columns=['url', 'id','region_url','county','vin'])
used_cars = used_cars.dropna(subset=['year','condition','model','manufacturer',"drive", "size", "type", "cylinders","odometer"])




# to convert the string to int , to analisis the data by grouping by numbers 
used_cars["model"] = replaceToHach(used_cars["model"])
used_cars["fuel"] = replaceToHach(used_cars["fuel"])
used_cars["manufacturer"] = replaceToHach(used_cars["manufacturer"])







# filter the attributes 
used_cars_filterd = pd.DataFrame()
used_cars_filterd["manufacturer"] = used_cars["manufacturer"]
used_cars_filterd["year"] = used_cars["year"]
used_cars_filterd["model"] = used_cars["model"]
used_cars_filterd["cylinders"] = used_cars["cylinders"]
used_cars_filterd["fuel"] = used_cars["fuel"]
used_cars_filterd["price"] = used_cars["price"]
used_cars_filterd["best_car"] = used_cars["best_car"]

# add all the columns to X 
X = used_cars_filterd[['year','cylinders', "fuel","manufacturer", "price"]]
#add the model to Y , that we want to train it , to see it if it the best or not  
y = used_cars_filterd['best_car']

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 1)

# fit into the model.
# make the predictions.
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
print("x test is", len(X_test))
predictions = model.predict(X_test)



plt.plot(X, y, 'ro')
#plt.plot(predictions,predictions)


meanSquaredError = 0

# for all values that predicted and not predicted 
for nonPredictedValue , predictedValue in zip(y_test, predictions):
    meanSquaredError = meanSquaredError + (nonPredictedValue - predictedValue)**2

# calcualted mean squared error 
meanSquaredError = (1/ len(predictions))* meanSquaredError

print("The mean squared Error is: ", meanSquaredError)







