# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:34:42 2020

@author: KhalidOM
"""


import pandas as pd
from sklearn.linear_model import LinearRegression







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
used_cars.to_csv("test9.csv")



# filter the attributes 
used_cars_filterd = pd.DataFrame()
used_cars_filterd["manufacturer"] = used_cars["manufacturer"]
used_cars_filterd["condition"] = used_cars["condition"]
used_cars_filterd["fuel"] = used_cars["fuel"]
used_cars_filterd["title_status"] = used_cars["title_status"]
used_cars_filterd["year"] = used_cars["year"]
used_cars_filterd["cylinders"] = used_cars["cylinders"]
used_cars_filterd["size"] = used_cars["size"]
used_cars_filterd["title_status"] = used_cars["title_status"]
used_cars_filterd["transmission"] = used_cars["transmission"]
used_cars_filterd["odometer"] = used_cars["odometer"]
used_cars_filterd["price"] = used_cars["price"]





# to convert the string to int , to analisis the data by grouping by numbers 
used_cars_filterd["manufacturer"] = replaceToHach(used_cars_filterd["manufacturer"])
used_cars_filterd["condition"] = replaceToHach(used_cars_filterd["condition"])
used_cars_filterd["fuel"] = replaceToHach(used_cars_filterd["fuel"])
used_cars_filterd["title_status"] = replaceToHach(used_cars_filterd["title_status"])
used_cars_filterd["transmission"] = replaceToHach(used_cars_filterd["transmission"])
used_cars_filterd["size"] = replaceToHach(used_cars_filterd["size"])




# the columns that price depends on, --- that effect the price 
column_names = ["year","manufacturer" ,"condition",
                "cylinders", "fuel","odometer", 
                "title_status", "transmission", "size"]

# add all the columns to X 
X = used_cars_filterd[column_names]
#add the odometer to Y , that we want to train and estimate it next time 
y = used_cars_filterd['price']

model = LinearRegression(fit_intercept=False)

# fit all the data that you want to graph it 
model.fit(X, y)

# add new coulumn to predict a odometer for --> Test the data train
used_cars_filterd['predicted'] = model.predict(X)


## plot it 
used_cars_filterd[['price', 'predicted']].plot()











