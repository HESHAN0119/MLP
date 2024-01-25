#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[ ]:


df=pd.read_csv("pima-indians-diabetes.csv")
df.shape
df=pd.read_csv("pima-indians-diabetes.csv")
dataset=df.values
x=dataset[:,0:8] 	#need to get the dataset first as 'dataset', need to have 9 columns in the data set but we have to get 8 columns
y=dataset[:,8] 
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42)
y_test


# In[ ]:


#model architecture
model=Sequential()
model.add(Dense(units=12, activation="relu", input_shape=(8,) )) 	#input shape is equals to number of columns in X
model.add(Dense(units=1, activation="sigmoid")) #if the Y values is a regression then the activation ='linear' (regression means counting values) ,units =1


# In[ ]:


#compile model
optimizer=Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) 		#if regression then  loss = 'mean_squared_error' , metrics=['mean_absolute_error']
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
import matplotlib.pyplot as plt


# In[ ]:


# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
results = model.evaluate(x_test,y_test)
print("loss", results[0])
print("accuracy", results[1])


# In[ ]:


# Make prediction on the test set
prediction = model.predict(x_train)


# In[ ]:


# Check if the predicted probability is greater than the 0.5
predicted_labels = 1 if prediction[0][0] > 0.5 else 0


# In[ ]:


# Print the predicted class label
print(predicted_labels)

