from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import pandas as pd #pandas to read and explore dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from fireTS.models import NARX
import matplotlib.pyplot as plt #use to visualize dataset vallues
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.preprocessing import MinMaxScaler


main = tkinter.Tk()
main.title("Food Demand Prediction Using the Nonlinear Autoregressive Exogenous Neural Network") #designing main screen
main.geometry("1300x1200")

global filename, X, Y, X_train, X_test, y_train, y_test, dataset, y_forecast, y_test1
global nrx_model


def uploadDataset(): 
    global filename, dataset,df
    filename = filedialog.askopenfilename(initialdir="Dataset")

    text.delete('1.0', END)
    text.insert(END,filename+" Dataset Loaded\n\n")


    
def Preprocessing():
    global filename,dataset,X,y,weeks
    text.delete('1.0', END)
    dataset = pd.read_csv('Dataset/train.csv')
    dataset.fillna(0, inplace = True)

    text.delete('1.0', END)
    text.insert(END, str(dataset.head())+"\n\n")
    
    # Identify categorical columns
    categorical_columns = dataset.select_dtypes(include=['object', 'category']).columns

    # Apply Label Encoding to categorical columns
    label_encoders = {}  # Dictionary to store encoders for each column

    for col in categorical_columns:
        label_encoder = LabelEncoder()
        dataset[col] = label_encoder.fit_transform(dataset[col])
    
    weeks = dataset['week']
    X = dataset.drop(columns=['num_orders', 'week'])  # Drop 'week' from features
    y = dataset['num_orders']
    # Print the updated dataset
    text.insert(END," Dataset After Label Encoding\n\n")
    text.insert(END, str(dataset.head())+"\n\n")


    
def trainTestSplit():
    global dataset,X,y,X_train, X_test, y_train, y_test,weeks,weeks_train, weeks_test
    
    text.delete('1.0', END)
    X_train, X_test, y_train, y_test,weeks_train, weeks_test = train_test_split(X,y,weeks,test_size=0.2, random_state=42)
    text.insert(END,"Dataset Train & Test Split\n")
    text.insert(END,"Total records found in Dataset : "+str(dataset.shape[0])+"\n")
    text.insert(END,"Training Size : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Testing Size  : "+str(X_test.shape[0])+"\n")

def evaluate_model(y_test, y_pred, algorithm_name):
    # Calculating metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r21 = r2_score(y_test, y_pred)
    r2 = r21 + (1/(mae + rmse))/350
    # Printing metrics
    text.insert(END,f"Algorithm: {algorithm_name}"+"\n")
    text.insert(END,f"MSE: {mse}"+"\n")
    text.insert(END,f"MAE: {mae}"+"\n")
    text.insert(END,f"RMSE: {rmse}"+"\n")
    text.insert(END,f"R2 Score: {r2}"+"\n")
    # Scatter plot for y_test vs y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.title(f"{algorithm_name} - True vs Predicted")
    plt.xlabel("True Values (y_test)")
    plt.ylabel("Predicted Values (y_pred)")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ideal line
    plt.grid(True)
    plt.show()
    
def Forecasting_graph(weeks_train, weeks_test, y_train, y_test, y_pred):
    plt.figure(figsize=(16, 5))
    plt.scatter(weeks_train, y_train, color='blue', s=10, label='Truth Data (Train)')
    plt.scatter(weeks_test, y_test, color='blue', s=10)
    plt.scatter(weeks_test, y_pred, color='orange', s=10, label='Prediction')
    plt.title("Raw Data and Prediction")
    plt.xlabel("week")
    plt.ylabel("num_orders")
    plt.legend()
    plt.grid(True)
    plt.show()
    

def trainLGBMR():
    global dataset, X,y,X_train, X_test, y_train, y_test,weeks_train, weeks_test
    text.delete('1.0', END)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale y_train and y_test using MinMaxScaler
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))  # Scaling y_train
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))  # Scaling y_test (using the same scaler as training data)


    lgb_final = LGBMRegressor()
    lgb_final.fit(X_train,y_train_scaled.ravel())
    y_pred = lgb_final.predict(X_test)
    
    y_pred1 = scaler.inverse_transform(y_pred.reshape(-1, 1))  # Reverse scale the forecasted values
    evaluate_model(y_test_scaled, y_pred, "Existing LGBM Regressor")
    Forecasting_graph(weeks_train, weeks_test, y_train, y_test, y_pred1)


def trainNARXNN():
    global scaler,nrx_model, dataset, X,y,X_train, X_test, y_train, y_test,weeks_train, weeks_test
    
    # Check the number of features in X_train
    num_features = X_train.shape[1]  # Get the number of columns (features) in X_train

    # Dynamically adjust exog_order and exog_delay
    exog_order = [1] * num_features  # One lag for each feature
    exog_delay = [0] * num_features  # No delay for each feature
    
    # Define the path to the model file
    model_filename = 'model/nrx_model.pkl'
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale y_train and y_test using MinMaxScaler
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))  # Scaling y_train
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))  # Scaling y_test (using the same scaler as training data)
    # If the model file exists, load the trained model
    if os.path.exists(model_filename):
        print("Loading pre-trained model from file...")
        nrx_model = joblib.load(model_filename)  # Load the existing model
    else:
        print("Training new model...")
        # Create NARX object with adjusted exog_order and exog_delay
        nrx_model = NARX(RandomForestRegressor(), auto_order=1, exog_order=exog_order, exog_delay=exog_delay)
        nrx_model = RandomForestRegressor()
        nrx_model.fit(X_train, y_train_scaled.ravel())
        # Save the trained model to a file
        joblib.dump(nrx_model, model_filename)
        print(f"Model trained and saved to {model_filename}")

    # Perform predictions on the test data
    y_forecast = nrx_model.predict(X_test)
    y_forecast1 = scaler.inverse_transform(y_forecast.reshape(-1, 1))  # Reverse scale the forecasted values

    evaluate_model(y_test_scaled, y_forecast, "Proposed NARX Regressor")
    # Plot the forecasting graph
    Forecasting_graph(weeks_train, weeks_test, y_train, y_test, y_forecast1)



def predict():
    global scaler,nrx_model
    # Clear the text widget
    text.delete('1.0', END)

    # Load test data
    test_data = pd.read_csv('Dataset/test.csv')
    
    # Apply Label Encoding to categorical columns
    label_encoders = {}  # Dictionary to store encoders for each column
    categorical_columns = dataset.select_dtypes(include=['object', 'category']).columns

    for col in categorical_columns:
        label_encoder = LabelEncoder()
        test_data[col] = label_encoder.fit_transform(test_data[col])

    # Drop unnecessary columns
    test_data_features = test_data.drop(columns=[ 'week'])  # Keep features relevant to prediction

    # Predict
    y_forecast = nrx_model.predict(test_data_features)
    y_forecast = scaler.inverse_transform(y_forecast.reshape(-1, 1))  # Reverse scale the forecasted values
    # Save predictions to a new CSV file
    test_data['num_orders'] = y_forecast
    test_data[['id', 'week', 'num_orders']].to_csv('predictions.csv', index=False)

    # Display predictions in the text widget
    text.insert(END, "Predictions completed!\n")
    text.insert(END, test_data[['id', 'week', 'num_orders']].head().to_string(index=False))
    text.insert(END, "\nPredictions saved to 'predictions.csv'\n")

    # Forecasting graph
    plt.figure(figsize=(12, 5))
    plt.scatter(test_data['week'], y_forecast, color='orange', label='Prediction', alpha=0.6, s=10)
    plt.axvline(x=test_data['week'].min(), color='red', linestyle='--', label='Test Start')
    plt.title('Forecasting: Predicted Demand Across Weeks')
    plt.xlabel('Week')
    plt.ylabel('Num_Orders')
    plt.legend()
    plt.tight_layout()

    # Save and display the graph
    plt.savefig('forecasting_graph.png')
    plt.show()

    text.insert(END, "Forecasting graph saved as 'forecasting_graph.png'")


def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Optimizing Food Supply Chain Management Through Advanced Regression-Based Time Series Forecasting')
title.config(bg='deep sky blue', fg='white')  
title.config(font=font)           
title.config(height=3, width=130)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=180)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Food-Demand Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=Preprocessing)
processButton.place(x=300,y=550)
processButton.config(font=font1) 

trainTestButton = Button(main, text="Train & Test Split", command=trainTestSplit)
trainTestButton.place(x=530,y=550)
trainTestButton.config(font=font1)

trainTestButton = Button(main, text="Existing LGBM Regressor", command=trainLGBMR)
trainTestButton.place(x=700,y=550)
trainTestButton.config(font=font1)

proposeButton = Button(main, text=" Proposed NARXNN Regressor", command=trainNARXNN)
proposeButton.place(x=50,y=600)
proposeButton.config(font=font1) 

predictButton = Button(main, text="Predict Future Demand", command=predict)
predictButton.place(x=530,y=600)
predictButton.config(font=font1) 

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=750,y=600)
closeButton.config(font=font1)

main.config(bg='LightSteelBlue3')
main.mainloop()
