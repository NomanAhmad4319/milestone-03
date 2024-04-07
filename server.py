import tensorflow as tf
from tensorflow import keras 
import os
import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

base_path = os.getcwd()
dataset_columns = pd.read_csv(f'{base_path}/MachineLearningCVE/NUSW-NB15_features.csv',encoding='ISO-8859-1')
path = f'{base_path}/MachineLearningCVE/UNSW-NB15_1.csv'  # There are 4 input csv files
data = pd.read_csv(path, header = None)
data = data.reset_index(drop=True)
data.columns = dataset_columns['Name']



from flask import Flask ,request, jsonify

from flask_restful import Resource,Api

from flask_cors import CORS



from dotenv import load_dotenv




app = Flask(__name__)
origins  = ['http://localhost:8000','http://localhost:3000' , 'http://localhost:5100' , 'http://localhost:7000' , 'http://localhost:5000']
CORS(app, origins=origins)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1  # Disable caching

api = Api(app)



load_dotenv()


class Predict(Resource):
    def post(self):
        try:
            data = None
            count = int(request.get_json()['file_count'])
            if count <= 4 and count >=1 :
                print("INside the if_else part")
                
            
                predictions = Predict_Data(file_count=count)
            else:
                return jsonify ({"message": "count must be in the range of 1-5"})
        except Exception as e :
            return jsonify({"message": "File not found with request body"})
       
        return jsonify({"Predictions":predictions})





api.add_resource(Predict, '/callmodal' )




if __name__ == "__main__":
    app.run(debug=True)
       
       
       
       
       
       
def Predict_Data(file_count):
    base_path = os.getcwd()
    dataset_columns = pd.read_csv(f'{base_path}/MachineLearningCVE/NUSW-NB15_features.csv',encoding='ISO-8859-1')
    path = f'{base_path}/MachineLearningCVE/UNSW-NB15_{file_count}.csv'  # There are 4 input csv files
    data = pd.read_csv(path, header = None)
    data = data.reset_index(drop=True)
    data.columns = dataset_columns['Name']
    data_processed = processData(data)
    checkpoint_path = f'{base_path}/IDS_cnn_lstm.keras'
    lstm_cnn  = tf.keras.models.load_model(filepath=checkpoint_path)
    prediction = lstm_cnn.predict(data_processed)
    resp = np.argmax(prediction,  axis=1)
    
    return resp

def PreProcess(input_data):
    
    
    
    
    
    
    input_data['attack_cat'] = input_data['attack_cat'].fillna(value='normal').apply(lambda x: x.strip().lower())
    input_data['attack_cat'] = input_data['attack_cat'].replace('backdoors','backdoor', regex=True).apply(lambda x: x.strip().lower())
    input_data['ct_ftp_cmd'] = input_data['ct_ftp_cmd'].replace(to_replace=' ', value=0).astype(int)
    input_data['is_ftp_login'] = input_data['is_ftp_login'].fillna(value=0)
    input_data['is_ftp_login'] = np.where(input_data['is_ftp_login']>1, 1, input_data['is_ftp_login'])
    input_data['service'] = input_data['service'].apply(lambda x:"None" if x=='-' else x)
    input_data['ct_ftp_cmd'] = input_data['ct_ftp_cmd'].replace(to_replace=' ', value=0).astype(int)
    input_data['ct_flw_http_mthd'] = input_data['ct_flw_http_mthd'].fillna(value=0)
    input_data['is_ftp_login'] = input_data['is_ftp_login'].fillna(value=0)
    input_data['is_ftp_login'] = np.where(input_data['is_ftp_login']>1, 1, input_data['is_ftp_login'])
    input_data['service'] = input_data['service'].apply(lambda x:"None" if x=='-' else x)
    input_data['ct_ftp_cmd'] = input_data['ct_ftp_cmd'].replace(to_replace=' ', value=0).astype(int)
    input_data['ct_flw_http_mthd'] = input_data['ct_flw_http_mthd'].fillna(value=0)
    input_data['ct_ftp_cmd'] = input_data['ct_ftp_cmd'].replace(to_replace=' ', value=0).astype(int)
    input_data['is_ftp_login'] = input_data['is_ftp_login'].fillna(value=0)
    input_data.drop(columns=['srcip','sport','dstip','dsport','Label',],inplace=True)
    
    return input_data




def processData(input_data):
    #--------------------------------
    base_path = os.getcwd()
    dfs = []
    for f in range(1,5):

        path = f'{base_path}/MachineLearningCVE/UNSW-NB15_{f}.csv'  # There are 4 input csv files
        dfs.append(pd.read_csv(path, header = None))
    combined_data = pd.concat(dfs).reset_index(drop=True)
    dataset_columns = pd.read_csv(f'{base_path}/MachineLearningCVE/NUSW-NB15_features.csv',encoding='ISO-8859-1')
    combined_data.columns = dataset_columns['Name']
    combined_data = PreProcess(input_data= combined_data)
    
    x_train, y_train = combined_data.drop(columns=['attack_cat']), combined_data[['attack_cat']]
    
    print("\n############", x_train.columns, "\n##############")
    cat_col = ['proto', 'service', 'state']
    num_col = list(set(x_train.columns) - set(cat_col))
    
    
    
    scaler = StandardScaler()
    scaler = scaler.fit(x_train[num_col])
    # combined_data[num_col] = scaler.transform(combined_data[num_col])
    # x_test[num_col] = scaler.transform(x_test[num_col])
    # x_val[num_col] = scaler.transform(x_val[num_col])
    
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), cat_col)], remainder='passthrough')
    x_train = np.array(ct.fit_transform(x_train))
    print(x_train.shape,"\n***********")
    # x_test = np.array(ct.transform(x_test))
    # x_val = np.array(ct.transform(x_val))
    # ------------------------------------
    input_data = PreProcess(input_data=input_data)
    x_return, y_train = input_data.drop(columns=['attack_cat']), input_data[['attack_cat']]
    print("\n############", x_return.columns, "\n##############")
  
    
    x_return[num_col] = scaler.transform(input_data[num_col])
    x_return = np.array(ct.transform(input_data))
    print(x_return.shape, "\n******************")
    del(scaler)
    del(ct)
    del(combined_data)
    return x_return