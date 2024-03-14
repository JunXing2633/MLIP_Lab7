import mlflow
import pandas as pd
import numpy as np

# TODO: Set tht MLFlow server uri
uri = "http://127.0.0.1:6001"
mlflow.set_tracking_uri(uri=uri)

# TODO: Provide model path/url
logged_model = 'runs:/b979008807bf48599a1962b42f20c87c/iris_model'

# mlflow models build-docker --model-uri 'runs:/b979008807bf48599a1962b42f20c87c/iris_model' --name "lab7"

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Input a random datapoint
# The model expects input data to be in the form of a pandas DataFrame with 4 columns representing the 4 features of the Iris dataset 
# (sepal length, sepal width, petal length, petal width).
data=np.array([[1.0,2.0,3.0,4.0]])

# TODO: Predict on a Pandas DataFrame. Due to the MLFlow functionality constrain.
#       The loaded model's predict function only accept dataframe as input instead of numpy array.
prediction=loaded_model.predict(pd.DataFrame(data))

# Print out prediction result
# The model outputs predicted class labels as integers: 0, 1, or 2, 
# representing the three classes of Iris flower species (Iris-setosa, Iris-versicolor, and Iris-virginica)
if prediction[0]==0:
    print("The predication is: Iris-setosa")
elif prediction[0]==1:
    print("The predication is: Iris-versicolor")
else:
    print("The predication is: Iris-virginica")