import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df=pd.read_csv(path)

    df["Gender"]=df["Gender"].map({
        "Male": 0,
        "Female": 1
    })

    X=df.drop("Depression",axis=1)
    Y=df["Depression"]

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    return X_train,X_test,Y_train,Y_test