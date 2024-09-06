from sklearn.datasets import load_iris
import pandas as pd

def load_dataset():
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Print the first few rows of the dataset
    df['target']=iris.target
    target_names = {
        0:'setosa',
        1:'versicolor',
        2:'virginica'
    }
    df['target_names'] = df['target'].map(target_names)
    print(df.head())
    return df
if __name__ == "__main__":
    load_dataset()