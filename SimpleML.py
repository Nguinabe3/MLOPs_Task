from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

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

def train(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], df["target"], test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def get_accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

def plot_feature(df, feature):
    # Plot a histogram of one of the features
    df[feature].hist()
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()


def plot_features(df):
    # Plot scatter plot of first two features.
    scatter = plt.scatter(
        df["sepal length (cm)"], df["sepal width (cm)"], c=df["target"]
    )
    plt.title("Scatter plot of the sepal features (width vs length)")
    plt.xlabel(xlabel="sepal length (cm)")
    plt.ylabel(ylabel="sepal width (cm)")
    plt.legend(
        scatter.legend_elements()[0],
        df["target_names"].unique(),
        loc="lower right",
        title="Classes",
    )
    plt.show()


def plot_model(model, X_test, y_test):
    # Plot the confusion matrix for the model
    ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    df=load_dataset()
    model, X_train, X_test, y_train, y_test = train(df)
    accuracy = get_accuracy(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    plot_feature(df, "sepal length (cm)")
    plot_features(df)
    plot_model(model, X_test, y_test)