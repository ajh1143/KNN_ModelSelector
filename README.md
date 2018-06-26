# KNN_ModelSelector
Tests, Trains, and Plots results with varying neighbor values for model selection

## Import Libraries
```Python3
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Set Plot Style
```Python3
plt.style.use('ggplot')
```

## Load File
```Python3
def loadFile(path):
    #Load Excel File into Pandas DataFrame
    df = pd.read_excel(path)
    return df
```

## Prelimnary Exploratory Data Analysis
```Python3
def minorEDA(df):
    lineBreak = '------------------'

    #Check Shape
    print(lineBreak*3)
    print("Shape:")
    print(df.shape)
    print(lineBreak*3)
    #Check Feature Names
    print("Column Names")
    print(df.columns)
    print(lineBreak*3)
    #Check types, missing, memory
    print("Data Types, Missing Data, Memory")
    print(df.info())
    print(lineBreak*3)
```

## Feature Selection
```Python3
def feature(feature, df):
    # Create arrays for the features and the response variable
    y = df[feature]
    X = df.drop(feature, axis=1)
    return X, y
```

## Test, Train, Plot
```Python3
def TestTrainFitPlot(X, y):
    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42, stratify=y)

    # Try KNN with 5 neighbors
    knn = KNeighborsClassifier()

    # Fit training data
    knn.fit(X_train, y_train)

    #Cneck Accuracy Score
    print('Default Accuracy: {}'.format(round(knn.score(X_test, y_test), 3)))
    # Enum Loop, accuracy results using range on 'n' values for KNN Classifier
    for acc, n in enumerate(neighbors):
        # Try KNeighbors with each of 'n' neighbors
        knn = KNeighborsClassifier(n_neighbors=n)

        # Fitting
        knn.fit(X_train, y_train)

        # Training Accuracy
        train_accuracy[acc] = knn.score(X_train, y_train)

        # Testing Accuracy
        test_accuracy[acc] = knn.score(X_test, y_test)

    #Plotting
    #Set Main Title
    plt.title('KNN Neighbors')
    #Set X-Axis Label
    plt.xlabel('Neighbors\n(#)')
    #Set Y-Axis Label
    plt.ylabel('Accuracy\n(%)', rotation=0, labelpad=35)
    #Place Testing Accuracy
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    #Place Training Accuracy
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    #Append Labels on Testing Accuracy
    for a,b in zip(neighbors, test_accuracy):
        plt.text(a, b, str(round(b,2)))
    #Add Legend
    plt.legend()
    #Generate Plot
    plt.show()
```
# Sample Output(UCI ML Repo, Absenteeism Dataset):
### EDA
<img src="https://github.com/ajh1143/blob/KNN_ModelSelector/images/EDA.png" class="inline"/><br>
### Default Accuracy n-neighbors
<img src="https://github.com/ajh1143/blob/KNN_ModelSelector/images/acc.png" class="inline"/><br>
### Checking Multiple n-neighbors
<img src="https://github.com/ajh1143/blob/KNN_ModelSelector/images/myplot.png" class="inline"/><br>
