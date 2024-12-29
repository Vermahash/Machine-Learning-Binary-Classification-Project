### Libraries
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import Perceptron
from sklearn import linear_model
import time


### Seaborn Styling and console settings
sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


### Function calls - Wrappers for loading files and models
def load_data(training_file):
    train_data = pd.read_csv(training_file, index_col=False)
    return train_data

def preprocess_data(train_data, target_column):
    train_data = train_data.dropna()
    train_data = train_data.map(lambda x: x.strip() if isinstance(x, str) else x)
    train_data = train_data.map(lambda x: float(x) if isinstance(x, str) else x)

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_train, X_test, y_train, Y_test = train_test_split(X_train, y_train,
                                                        test_size=0.3, random_state=42, stratify=y_train)
    return X_train, y_train, X_test, Y_test

def train_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2,_min_samples_leaf=2):
    start_time = time.time()

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42, min_samples_leaf=_min_samples_leaf)
    model.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    return model, elapsed_time

def train_Perceptron(X_train, y_train, _max_iter=100, _eta=0.1, _early_stopping=False):
    start_time = time.time()

    perceptron_model = Perceptron(max_iter=_max_iter, eta0=_eta, early_stopping=_early_stopping,
                                  random_state=42, shuffle=False)
    perceptron_model.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    return perceptron_model, elapsed_time

def train_KNN(X_train, y_train, _n_neighbors=3, _leafsize = 30):
    start_time = time.time()

    knn = KNeighborsClassifier(n_neighbors=_n_neighbors,leaf_size=_leafsize)
    knn.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    return knn, elapsed_time

def train_LogRegression(X_train, y_train, _max_iter=100, _solver='newton-cg'):
    start_time = time.time()

    logr = linear_model.LogisticRegression(max_iter=_max_iter, solver=_solver, random_state=42)
    logr.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    return logr, elapsed_time

def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return f1, r2, accuracy, report

def plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

def plot_learning_curve_avg(model, X_train, y_train, title='Performance Avg'):
    plt.figure(figsize=(8, 6))

    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, n_jobs=-1,
                                                           train_sizes=np.linspace(0.1, 1.0, 10),
                                                           cv=5, scoring='f1')
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, label="Training score", color="blue")
    plt.plot(train_sizes, val_mean, label="Validation score", color="grey")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color="grey", alpha=0.2)

    plt.xlabel('Training Size')
    plt.ylabel('F1 Score')
    plt.title(f'Validation Curve: ' + title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_learning_curve(model, X_train, y_train, title='Performance', model2=None, model3=None):
    plt.figure(figsize=(8, 6))

    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, n_jobs=-1,
                                                           train_sizes=np.linspace(0.1, 1.0, 10),
                                                           cv=5, scoring='f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, label='Training', color='blue', linestyle='dashed')
    plt.plot(train_sizes, val_scores_mean, label='Test', color='grey')

    if model2 is not None:
        train_sizes, train_scores, val_scores = learning_curve(model2, X_train, y_train, n_jobs=-1,
                                                               train_sizes=np.linspace(0.1, 1.0, 10),
                                                               cv=5, scoring='f1')
        train_scores_mean = np.mean(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        plt.plot(train_sizes, train_scores_mean, label='Model 2 Training', color='green', linestyle='dashed')
        plt.plot(train_sizes, val_scores_mean, label='Model 2 Test', color='purple')

    if model3 is not None:
        train_sizes, train_scores, val_scores = learning_curve(model3, X_train, y_train, n_jobs=-1,
                                                               train_sizes=np.linspace(0.1, 1.0, 10),
                                                               cv=5, scoring='f1')
        train_scores_mean = np.mean(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        plt.plot(train_sizes, train_scores_mean, label='Model 3 Training', color='orange', linestyle='dashed')
        plt.plot(train_sizes, val_scores_mean, label='Model 3 Test', color='red')

    plt.xlabel('Training Size')
    plt.ylabel('F1 Score')
    plt.title(f'Validation Curve: ' + title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_feature_importance(model, subtitle=""):
    feature_importance_df = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    })

    feature_importance_df = feature_importance_df[feature_importance_df['Importance'] != 0]

    feature_importance_df = feature_importance_df.sort_values('Importance')

    plt.figure(figsize=(15, 10))

    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Name')
    plt.title('Feature Importance: Decision Tree ' + subtitle)

    plt.grid(axis='x')
    plt.show()

    print(feature_importance_df)


### Main with Parameter to choose what file of data to load
def main(dataLoad='CarClaimable'):


    if dataLoad == 'CarClaimable':
        print("\r\nCar Claimable Dataset\r\n")

        # Load the data
        training_file = 'carpolicy_TrainingTesting.csv'  # Data path
        target_column = 'is_claim'  # Target column name
        trainingTest_data = load_data(training_file)


    else:
        print("\r\n Client Subscribed A Term Deposit Dataset\r\n")

        # Load the data
        training_file = 'bank_TrainingTesting.csv'  # Data Path
        target_column = 'y'  # Target column name
        trainingTest_data = load_data(training_file)


    X_train, y_train, X_test, y_test = preprocess_data(trainingTest_data, target_column)

    # Decision Tree Testing
    print("\r\nDecision Tree Testing")


    model, elapsed_time = train_decision_tree(X_train, y_train, max_depth=5,
                                              min_samples_split=20,_min_samples_leaf=1)
    y_pred = model.predict(X_test)

    model2, elapsed_time2 = train_decision_tree(X_train, y_train, max_depth=3,
                                              min_samples_split=8,_min_samples_leaf=1)
    y_pred2 = model2.predict(X_test)

    model3, elapsed_time3 = train_decision_tree(X_train, y_train, max_depth=7,
                                                min_samples_split=15,_min_samples_leaf=1)
    y_pred3 = model3.predict(X_test)


    print("DTM1 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred)
    print(f"Processing Time: {elapsed_time:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    plot_feature_importance(model, "M1")


    print("DTM2 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred2)
    print(f"Processing Time: {elapsed_time2:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    plot_feature_importance(model2, "M2")

    print("DTM3 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred3)
    print(f"Processing Time: {elapsed_time3:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    plot_feature_importance(model3, "M3")


    plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix - Decision Tree M1')
    plot_confusion_matrix(y_test, y_pred2, 'Confusion Matrix - Decision Tree M2')
    plot_confusion_matrix(y_test, y_pred3, 'Confusion Matrix - Decision Tree M3')

    plot_learning_curve(model, X_train, y_train, 'Decision Tree Performance', model2, model3)

    plot_learning_curve_avg(model, X_train, y_train, 'Decision Tree Performance Confidence M1')
    plot_learning_curve_avg(model2, X_train, y_train, 'Decision Tree Performance Confidence M2')
    plot_learning_curve_avg(model3, X_train, y_train, 'Decision Tree Performance Confidence M3')

    # Perceptron
    print("\r\nPerceptron Testing")


    model, elapsed_time = train_Perceptron(X_train, y_train, _max_iter=100, _eta=0.1, _early_stopping=False)
    model2, elapsed_time2 = train_Perceptron(X_train, y_train, _max_iter=1000, _eta=0.1, _early_stopping=False)
    model3, elapsed_time3 = train_Perceptron(X_train, y_train, _max_iter=100, _eta=0.01, _early_stopping=True)

    y_pred = model.predict(X_test)
    y_pred2 = model2.predict(X_test)
    y_pred3 = model3.predict(X_test)

    # Report
    print("PM1 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred)
    print(f"Processing Time: {elapsed_time:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(report)

    print("PM2 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred2)
    print(f"Processing Time: {elapsed_time2:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(report)

    print("PM3 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred3)
    print(f"Processing Time: {elapsed_time3:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(report)


    plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix - Perceptron M1')
    plot_confusion_matrix(y_test, y_pred2, 'Confusion Matrix - Perceptron M2')
    plot_confusion_matrix(y_test, y_pred3, 'Confusion Matrix - Perceptron M3')

    plot_learning_curve(model, X_train, y_train, 'Perceptron Performance',model2, model3)

    plot_learning_curve_avg(model, X_train, y_train, 'Perceptron Performance Confidence M1')
    plot_learning_curve_avg(model2, X_train, y_train, 'Perceptron Performance Confidence M2')
    plot_learning_curve_avg(model3, X_train, y_train, 'Perceptron Performance Confidence M3')


    # KNeighborsClassifier
    print("\r\nKNN Testing")

    model, elapsed_time = train_KNN(X_train, y_train, _n_neighbors=3)
    model2, elapsed_time2 = train_KNN(X_train, y_train, _n_neighbors=6)
    model3, elapsed_time3 = train_KNN(X_train, y_train, _n_neighbors=4, _leafsize=15)

    y_pred = model.predict(X_test)
    y_pred2 = model2.predict(X_test)
    y_pred3 = model3.predict(X_test)

    print("KNNM1 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred)
    print(f"Processing Time: {elapsed_time:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    print("KNNM2 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred2)
    print(f"Processing Time: {elapsed_time2:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    print("KNNM3 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred3)
    print(f"Processing Time: {elapsed_time3:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix - KNN M1')
    plot_confusion_matrix(y_test, y_pred2, 'Confusion Matrix - KNN M2')
    plot_confusion_matrix(y_test, y_pred3, 'Confusion Matrix - KNN M3')

    plot_learning_curve(model, X_train, y_train, 'KNN Performance M1',model2, model3)

    plot_learning_curve_avg(model, X_train, y_train, 'KNN Performance Confidence M1')
    plot_learning_curve_avg(model2, X_train, y_train, 'KNN Performance Confidence M2')
    plot_learning_curve_avg(model3, X_train, y_train, 'KNN Performance Confidence M3')


    # Logistic Regression
    print("\r\nLogistic Regression Testing")

    model, elapsed_time = train_LogRegression(X_train, y_train, _max_iter=400, _solver='newton-cg')
    model2, elapsed_time2 = train_LogRegression(X_train, y_train, _max_iter=800, _solver='newton-cg')
    model3, elapsed_time3 = train_LogRegression(X_train, y_train, _max_iter=800, _solver='newton-cg')

    y_pred = model.predict(X_test)
    y_pred2 = model2.predict(X_test)
    y_pred3 = model3.predict(X_test)

    print("LRM1 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred)
    print(f"Processing Time: {elapsed_time:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    print("LRM2 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred2)
    print(f"Processing Time: {elapsed_time2:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    print("LRM3 Report")
    f1, r2, accuracy, report = evaluate_model(y_test, y_pred3)
    print(f"Processing Time: {elapsed_time3:.2f} seconds")
    print(f"F1 Score: {f1:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix - Logistic Regression M1')
    plot_confusion_matrix(y_test, y_pred2, 'Confusion Matrix - Logistic Regression M2')
    plot_confusion_matrix(y_test, y_pred3, 'Confusion Matrix - Logistic Regression M3')

    plot_learning_curve(model, X_train, y_train, 'Logistic Regression Performance',model2, model3)

    plot_learning_curve_avg(model, X_train, y_train, 'Logistic Regression Performance M1')
    plot_learning_curve_avg(model2, X_train, y_train, 'Logistic Regression Performance M2')
    plot_learning_curve_avg(model3, X_train, y_train, 'Logistic Regression Performance M3')




if __name__ == "__main__":
    main('CarClaimable')
    main('MarketingDeposit')

