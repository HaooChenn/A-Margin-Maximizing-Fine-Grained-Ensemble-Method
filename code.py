# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:57:37 2024

@author: lenovo
"""
#%%Imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import torch
import torch.optim as optim
from lightgbm import LGBMClassifier
import sklearn
import random
import pandas as pd
import os
import scipy.io as sio
from scipy import sparse


# NOTE: The confidence in Θ refers to the accuracy of the classifier, 
#       its shape is (c × k);
# he elements in Gj(x_i) are the probabilities of classifier Gj predicting x_i as each category
#%%Functions
def Train(X, y, k, deep = 10):
    """
    Train k base classifiers

    Parameters:
    X: Feature data
    y: Label data
    k: Number of base classifiers
    deep: Maximum depth of base classifiers

    Returns:
    classifiers: List of trained base classifiers
    G: Prediction result matrix
    accuracies: List of accuracies for base classifiers
    """

    # deep is the maximum depth
    c = len(np.unique(y))  # Number of classes
    n = len(X)

    # Initialize k classifiers and matrix G
    classifiers = []
    
    # NOTE: G is not mentioned in the paper
    # Initialize G matrix, kc rows, n columns, k classifiers, each classifier corresponds to c-dimensional one-hot encoded vector
    G = np.zeros((k * c, n))  
    
    # NOTE: Each column in G corresponds to a sample, every c rows correspond to a classifier's prediction 
    # The i-th column of G = [G1(x_i) -> c dimensional column vector
    #                            ,
    #                         G2(x_i) -> c dimensional column vector
    #                            ,
    #                           ...
    #                            ,
    #                         Gk(x_i) -> c dimensional column vector]
    accuracies = []

    # Train k classifiers with different parameter settings
    for i in range(k):
        # Part1 Bootstrap Sampling
        # Use bootstrap sampling to create different subsamples
        size = round(len(X)*0.95)       # Extract a subsample set of 95% size of the original dataset

        # Randomly sample size indices with replacement from the original dataset to form the subsample index list bootstrap_indices.
        # replace = True: Indicates sampling with replacement, meaning that after each draw, 
        #                 the sample is put back into the sampling pool and may be drawn again. 
        #                 This allows for repeated elements in the bootstrap_indices list.
        bootstrap_indices = np.random.choice(len(X), size = size, replace = True)

        # Extract corresponding samples and labels from the original dataset X and label set y to form a new subsample set.
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]

        # Part2 Training Classifiers
        # Choose different classifiers based on the remainder of i
        classifier = DecisionTreeClassifier(max_depth = deep, random_state=1)

        # Train the classifier
        classifier.fit(X_bootstrap, y_bootstrap)
        classifiers.append(classifier)

        # Predict results
        y_pred = classifier.predict(X) - 1

        # Part3 Constructing G Matrix
        # c rows, n columns: classifier.predict_proba(X) produces a matrix of shape (n,c), 
        #                    where each row is the prediction probabilities of a sample for c categories.
        # NOTE: The elements in G are the prediction probabilities of samples for each category
        G[i*c: i*c+c, :] = classifier.predict_proba(X).T

        #  Calculate the accuracy of the classifier
        accuracy = classifier.score(X, y)
        accuracies.append(accuracy)
        # print(f"Accuracy of base classifier {i+1}: {accuracy:.2f}")

    g = []
    for j in range(n):
        g_j = G[:, j:j+1].reshape(k, c)
        g.append(g_j)

    g = np.hstack(g)

    # Return all base classifiers, matrix G, and accuracy list
    return classifiers, g, accuracies


def predict_g(classifiers, X, c):
    k = len(classifiers)
    n = len(X)      # Number of samples

    g = np.zeros((k, n * c))  # Initialize G matrix, kc rows, n columns

    # Calculate prediction results for each base classifier and update matrix G
    for i, clf in enumerate(classifiers):
        y_pred = clf.predict(X) - 1
        g[i, :] = clf.predict_proba(X).reshape(1, -1)

    return g


def one_hot(y):
    y = y.astype(int)
    """
    Convert the label vector y to a one-hot encoded matrix Y

    Parameters:
    y : numpy array, shape (n,), label vector containing class labels for each sample (from 1 to c)

    Returns:
    Y : numpy array, shape (c, n), each column is the one-hot encoding of the corresponding element in y
    """
    n = len(y)  # Number of samples
    c = int(np.max(y))  # Number of classes, assuming class labels start from 1 to c

    Y = np.zeros((c, n))  # NOTE: Samples are placed column by column, rows represent classes, columns represent samples

    for i in range(n):
        # The label of the i-th sample is y[i], the index corresponding to label y[i] is y[i]-1
        Y[y[i]-1 , i] = 1  # Labels start from 1, so the index corresponding to label y[i] is y[i]-1

    return Y


def init_Theta(accuracies,c):
    """ Theta's shape is c × k, each classifier corresponds to a column (k columns in total), each class corresponds to a row, representing the confidence of a classifier classifying a sample into that class.
      During initialization, the confidence of a classifier classifying a sample into each class is initialized to the accuracy of that classifier."""
    # Replicate each element in the vector c times
    # replicated = np.repeat(accuracies, c)

    # Replicate accuracies (corresponding to a row of k classifier accuracies) c times and stack vertically
    stacked = np.tile(accuracies, (c, 1))

    return stacked


def compute_S_I_Theta_g_1(Theta, g):
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g, dtype=torch.float)
    Theta_g = torch.matmul(Theta, g)
    Theta_g = torch.split(Theta_g, c, dim=1)
    I_Theta_g_1 = torch.hstack(
                    [torch.diagonal(element).reshape(-1, 1) for element in Theta_g]
                  )
    S_I_Theta_g_1 = torch.softmax(I_Theta_g_1, dim=0)

    return S_I_Theta_g_1


def compute_loss(Theta, g, Y, gamma, alpha = 100):
    # Theta's shape is c × k
    c = Theta.size()[0]
    n = int(g.shape[1] / c)

    # Convert to tensors
    g = torch.tensor(g, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)


    S_I_Theta_g_1 = compute_S_I_Theta_g_1(Theta, g)

    # Initialize margin and cross-entropy loss
    M = 0
    C = 0

    for idx in range(n):
        M += torch.matmul(Y[:, idx:idx+1].T, S_I_Theta_g_1[:, idx:idx+1])
        C -= torch.matmul(Y[:, idx:idx+1].T, torch.log(S_I_Theta_g_1[:, idx:idx+1]))

    M -= 1 / alpha * torch.sum(
                     torch.logsumexp(alpha * (S_I_Theta_g_1 - Y * S_I_Theta_g_1),
                                     dim=0),
                     )

    L = (C - gamma * M) / n

    return L


def generate_ring_data(num_samples, inner_radius, outer_radius, noise_std):
    # Generate the first crescent data (upper semicircle)
    theta_first = np.linspace(0, np.pi, num_samples)
    x_first = inner_radius * np.cos(theta_first) + noise_std * np.random.randn(num_samples)
    y_first = inner_radius * np.sin(theta_first) + noise_std * np.random.randn(num_samples)

    # Generate the second crescent data (lower semicircle, offset)
    theta_second = np.linspace(0, np.pi, num_samples)
    x_second = outer_radius * np.cos(theta_second) + inner_radius
    x_second += noise_std * np.random.randn(num_samples)
    y_second = -outer_radius * np.sin(theta_second) + noise_std * np.random.randn(num_samples)

    # Merge the two crescent data
    x = np.concatenate((x_first, x_second))
    y = np.concatenate((y_first, y_second))

    # Create label vector y, first crescent is labeled 1, second crescent is labeled 2
    labels = np.ones(num_samples * 2)
    labels[num_samples:] = 2

    return x, y, labels


def loadmat(path, to_dense = True):
    data = sio.loadmat(path)
    X = data["X"]
    y_true = data["Y"].astype(np.int32).reshape(-1)

    if sparse.isspmatrix(X) and to_dense:
        X = X.toarray()

    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y_true))
    return X, y_true, N, dim, c_true


def compute_classifiers_Y_G_accracies(X_train, y_train, k, deep):
    classifiers, g, accuracies = Train(X_train, y_train, k, deep=deep)
    Y = one_hot(y_train)
    # g = torch.tensor(g, dtype=torch.float)

    return classifiers, Y, g, accuracies


def compute_results(X, labels,
                    y_train,
                    X_test, y_test,
                    classifiers, Theta,
                    g, c):
    g_test = predict_g(classifiers, X_test, c)
    g_test = torch.tensor(g_test, dtype=torch.float)

    y_dataset_test= torch.argmax(compute_S_I_Theta_g_1(Theta, g_test), dim=0).numpy() + 1
    acc_dataset_test = accuracy_score(y_dataset_test, y_test)

    y_dataset_train = torch.argmax(compute_S_I_Theta_g_1(Theta, g), dim=0).numpy() + 1
    acc_dataset_train = accuracy_score(y_dataset_train, y_train)

    g_all = predict_g(classifiers, X, c)
    g_all = torch.tensor(g_all,dtype=torch.float)
    y_dataset_all= torch.argmax(compute_S_I_Theta_g_1(Theta, g_all), dim=0).numpy() + 1
    acc_dataset_all = accuracy_score(y_dataset_all, labels)

    return y_dataset_test, acc_dataset_test, y_dataset_train, acc_dataset_train, y_dataset_all, acc_dataset_all




SEED = 1
# python built-in seed
random.seed(SEED)
# torch seed
torch.manual_seed(1)
# numpy seed
np.random.seed(1)
# sklearn seed
sklearn.random.seed(SEED)

deep = 9
# Get all .mat files
data_path = r'data'
mat_files = ['BASEHOCK.mat', 'breast_uni.mat', 'chess.mat', 'iris.mat', 'jaffe.mat', 'pathbased.mat', 'RELATHE.mat', 'wine.mat']

mat_files = [os.path.join(data_path, f) for f in mat_files]

# Create results DataFrame
columns = ['wrf_train', 'wrf_test', 'wrf_all',
           'rf50_train', 'rf50_test', 'rf50_all',
           'rf100_train', 'rf100_test', 'rf100_all',
           'svc_train', 'svc_test', 'svc_all',
           'xgb_train', 'xgb_test', 'xgb_all',
           'lgbm_train', 'lgbm_test', 'lgbm_all']
results = pd.DataFrame(columns = columns)

# Process each dataset
for mat_file in mat_files:
    dataset_name = os.path.splitext(os.path.basename(mat_file))[0]
    print(f"Processing dataset: {dataset_name}")

    try:
        # Load data
        X, labels, num, dim, c = loadmat(mat_file)
        X = X.astype(np.float32)
        if min(labels) == 0:
            labels = labels + 1
        if min(labels) == -1:
            labels = (labels + 1) / 2 + 1
        c = int(max(labels))
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)


        # Part1 WRF
        k = 10
        gamma = 10
        n = 1000
        loss_array = np.ones(n)

        classifiers, Y, g, accuracies = compute_classifiers_Y_G_accracies(X_train, y_train, k, deep)

        # Calculate Theta
        Theta = init_Theta(accuracies, c)
        Theta = torch.tensor(Theta, requires_grad=True, dtype=torch.float)

        # Define optimizer
        optimizer = optim.SGD([Theta], lr=2)

        # Training process
        for epoch in range(n):
            optimizer.zero_grad()  # Clear gradients
            loss = compute_loss(Theta, g, Y, gamma)  # Calculate loss
            # Extract the scalar value of the single element from the array using the .item() method.
            loss_array[epoch] = loss.detach().numpy().item()
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n}], Loss: {loss.item()}')

        # Get optimized parameter values
        optimal_Theta = Theta.detach().numpy()

        y_wrf_test, acc_wrf_test, y_wrf_train, acc_wrf_train, y_wrf_all, acc_wrf_all = compute_results(X, labels, y_train, X_test, y_test, classifiers, Theta, g, c)


        # Part2 RF50
        k_rf = 50

        classifiers, Y, g, accuracies = compute_classifiers_Y_G_accracies(X_train, y_train, k_rf, deep)

        # Calculate Theta
        Theta = init_Theta(accuracies, c)
        Theta = torch.tensor(Theta, requires_grad=True, dtype=torch.float)

        y_rf50_test, acc_rf50_test, y_rf50_train, acc_rf50_train, y_rf50_all, acc_rf50_all = compute_results(X, labels, y_train, X_test, y_test, classifiers, Theta, g, c)


        # Part3 RF100
        k_rf = 100

        classifiers, Y, g, accuracies = compute_classifiers_Y_G_accracies(X_train, y_train, k_rf, deep)

        # Calculate Theta
        Theta = init_Theta(accuracies, c)
        Theta = torch.tensor(Theta, requires_grad=True, dtype=torch.float)

        y_rf100_test, acc_rf100_test, y_rf100_train, acc_rf100_train, y_rf100_all, acc_rf100_all = compute_results(X, labels, y_train, X_test, y_test, classifiers, Theta, g, c)


        # SVC
        svm_clf = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale', decision_function_shape = 'ovr')

        # Train the model
        svm_clf.fit(X_train, y_train)

        # Predict
        y_svc_test = svm_clf.predict(X_test)
        # Calculate accuracy
        acc_svc_test = accuracy_score(y_test, y_svc_test)
        y_svc_train = svm_clf.predict(X_train)
        acc_svc_train = accuracy_score(y_train, y_svc_train)
        y_svc_all = svm_clf.predict(X)
        acc_svc_all = accuracy_score(labels, y_svc_all)


        # xgboost
        xgb_clf = XGBClassifier(max_depth = deep, learning_rate = 0.2, n_estimators = k, objective = 'multi:softmax',  # 或者 'multi:softprob'
            num_class = c)

        # Train the model
        xgb_clf.fit(X_train, y_train-1)

        # Predict
        y_xgb_test = xgb_clf.predict(X_test)
        # Calculate accuracy
        acc_xgb_test = accuracy_score(y_test, y_xgb_test+1)
        y_xgb_train = xgb_clf.predict(X_train)
        # Calculate accuracy
        acc_xgb_train = accuracy_score(y_train, y_xgb_train+1)
        y_xgb_all = xgb_clf.predict(X)
        # Calculate accuracy
        acc_xgb_all = accuracy_score(labels, y_xgb_all+1)


        # lgbm
        lgbm_clf = LGBMClassifier(max_depth = deep, learning_rate = 0.225, n_estimators = k, objective = 'multiclass', num_class = c)

        # Train the model
        lgbm_clf.fit(X_train, y_train-1)

        # Predict
        y_lgbm_test = lgbm_clf.predict(X_test)
        # Calculate test set accuracy
        acc_lgbm_test = accuracy_score(y_test, y_lgbm_test+1)

        # Predict training set
        y_lgbm_train = lgbm_clf.predict(X_train)
        # Calculate training set accuracy
        acc_lgbm_train = accuracy_score(y_train, y_lgbm_train+1)

        # Predict all data
        y_lgbm_all = lgbm_clf.predict(X)
        # Calculate accuracy for all data
        acc_lgbm_all = accuracy_score(labels, y_lgbm_all+1)


        # Store results
        results.loc[dataset_name] = [
            acc_wrf_train, acc_wrf_test, acc_wrf_all,
            acc_rf50_train, acc_rf50_test, acc_rf50_all,
            acc_rf100_train, acc_rf100_test, acc_rf100_all,
            acc_svc_train, acc_svc_test, acc_svc_all,
            acc_xgb_train, acc_xgb_test, acc_xgb_all,
            acc_lgbm_train, acc_lgbm_test, acc_lgbm_all
        ]
    except Exception as e:
        print(f"######## Error processing dataset {dataset_name}: {str(e)} ########")
        continue

# Save results to CSV
results.to_csv(r"results.csv")
print("Results saved to algorithm_comparison_results.csv")














# Boundary
# from matplotlib.colors import ListedColormap
# def boundary(clf,X,labels,Theta = np.array([[1,2],[3,4]]),c = 2):
#     cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
#     cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
#     h = .1
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     data = np.c_[xx.ravel(), yy.ravel()]
#     if sum(sum(Theta)) =  = 10:
#         Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#         Z = Z + 1  # 调整预测标签范围到1, 2
#         Z = Z.reshape(xx.shape)

#         plt.figure(figsize = (10, 6))
#         plt.contourf(xx, yy, Z, cmap = cmap_light, alpha = 0.8)
#         plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = cmap_bold, edgecolor = 'k', s = 20)
#         plt.xlim(xx.min(), xx.max())
#         plt.ylim(yy.min(), yy.max())
#         plt.title("XGBoost Decision Boundary")
#         plt.xlabel('Feature 1')
#         plt.ylabel('Feature 2')
#         plt.show()
#     else:
#         G = predict_g(clf, data , c)
#         G = torch.tensor(G,dtype=torch.float)
#         Z= torch.argmax(torch.matmul(Theta,G), dim = 0).numpy()+1
#         Z = Z.reshape(xx.shape)
#         plt.figure(figsize = (10, 6))
#         plt.contourf(xx, yy, Z, cmap = cmap_light, alpha = 0.8)
#         plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = cmap_bold, edgecolor = 'k', s = 20)
#         plt.xlim(xx.min(), xx.max())
#         plt.ylim(yy.min(), yy.max())
#         plt.title("XGBoost Decision Boundary")
#         plt.xlabel('Feature 1')
#         plt.ylabel('Feature 2')
#         plt.show()
#     return (xx,yy,Z)

#boundary(classifiers,X,labels,Theta,c = 2)
