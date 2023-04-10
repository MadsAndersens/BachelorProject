import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd, norm
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
import ast
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets, svm, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
#Import naive bayes
from sklearn.naive_bayes import GaussianNB
#import Knn
from sklearn.neighbors import KNeighborsClassifier
#Import confusion matrixdisplay
from sklearn.metrics import ConfusionMatrixDisplay
#For truncated svd, we need to import the base estimator and classifier mixin
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the data
train_data = pd.read_csv('/BachelorProject/Data/VitusData/Train.csv')
train_data = train_data.drop(['Unnamed: 0'], axis=1)
test_data = pd.read_csv('/BachelorProject/Data/VitusData/Val.csv')
test_data = test_data.drop(['Unnamed: 0'], axis=1)
base_dir = '/BachelorProject/Data/'
# define class labels
labels = {
    0: 'Negative',
    1: 'Positive',
}
#Inverse labels
inv_labels = {v: k for k, v in labels.items()}

# Training data size
M = 60 # Number of images

def load_image(path):
    img = Image.open(f'{base_dir}{path}')
    #Resize the image to 320x320
    img = img.resize((320, 320))
    img = np.array(img)
    img = np.array(img)
    img = img.flatten()
    #img = img.reshape((-1,1))
    return img

def get_images(data, M=102,train=True):
    # Load the data such that the collumns of X is the image as a vector
    # Initialize np array of size (320*320,len(data))
    X = [] #np.zeros(((320 * 320),M*2))
    y = []#[None] * (M*2)
    neg_count = 0
    pos_count = 0
    for (i, row) in tqdm(data.iterrows()):
        label = ast.literal_eval(row['Label'])

        if label[0] == 'Negative' and neg_count < M:
            im = load_image(row['ImageDir'])
            #X[:,i] = im
            X.append(im)
            #y[i] = -1
            y.append(-1)
            if train:
                neg_count += 1
        elif label[0] != 'Negative' and pos_count < M:
            im = load_image(row['ImageDir'])
            #X[:,i] = im
            X.append(im)
            #y[i] = 1
            y.append(1)
            if train:
                pos_count += 1
        elif neg_count >= M and pos_count >= M:
            print("What")
            break
        else:
            continue
    X = np.array(X)
    X = X.T#We want the collumns to be the images
    #X = X[:, ~np.all(X == 0, axis=0)]
    #y = [x for x in y if x is not None]
    return X, np.array(y)

#Create the X and y matrices
X_train,y_train = get_images(train_data,train=True)
X_test,y_test = get_images(test_data,train=False)

#Scale the data
scaler = StandardScaler()

#Make pipeline
hgb_pipe = make_pipeline(StandardScaler(), HistGradientBoostingClassifier())
parameters = {
 'histgradientboostingclassifier__max_iter': [1000,1200,1500],
 'histgradientboostingclassifier__learning_rate': [0.1],
 'histgradientboostingclassifier__max_depth' : [25, 50, 75],
 'histgradientboostingclassifier__l2_regularization': [1.5],
 'histgradientboostingclassifier__scoring': ['f1_micro'],
 'histgradientboostingclassifier__random_state' : [42],
 }

hgb_grid = GridSearchCV(hgb_pipe, parameters, n_jobs=5, cv=5, scoring='f1_micro', verbose=2, refit=True)

hgb_grid.fit(X_train.T, y_train)

#Make classification report
y_pred = hgb_grid.predict(X_test.T)

#Print classification report
print(classification_report(y_test, y_pred))

#Print confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels.values())
disp.plot()
plt.show()










X_train = scaler.fit_transform(X_train.T).T
X_test = scaler.transform(X_test.T).T

#Write class for truncated SVD such that it acts like an sklearn classifier
class TruncatedSVDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,inverter='pinv'):
        self.inverter = inverter

    def fit(self, X, y):
        # Make the y vector be -1 or 1 instead of 0 or 1
        y_train = np.where(y == 0, -1, y)
        X_pinv = np.linalg.pinv(X_train)
        A = np.matmul(y_train, X_pinv)
        self.A = A
        return self

    def predict(self, X):
        y_pred = np.matmul(self.A, X)
        return np.sign(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y)/len(y)

#Create the truncated SVD classifier
clf = TruncatedSVDClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

#Confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, display_labels=['Negative', 'Positive'], xticks_rotation="vertical",colorbar=False
    )
disp.plot()
plt.show()


# Make list of classifiers from the sklearn library
classifiers = [
    svm.SVC(kernel='linear', C=0.01),
    svm.SVC(kernel='rbf', gamma=0.7, C=1),
    svm.SVC(kernel='poly', degree=3, C=1),
    RandomForestClassifier(n_estimators=100, random_state=0),
    AdaBoostClassifier(n_estimators=100, random_state=0),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=5),
    KNeighborsClassifier(n_neighbors=7)
]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
# Loop over the classifiers
for clf in tqdm(classifiers):
    clf.fit(X_train.T, y_train)
    y_pred = clf.predict(X_test.T)
    print(clf)
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    disp = ConfusionMatrixDisplay.from_estimator(
        clf, X_test.T, y_test, display_labels=['Negative', 'Positive'], xticks_rotation="vertical",colorbar=False)
    disp.plot(ax=axes.flat[classifiers.index(clf)])
    #Set title if svm add kernel
    if clf.__class__.__name__ == 'SVC':
        axes.flat[classifiers.index(clf)].set_title(clf.__class__.__name__ + ' ' + clf.kernel)
    else:
        axes.flat[classifiers.index(clf)].set_title(clf.__class__.__name__)


#plt.tight_layout()
##Set fig title
#fig.suptitle('Confusion matrix for different non DL classifiers', fontsize=16)
#fig.show()

#Remove the colorbars
#for ax in fig.axes:
#    ax.collections[0].colorbar.remove()



#Try with stacking classifiers
estimators = [(clf.__class__.__name__ + ' ' + clf.kernel if clf.__class__.__name__ == 'SVC' else clf.__class__.__name__, clf) for clf in classifiers[:-2]]
clf =StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
clf.fit(X_train.T, y_train)
y_pred = clf.predict(X_test.T)
print(clf)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test.T, y_test, display_labels=['Negative', 'Positive'],colorbar=False
)

disp.plot()
fig.show()













##
##Find the model parameters by taking the pseudo inverse of X
#X_pinv = np.linalg.pinv(X_train)
#A = np.matmul(y_train,X_pinv)
#
##Predict the test data
#y_pred = np.matmul(A,X_test)
#y_pred = np.sign(y_pred)
#
##Find the accuracy
#acc = np.sum(y_pred == y_test)/len(y_test)
#
##Compute the confusion matrix
#confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
#print(confusion_matrix)
#
##Compute the classification report
#print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))









##Plot the confusion matrix
#plt.figure()
#disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
#                                display_labels=labels.values())
#disp.plot()
#plt.show()
#
## Train model
#clf_4 = RandomForestClassifier(random_state=123)
#clf_4.fit(X_train.T, y_train)
#
## Predict on test set
#y_pred = clf_4.predict(X_test)
#
## Print results
#print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
#ConfusionMatrixDisplay.from_estimator(
#    clf_4, X_test, y_test, display_labels=['Negative', 'Positive'], xticks_rotation="vertical"
#)
#plt.tight_layout()
#plt.show()



##Save the X and y matrices to csv files to save time
#np.savetxt("/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/NoneDeepLearning/X_train.csv", X_train, delimiter=",")
#np.savetxt("/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/NoneDeepLearning/y_train.csv", y_train, delimiter=",")
#np.savetxt("/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/NoneDeepLearning/X_test.csv", X_test, delimiter=",")
#np.savetxt("/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/NoneDeepLearning/y_test.csv", y_test, delimiter=",")
#
##Load the X and y matrices
#X_train = np.loadtxt("/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/NoneDeepLearning/X_train.csv", delimiter=",")
#y_train = np.loadtxt("/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/NoneDeepLearning/y_train.csv", delimiter=",")
#X_test = np.loadtxt("/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/NoneDeepLearning/X_test.csv", delimiter=",")
#y_test = np.loadtxt("/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/NoneDeepLearning/y_test.csv", delimiter=",")

#Create the classifier
#clf = svm.SVC(gamma=0.001)
#clf.fit(X_train, y_train)





####################### Baseline 1- SVD############################
# Currently not working since it runs out of memory, but the idea is to use SVD to find the best fit for each class
# Setup the alpha matrices
#alpha_matrices= {"A0":X_train[:,y_train == 0][:,0:1000],
                 #"A1":X_train[:,y_train == 1]}

#svd_neg = TruncatedSVD(n_components=10)
#svd_neg.fit(alpha_matrices["A0"])
#svd_pos = TruncatedSVD(n_components=10)
#svd_pos.fit(alpha_matrices["A1"])

#svd = [svd_neg,svd_pos]

#Find the best fit for each class for each image
#for i in range(2):



#Predict the class of the test data

#del(X_train,y_train) #Delete the X_train and y_train matrices to save memory
#Print shapes
#print(f"Shape of A0: {alpha_matrices['A0'].shape}")
#print(f"Shape of A1: {alpha_matrices['A1'].shape}")






#del(alpha_matrices)
#def get_USV(alpha_matrices):
#    # Get the U,S,V matrices
#    left_singular = {}
#    singular_matix = {}
#    right_singular = {}
#    for key in alpha_matrices:
#        U, S, V = svd(alpha_matrices[key], full_matrices=False)
#        del(S,V)
#        left_singular[key] = U
#        #singular_matix[key] = S
#        #right_singular[key] = V
#    return left_singular,singular_matix,right_singular

#Get the U,S,V matrices
#left_singular,singular_matix,right_singular = get_USV(alpha_matrices)
#del(alpha_matrices)

#Predict the class of the test data
#def predict(x_test,y_test,left_singular):
#    I = np.eye(x_test.shape[0])
#    kappas = np.arange(5, 21)
#    len_test = x_test.shape[1]
#    predictions = np.empty((len(y_test), 0), dtype=int)
#    for t in list(kappas):
#        prediction = []
#        for i in range(len_test):
#            residuals = []
#            for j in range(2):
#                u = left_singular["A" + str(j)][:, 0:t]
#                res = norm(np.dot(I - np.dot(u, u.T), x_test[i]))
#                residuals.append(res)
#            index_min = np.argmin(residuals)
#            prediction.append(index_min)
#
#        prediction = np.array(prediction)
#        predictions = np.hstack((predictions, prediction.reshape(-1, 1)))
#    scores = []
#    for i in range(len(kappas)):
#        score = accuracy_score(y_test.loc[0, :], predictions[:, i])
#        scores.append(score)
#    data = {"Number of basis vectors": list(kappas), "accuracy_score": scores}
#    df = pd.DataFrame(data).set_index("Number of basis vectors")
#    return df,predictions





#Predict the class of the test data
#df,predictions = predict(X_test,y_test,left_singular)

#plot the accuracy score
#df.plot()
#plt.show()

