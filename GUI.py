import tkinter as tk
from tkinter import *
from tkinter import ttk
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
import threading
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score,get_scorer
import numpy as np

# flobal variabel
stop = False
ratio = .2
selectedmodel = ""
dataloaded = False
# dataset
X_train = None
X_test = None
y_train = None
y_test = None

# load dataset and split
def loadData():
    global dataloaded
    global X_train, X_test, y_train, y_test
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size=0.20)
    dataloaded = True
    event_text.insert(tk.END, "\nDataset has loaded\n")# append message to user screen

# build decision tree classifier
def decisiontree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(max_depth=4,splitter="best",criterion="entropy", random_state=12345)
    clf = clf.fit(X_train, y_train)
    #evaluate classifier
    score = r2_score(y_test, clf.predict(X_test))
    event_text.insert(tk.END, "\nR2 Score: " + str(score))
    event_text.insert(tk.END, "\nMean squared error: " + str(np.mean((clf.predict(X_test) - y_test) ** 2)))

# build linear regression model
def linearregression(X_train, X_test, y_train, y_test):
    clf = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    clf = clf.fit(X_train, y_train)
    #evalaute classifier
    score = r2_score(y_test, clf.predict(X_test))
    event_text.insert(tk.END,'\nCoefficients: '+str(clf.coef_))
    event_text.insert(tk.END, "\nR2 Score: " + str(score))
    event_text.insert(tk.END,"\nMean squared error: "+str(np.mean((clf.predict(X_test) - y_test) ** 2)))

# build ridge classifier
def ridge(X_train, X_test, y_train, y_test):
    clf = Ridge(alpha=0.01)
    clf = clf.fit(X_train, y_train)
    # evaluate classifier
    score = r2_score(y_test, clf.predict(X_test))
    event_text.insert(tk.END, '\nCoefficients: ' + str(clf.coef_))
    event_text.insert(tk.END, "\nR2 Score: " + str(score))
    event_text.insert(tk.END, "\nMean squared error: " + str(np.mean((clf.predict(X_test) - y_test) ** 2)))

# build lasso classifier
def lasso(X_train, X_test, y_train, y_test):
    clf = Lasso(alpha=0.01)
    clf = clf.fit(X_train, y_train)
    # evaluate classifier
    score = r2_score(y_test, clf.predict(X_test))
    event_text.insert(tk.END, '\nCoefficients: ' + str(clf.coef_))
    event_text.insert(tk.END, "\nR2 Score: " + str(score))
    event_text.insert(tk.END, "\nMean squared error: " + str(np.mean((clf.predict(X_test) - y_test) ** 2)))

# build random forest classififer
def randomforest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators = 10,bootstrap=True,max_features='sqrt')
    clf = clf.fit(X_train, y_train)
    # evaluate classifier
    score = r2_score(y_test, clf.predict(X_test))
    event_text.insert(tk.END, "\nR2 Score: " + str(score))
    event_text.insert(tk.END, "\nMean squared error: " + str(np.mean((clf.predict(X_test) - y_test) ** 2)))

# build svm
def supportvectormachine(X_train, X_test, y_train, y_test):
    clf = svm.SVR(kernel='linear', C=1e3)
    clf = clf.fit(X_train, y_train)
    # evaluate classifier
    score = r2_score(y_test, clf.predict(X_test))
    event_text.insert(tk.END, "\nR2 Score: " + str(score))
    event_text.insert(tk.END, "\nMean squared error: " + str(np.mean((clf.predict(X_test) - y_test) ** 2)))

# execute classifier
def run_classifier():
    # global variables
    global dataloaded
    global X_train, X_test, y_train, y_test
    global selectedmodel
    selectedmodel = classifiers.get()# get selected model from user
    if selectedmodel == "":# if user has not selected model
        event_text.insert(tk.END,"\nPlease select Model\n")
        return
    if dataloaded == False:# if user has not loaded dataset
        event_text.insert(tk.END, "\nPlease Load dataset\n")
        return

    # message to user
    event_text.insert(tk.END,"\nSelected Classifier: "+selectedmodel)
    event_text.insert(tk.END, "\n" + "Building Mode, please wait...")

    # if user selected deicison tree
    if selectedmodel == "Decision Tree":
        try:
            thread1 = threading.Thread(target=decisiontree, args=(X_train, X_test, y_train, y_test))# create thread and call classifier
            thread1.start()
        except:
            print("Error: unable to start thread")

    elif selectedmodel == "Linear Regression":
        try:
            thread1 = threading.Thread(target=linearregression, args=(X_train, X_test, y_train, y_test))
            thread1.start()
        except:
            print("Error: unable to start thread")

    if selectedmodel == "Ridge":
        try:
            thread1 = threading.Thread(target=ridge, args=(X_train, X_test, y_train, y_test))
            thread1.start()
        except:
            print("Error: unable to start thread")

    if selectedmodel == "Lasso":
        try:
            thread1 = threading.Thread(target=lasso, args=(X_train, X_test, y_train, y_test))
            thread1.start()
        except:
            print("Error: unable to start thread")

    if selectedmodel == "Random Forest":
        try:
            thread1 = threading.Thread(target=randomforest, args=(X_train, X_test, y_train, y_test))
            thread1.start()
        except:
            print("Error: unable to start thread")

    if selectedmodel == "SVM":
        try:
            thread1 = threading.Thread(target=supportvectormachine, args=(X_train, X_test, y_train, y_test))
            thread1.start()
        except:
            print("Error: unable to start thread")

window = tk.Tk()# create window
window.title("Diabetes Simulator") # set title
# set windows size
window.rowconfigure(1, minsize=400, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

# add text widget to window
ybar= tk.Scrollbar(window)
event_text=tk.Text(window, height=10, width=10)
ybar.config(command=event_text.yview)
event_text.config(yscrollcommand=ybar.set)


fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)# button frame
btn_open = tk.Button(fr_buttons, text="Load",command = loadData)# create load data button

label_ratio = tk.Label(fr_buttons,text="Test Ratio")# test ratio
ratio_text = tk.Text(fr_buttons, height=1, width=4)
ratio_text.insert(tk.END, str(ratio))

btn_run = tk.Button(fr_buttons, text="Run", command=run_classifier)

# Combobox creation
n = tk.StringVar()
classifiers = ttk.Combobox(fr_buttons,width=27, textvariable=n)

# Adding combobox drop down list
classifiers['values'] = ('Decision Tree',
                          'Linear Regression',
                          'Ridge',
                          'Lasso',
                          'Random Forest',
                          'SVM')

btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
label_ratio.grid(row=0,column=3, sticky="ew", padx=5)
ratio_text.grid(row=0, column=4, sticky="ew", padx=5)
classifiers.grid(row=0, column=5, sticky="ew", padx=5)

btn_run.grid(row=0, column=6, sticky="ew", padx=5)

fr_buttons.grid(row=0, column=1, sticky="nsew")
event_text.grid(row=1, column=1, sticky="nsew")
ybar.grid(row=1, column=2, sticky="ns")

window.mainloop()
