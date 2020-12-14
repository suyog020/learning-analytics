# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:48:59 2020

@author: Suyog
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split 
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Confusion matrix")
        plot_confusion_matrix(model,xtest,ytest)
        st.pyplot()
    if "ROC" in metrics_list:
        st.subheader("ROC")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plot_roc_curve(model,xtest,ytest)
        st.pyplot(width=5)
    if "Precision Recall Curve" in metrics_list:
        st.subheader("Precision Recall Curve")
        plot_precision_recall_curve(model,xtest,ytest)
        st.pyplot()


df = pd.read_csv("dataset/finalll.csv")

class_name = ["final_result"]

st.title('Learning Analytics ')
st.write('\n')  
st.write('\n')

slct = st.sidebar.selectbox(" ",("Home","Show Dataset","Data Visualization","Classification Model"))

if slct == "Home":
    st.header("About the Project")
    st.write("Don't you wish to get better grades in university? Every student has a strategy towards studying for their modules. Some methods include cramming or spacing out the revision of topics covered. This project aims towards studying the patterns of top students in a university and reconfirm existing tips that can help improve learning and score better grades…….")
    
    st.header("What is the OULA dataset about ?")
    st.write("Open University Learning Analytics dataset contains information about 22 courses, 32,593 students, their assessment results, and logs of their interactions with the VLE represented by daily summaries of student clicks (10,655,280 entries). The dataset is divided into 7 csv files. ")
        
    st.header("Why do we choose this dataset ?")
    st.write("We always thought about how we could improve the way we study and get better grades in university. There are many books and online courses out there that share different methods of studying and how to better retain information. Techniques such as spaced repetition and active recall leads to better long-term learning while cramming, although highly effective for tests/exams, results in faster forgetting. ")
    st.write("Our goal is to find out whether consistency in work will result in better grades. What are some of the habits that successful students have that allow them to get the grades they want? Are they just plain smarter? What strategy do they use when it comes to studying for an important exam like finals? Do they do anything different from average students? ")
    st.write("The Open University currently collects similar data on an on-going basis as input to algorithms they developed to identify students at risk for failing a course. Identification of at-risk students then triggers automated intervention measures to encourage behavior that would create success. For example, the algorithm might identify a student with low grades on intermediate assessments (quizzes). That student may be sent an automated email reminder about available tutoring options. The goal of the data collection effort is to maximize student success, which has numerous benefits for the University. ")
    st.write("This subset of anonymized data was made available to the public for educational purposes on Machine Learning approaches. ")
    
    st.header("Purpose of Our Project")
    st.write("For the purpose of this project, the data will be used to determine if socio-economic and/or behavior-based data can be used to predict a student's performance in a course. Performance is determined by the final result of the student’s effort and is characterized by completing the course with a passing score, either with or without Distinction. ")
    
    st.header("Specific Questions of Interest")
    st.write("● Can we predict a student's final status in a course based on socio-economic factors and/or patterns of interaction with the VLE?")
    st.write("● Desired Targets:\n 1. Prediction of student pass/ no pass the course after course completion (goal: 90% accuracy)\n 2. Prediction of student pass/ no pass the course after 30 days of commencement (goal: 75% accuracy) ")

    

if slct == "Show Dataset":
    st.header("Student Data")
    st.write(df)
    
    st.write('\n\n')
    
if slct == "Classification Model":

    clfr = st.sidebar.selectbox("Classifier",("Logistic Regression","Decision Tree","Random Forest","KNN"))

    if clfr == "Logistic Regression":
        
        x =df.drop(columns=['final_result'])
        y=df['final_result']
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
        #parameters
        st.sidebar.subheader("Parameters: ")
        iterations = st.sidebar.slider("Iterations", min_value=10, max_value=1000, value=100, step=5, format=None, key='iterations')
        C = st.sidebar.number_input("Regularization Factor",min_value=0.01,max_value=1.0,step=0.01,key='C')
        solver = st.sidebar.radio("Solver",("newton-cg", "lbfgs", "liblinear", "sag", "saga"),key='solver')
        #metrics
        metrics = st.sidebar.multiselect("Select metrics",("Confusion Matrix","ROC","Precision Recall Curve"))
        #classify
        if st.sidebar.button("Classify"):
            model = LogisticRegression(max_iter=iterations,solver=solver,C=C)
            model.fit(xtrain,ytrain)
            ypred = model.predict(xtest)
            st.write("Accuracy: ",model.score(xtest,ytest))
            st.write("")
            accuracy = cross_val_score(model,x, y, scoring='accuracy', cv = 10)
            st.write("Cross validated Accuracy : " , accuracy.mean())
            st.write("")
            st.write("Model Precision: ", precision_score(ytest,ypred,labels=class_name))
            plot_metrics(metrics)
    
    if clfr == "Decision Tree":
        
        x =df.drop(columns=['final_result'])
        y=df['final_result']
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
        #parameters
        st.sidebar.subheader("Parameters: ")
        max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes",50,200,step=1,key='max_leaf_nodes')
        criterion = st.sidebar.radio("Criterion",("gini", "entropy"),key='criterion')
        max_features = st.sidebar.radio("Features",("auto", "sqrt", "log2"),key='max_features')
        #metrics
        metrics = st.sidebar.multiselect("Select metrics",("Confusion Matrix","ROC","Precision Recall Curve"))
        #classify
        if st.sidebar.button("Classify"):
            model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,criterion=criterion,max_features=max_features)
            model.fit(xtrain,ytrain)
            ypred = model.predict(xtest)
            st.write("Accuracy: ",model.score(xtest,ytest))
            st.write("")
            accuracy = cross_val_score(model,x, y, scoring='accuracy', cv = 10)
            st.write("Cross validated Accuracy : " , accuracy.mean())
            st.write("")
            st.write("Model Precision: ", precision_score(ytest,ypred,labels=class_name))
            plot_metrics(metrics)
            
    if clfr == "Random Forest":
        
        x =df.drop(columns=['final_result'])
        y=df['final_result']
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
        #parameters
        st.sidebar.subheader("Parameters: ")
        n_estimators = st.sidebar.slider("No.of trees in the forest", min_value=10, max_value=1000, value=100, step=10, format=None, key='n_estimators')
        criterion = st.sidebar.radio("Criterion",("gini", "entropy"),key='criterion')
        max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes",50,200,step=1,key='max_leaf_nodes')
        random_state= st.sidebar.slider("Random State", min_value=0, max_value=42, value=0, step=1, format=None, key='random_state')
        max_features = st.sidebar.radio("Features",("auto", "sqrt", "log2"),key='max_features')
        #metrics
        metrics = st.sidebar.multiselect("Select metrics",("Confusion Matrix","ROC","Precision Recall Curve"))
        #classify
        if st.sidebar.button("Classify"):
            model = DecisionTreeClassifier()
            model = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,criterion=criterion,max_features=max_features,random_state=random_state)
            model.fit(xtrain,ytrain)
            ypred = model.predict(xtest)
            st.write("Accuracy: ",model.score(xtest,ytest))
            st.write("")
            accuracy = cross_val_score(model,x, y, scoring='accuracy', cv = 10)
            st.write("Cross validated Accuracy : " , accuracy.mean())
            st.write("")
            st.write("Model Precision: ", precision_score(ytest,ypred,labels=class_name))
            plot_metrics(metrics)

    if clfr == "KNN":
        
        x =df.drop(columns=['final_result'])
        y=df['final_result']
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
        #parameters
        st.sidebar.subheader("Parameters: ")
        n_neighbors = st.sidebar.slider("No. of neighbors", min_value=3, max_value=20, value=5, step=1, format=None, key='n_neighbors')
        algorithm = st.sidebar.radio("Algorithm",("auto", "ball_tree", "kd_tree", "brute"),key='algorithm')
        leaf_size = st.sidebar.slider("Leaf size", min_value=10, max_value=50, value=30, step=1, format=None, key='leaf_size')
        p = st.sidebar.radio("Power parameter",(1,2),key='p')
        #metrics
        metrics = st.sidebar.multiselect("Select metrics",("Confusion Matrix","ROC","Precision Recall Curve"))
        #classify
        if st.sidebar.button("Classify"):
            model = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=algorithm,leaf_size=leaf_size,p=p)
            model.fit(xtrain,ytrain)
            ypred = model.predict(xtest)
            st.write("Accuracy: ",model.score(xtest,ytest))
            st.write("")
            accuracy = cross_val_score(model,x, y, scoring='accuracy', cv = 10)
            st.write("Cross validated Accuracy : " , accuracy.mean())
            st.write("")
            st.write("Model Precision: ", precision_score(ytest,ypred,labels=class_name))
            plot_metrics(metrics)
