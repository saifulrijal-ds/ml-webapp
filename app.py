import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")
    
    @st.cache(persist=True)
    def load_data():
        # muat data dari csv ke dataframe pandas
        data = pd.read_csv('mushrooms.csv')
        # pra-pemrosesan data kategori ke dalam bentuk numerik
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split_data(df):
        y = df['class']
        X = df.drop(columns=['class'])
        # membagi data menjadi data train dan test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        return X_train, X_test, y_train, y_train

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()

        if 'Pecision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()
        
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    class_names = ['edible', 'poisonous']

    st.sidebar.subheader("Choose Classifier")
    # membuat dropdown menu untuk memilih classifier
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regulariztion Parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kenrel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

    

    if st.sidebar.checkbox("Display raw data", False):
        st.subheader("Mushroom Dataset (Classification)")
        st.write(df['class'].value_counts()/df['class'].count())
        st.write(df)


if __name__ == '__main__':
    main()