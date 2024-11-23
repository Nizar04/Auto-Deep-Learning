import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from keras_tuner import RandomSearch
import json
import os
from PIL import Image
import io

class AutoDL:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        
    def preprocess_data(self, data, target_column, task_type):
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = pd.get_dummies(X[col], prefix=col)
            
        # Scale features
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        # Encode target for classification
        if task_type == 'classification':
            y = self.label_encoder.fit_transform(y)
            
        return X, y
    
    def build_model(self, input_shape, output_shape, task_type):
        def model_builder(hp):
            model = models.Sequential()
            
            # Input layer
            model.add(layers.Dense(
                hp.Int('units_0', min_value=32, max_value=512, step=32),
                activation=hp.Choice('activation_0', ['relu', 'tanh']),
                input_shape=input_shape
            ))
            
            # Hidden layers
            for i in range(hp.Int('num_layers', 1, 5)):
                model.add(layers.Dense(
                    hp.Int(f'units_{i+1}', min_value=32, max_value=512, step=32),
                    activation=hp.Choice(f'activation_{i+1}', ['relu', 'tanh'])
                ))
                model.add(layers.Dropout(hp.Float(f'dropout_{i}', 0, 0.5, step=0.1)))
            
            # Output layer
            if task_type == 'classification':
                model.add(layers.Dense(output_shape, activation='softmax'))
                model.compile(
                    optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            else:
                model.add(layers.Dense(output_shape))
                model.compile(
                    optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                    loss='mse',
                    metrics=['mae']
                )
                
            return model
        
        return model_builder
    
    def train_model(self, X, y, task_type, epochs=50):
        input_shape = X.shape[1:]
        output_shape = len(np.unique(y)) if task_type == 'classification' else 1
        
        tuner = RandomSearch(
            self.build_model(input_shape, output_shape, task_type),
            objective='val_loss',
            max_trials=5,
            directory='model_search',
            project_name='autodl'
        )
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        tuner.search(X_train, y_train,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
        
        self.model = tuner.get_best_models(num_models=1)[0]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
        )
        
        return self.model, self.history

def main():
    st.set_page_config(page_title="AutoDL", layout="wide")
    
    st.title("AutoDL - Automated Deep Learning Platform")
    st.write("Upload your data and let AI handle the rest!")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
                
            st.write("Data Preview:")
            st.dataframe(data.head())
            
            # Configuration
            col1, col2 = st.columns(2)
            
            with col1:
                target_column = st.selectbox(
                    "Select target column",
                    data.columns
                )
                
            with col2:
                task_type = st.selectbox(
                    "Select task type",
                    ['classification', 'regression']
                )
                
            if st.button("Start Training"):
                with st.spinner("Training in progress..."):
                    # Initialize AutoDL
                    autodl = AutoDL()
                    
                    # Preprocess data
                    X, y = autodl.preprocess_data(data, target_column, task_type)
                    
                    # Train model
                    model, history = autodl.train_model(X, y, task_type)
                    
                    # Plot training history
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        name='Training Loss'
                    ))
                    fig.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        name='Validation Loss'
                    ))
                    
                    st.plotly_chart(fig)
                    
                    # Display metrics
                    metrics = model.evaluate(X, y)
                    st.write("Model Performance:")
                    for name, value in zip(model.metrics_names, metrics):
                        st.write(f"{name}: {value:.4f}")
                    
                    # Save model
                    model.save('model.h5')
                    st.download_button(
                        "Download Model",
                        data=open('model.h5', 'rb'),
                        file_name='model.h5'
                    )
                    
                st.success("Training completed!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
