import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os
import datetime
import re
import string
import argparse

class SentimentAnalyzer:
    def __init__(self, vocab_size=10000, sequence_length=250, embedding_dim=100, batch_size=64):
        """
        Initialize sentiment analysis model with configurable parameters
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        # Create output directory
        self.output_dir = './sentiment_analysis_custom'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load datasets
        self.load_datasets()
        
        # Initialize vectorization layer
        self.vectorize_layer = self.create_vectorization_layer()  # Directly initialize here

         # Prepare datasets
        self.prepare_datasets()
        
        # Prepare models dictionary
        self.models = {}
        
        # Best model tracking
        self.best_model_name = None
        self.best_model_accuracy = 0

        #Loss History
        self.loss_histories = {}
        self.accuracy_histories = {}

    
    def load_datasets(self):
        """
        Load IMDB reviews dataset
        """
        self.train_ds, self.val_ds, self.test_ds = tfds.load(
            'imdb_reviews', 
            split=['train', 'test[:50%]', 'test[50%:]'],
            as_supervised=True
        )
    
    def standardization(self, input_data):
        """
        Standardize input text by lowercasing and removing punctuation
        """
        lowercase = tf.strings.lower(input_data)
        no_tag = tf.strings.regex_replace(lowercase, "<[^>]+>", "")
        no_punctuation = tf.strings.regex_replace(no_tag, r'[^\w\s]', '')
        return no_punctuation
    
    def create_vectorization_layer(self):
        """
        Create and adapt text vectorization layer
        """
        vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=self.standardization,
            max_tokens=self.vocab_size,
            output_mode='int',
            output_sequence_length=self.sequence_length
        )
        
        # Adapt vectorization layer to training data
        if hasattr(self, 'train_ds'): # only adapt if we have a dataset
            training_data = self.train_ds.map(lambda x, y: x)
            vectorize_layer.adapt(training_data)
        return vectorize_layer
    
    def vectorize_data(self, dataset):
        """
        Vectorize dataset
        """
        return dataset.map(lambda x, y: (self.vectorize_layer(x), y))
    
    def prepare_datasets(self):
        """
        Prepare training, validation, and test datasets
        """
        self.train_dataset = self.vectorize_data(self.train_ds).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.val_dataset = self.vectorize_data(self.val_ds).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.test_dataset = self.vectorize_data(self.test_ds).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    
    def create_model_architectures(self):
        """
        Create different model architectures for sentiment analysis
        """
        # SimpleRNN Model
        self.models['simple_rnn'] = self._create_simple_rnn_model()
        
        # LSTM Model
        self.models['lstm'] = self._create_lstm_model()
        
        # GRU Model
        self.models['gru'] = self._create_gru_model()
        
        # Conv1D Model
        self.models['conv1d'] = self._create_conv1d_model()
    
    def _create_simple_rnn_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length,)),
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, 
                                      embeddings_initializer='uniform'),
            tf.keras.layers.SimpleRNN(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_lstm_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length,)),
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, 
                                      embeddings_initializer='uniform'),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_gru_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length,)),
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, 
                                      embeddings_initializer='uniform'),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_conv1d_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length,), dtype="int64"),
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, 
                                      embeddings_initializer='uniform'),
            tf.keras.layers.Conv1D(64, 5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def train_models(self, epochs=5, force_retrain=False):
        """
        Train all defined models
        """
        
        if not force_retrain and self._check_existing_models():
             print("Existing models found, skipping training. Use --train to force retrain.")
             return
        
        self.create_model_architectures()
        
        for model_name, model in self.models.items():
            print(f"Training {model_name} model...")
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            history = model.fit(self.train_dataset, 
                                epochs=epochs, 
                                validation_data=self.val_dataset,
                                verbose = 0
                            )
            self.loss_histories[model_name] = history.history['loss']
            self.accuracy_histories[model_name] = history.history['accuracy']
            
            # Save the model
            model_path = os.path.join(self.output_dir, f'{model_name}_model.h5')
            model.save(model_path)
            print(f"Finished training {model_name}")
            
    def evaluate_models(self):
        """
        Evaluate all trained models and save the best one
        """
        if not self.models:
            print("No models trained, cannot evaluate.")
            return
        
        for model_name, model in self.models.items():
           model_path = os.path.join(self.output_dir, f'{model_name}_model.h5')
           model = tf.keras.models.load_model(model_path)

           loss, accuracy = model.evaluate(self.test_dataset, verbose = 0)
           print(f"{model_name} Model Accuracy: {accuracy:.4f}")
            
           if accuracy > self.best_model_accuracy:
             self.best_model_accuracy = accuracy
             self.best_model_name = model_name
        
        # Save the best model information
        with open(os.path.join(self.output_dir, 'best_model.txt'), 'w') as f:
           f.write(f'Best Model: {self.best_model_name}\n')
           f.write(f'Best Accuracy: {self.best_model_accuracy:.4f}\n')
        print(f'Best Model: {self.best_model_name} with accuracy {self.best_model_accuracy:.4f}')

        self.plot_training_history()
            
    def create_inference_model(self, model_name=None):
      """
      Creates and returns the best model or the model that is passed
      """

      if not model_name:
          if not os.path.exists(os.path.join(self.output_dir, 'best_model.txt')):
              raise ValueError("No best model file found, train the models first.")
          
          with open(os.path.join(self.output_dir, 'best_model.txt'), 'r') as f:
             best_model_line = f.readline()
             if best_model_line:
               self.best_model_name = best_model_line.split(': ')[1].strip()

          model_path = os.path.join(self.output_dir, f'{self.best_model_name}_model.h5')
          if not os.path.exists(model_path):
            raise ValueError(f"Best model {self.best_model_name} not found in the output dir.")
      else:
          model_path = os.path.join(self.output_dir, f'{model_name}_model.h5')
          if not os.path.exists(model_path):
              raise ValueError(f'Model {model_name} not found. Please train the model first.')
          self.best_model_name = model_name

      model = tf.keras.models.load_model(model_path)
      return model
    
    def predict_sentiment(self, model, text):
        """
        Predicts the sentiment of the given text using the provided model.
        """
        vectorized_text = self.vectorize_layer(tf.constant([text]))
        prediction = model.predict(vectorized_text, verbose = 0)[0][0]

        sentiment = "Positive" if prediction > 0.5 else "Negative"
        probability = f"{prediction:.4f}" if prediction > 0.5 else f"{1-prediction:.4f}"
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} with probability: {probability}")
    
    def _check_existing_models(self):
        """
        Checks if models have already been trained and saved.
        """
        if not os.path.exists(self.output_dir):
            return False
        
        for model_name in ['simple_rnn', 'lstm', 'gru', 'conv1d']:
            model_path = os.path.join(self.output_dir, f'{model_name}_model.h5')
            if not os.path.exists(model_path):
                return False
        return True
    
    def plot_training_history(self):
      """
        Plots the loss and accuracy history for each model.
      """
      plt.figure(figsize=(12, 6))
        
      # Plot loss history
      plt.subplot(1, 2, 1)
      for model_name, loss_history in self.loss_histories.items():
          plt.plot(loss_history, label=model_name)
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.title('Training Loss History')
      plt.legend()

      # Plot accuracy history
      plt.subplot(1, 2, 2)
      for model_name, accuracy_history in self.accuracy_histories.items():
        plt.plot(accuracy_history, label=model_name)
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.title('Training Accuracy History')
      plt.legend()

      plt.tight_layout()
      plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
      plt.show()

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Sentiment Analysis CLI')
    parser.add_argument('--train', action='store_true', 
                        help='Force retraining of models')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--model', type=str, default=None, 
                        help='Specify a specific model to use (optional)')
    
    args = parser.parse_args()
    
    # Initialize Sentiment Analyzer
    analyzer = SentimentAnalyzer()
    
    # Train Models if requested or no existing models
    if args.train or not os.path.exists(os.path.join(analyzer.output_dir, 'best_model.txt')):
        print("Training models...")
        analyzer.train_models(epochs=args.epochs, force_retrain=args.train)
        analyzer.evaluate_models()
    
    # Create Inference Model
    try:
        inference_model = analyzer.create_inference_model(args.model)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Interactive Prediction
    while True:
        user_input = input("\nEnter a movie review (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        analyzer.predict_sentiment(inference_model, user_input)

if __name__ == "__main__":
    main()