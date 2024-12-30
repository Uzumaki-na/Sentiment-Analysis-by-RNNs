import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Embedding

class GeneralSentimentAnalyzer:
    def __init__(self, vocab_size=10000, sequence_length=250, embedding_dim=100, batch_size=64, use_pretrained_embedding=True):
        """
        Initializes the general sentiment analysis model.
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.use_pretrained_embedding = use_pretrained_embedding
        
        self.output_dir = './general_sentiment_analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        self.models = {}
        self.best_model_name = None
        self.best_model_accuracy = 0
        self.loss_histories = {}
        self.accuracy_histories = {}
        
        # Initialize vectorization layer only once
        if 'vectorize_layer' not in st.session_state:
           self._load_data()

    def _load_data(self):
      """
      Loads the dataset, initializes the vectorization layer, and prepares datasets.
      This function will be called only once per session
      """
      self.load_datasets()
      self.vectorize_layer = self.create_vectorization_layer()
      self.prepare_datasets()
      st.session_state['vectorize_layer'] = self.vectorize_layer

    def load_datasets(self):
        """Loads the 'ag_news_subset' dataset."""
        self.train_ds, self.val_ds = tfds.load(
            'ag_news_subset',
            split=['train[:10%]', 'train[10%:15%]'],  # 10% training, 5% validation
            as_supervised=True
        )

    def standardization(self, input_data):
        """Standardizes input text (lowercase, remove tags/punctuation)."""
        lowercase = tf.strings.lower(input_data)
        no_tag = tf.strings.regex_replace(lowercase, "<[^>]+>", "")
        no_punctuation = tf.strings.regex_replace(no_tag, r'[^\w\s]', '')
        return no_punctuation
    
    def create_vectorization_layer(self):
        """Creates and adapts the text vectorization layer."""
        vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=self.standardization,
            max_tokens=self.vocab_size,
            output_mode='int',
            output_sequence_length=self.sequence_length
        )
        if hasattr(self, 'train_ds'):
            training_data = self.train_ds.map(lambda x, y: x)
            vectorize_layer.adapt(training_data)
        return vectorize_layer
    
    def vectorize_data(self, dataset):
        """Vectorizes the dataset."""
        return dataset.map(lambda x, y: (self.vectorize_layer(x), tf.cast(y, tf.float32)/3.0))

    def prepare_datasets(self):
        """Prepares training and validation datasets."""
        self.train_dataset = self.vectorize_data(self.train_ds).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.val_dataset = self.vectorize_data(self.val_ds).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def create_model_architectures(self):
        """Creates different model architectures."""
        self.models['simple_rnn'] = self._create_simple_rnn_model()
        self.models['lstm'] = self._create_lstm_model()
        self.models['gru'] = self._create_gru_model()
        self.models['conv1d'] = self._create_conv1d_model()
    
    def _create_simple_rnn_model(self):
        """Creates a simple RNN model."""
        embedding_layer = self._get_embedding_layer()
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length,)),
            embedding_layer,
            tf.keras.layers.SimpleRNN(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_lstm_model(self):
        """Creates an LSTM model."""
        embedding_layer = self._get_embedding_layer()
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length,)),
           embedding_layer,
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_gru_model(self):
        """Creates a GRU model."""
        embedding_layer = self._get_embedding_layer()
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length,)),
            embedding_layer,
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_conv1d_model(self):
        """Creates a Conv1D model."""
        embedding_layer = self._get_embedding_layer()
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length,), dtype="int64"),
           embedding_layer,
            tf.keras.layers.Conv1D(64, 5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def _get_embedding_layer(self):
        """Creates and returns an embedding layer (pretrained or random)."""
        if self.use_pretrained_embedding:
            return self._create_pretrained_embedding()
        else:
            return Embedding(self.vocab_size, self.embedding_dim, embeddings_initializer='uniform')

    def _create_pretrained_embedding(self):
        """Creates and returns a pre-trained embedding layer."""
        embedding_matrix = self._load_glove_embeddings()
        return tf.keras.layers.Embedding(
            input_dim = self.vocab_size,
            output_dim = self.embedding_dim,
            embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
            trainable = False
        )

    def _load_glove_embeddings(self):
        """Loads pre-trained GloVe embeddings."""
        if 'embeddings_index' not in st.session_state:
            embeddings_index = {}
            glove_path = './glove.6B.100d.txt'
            if not os.path.exists(glove_path):
                print("Downloading pre-trained embeddings...")
                glove_path = tf.keras.utils.get_file(
                     "glove.6B.100d.txt",
                    "http://nlp.stanford.edu/data/glove.6B.zip",
                    extract=True,
                    cache_subdir = '.',
                 )
                glove_path = os.path.join(os.path.dirname(glove_path), 'glove.6B.100d.txt')

            with open(glove_path, encoding="latin-1") as f:
                for line in f:
                   try:
                       values = line.split()
                       if len(values) > 1:
                           word = values[0]
                           coefs = np.asarray(values[1:], dtype="float32")
                           embeddings_index[word] = coefs
                   except (ValueError, IndexError):
                       continue
            embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
            if hasattr(self, 'vectorize_layer'):
                for word, i in enumerate(self.vectorize_layer.get_vocabulary(include_special_tokens=False)[:self.vocab_size]):
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
            st.session_state['embeddings_index'] = embeddings_index
            st.session_state['embedding_matrix'] = embedding_matrix
        return st.session_state['embedding_matrix']
    
    def train_models(self, epochs=5, force_retrain=False):
        """Trains all defined models."""
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
            
            model_path = os.path.join("sentiment_analysis_custom", f'{model_name}_model.h5')
            os.makedirs(os.path.dirname(model_path), exist_ok = True) # make the directory if it doesn't exists
            model.save(model_path)
            print(f"Finished training {model_name}")
            
    def evaluate_models(self):
        """Evaluates all trained models and saves the best one."""
        if not self.models:
            print("No models trained, cannot evaluate.")
            return
        
        for model_name, model in self.models.items():
           model_path = os.path.join("sentiment_analysis_custom", f'{model_name}_model.h5')
           model = tf.keras.models.load_model(model_path, compile=False)
           loss, accuracy = model.evaluate(self.val_dataset, verbose = 0)
           print(f"{model_name} Model Accuracy: {accuracy:.4f}")
            
           if accuracy > self.best_model_accuracy:
             self.best_model_accuracy = accuracy
             self.best_model_name = model_name
        
        with open(os.path.join(self.output_dir, 'best_model.txt'), 'w') as f:
           f.write(f'Best Model: {self.best_model_name}\n')
           f.write(f'Best Accuracy: {self.best_model_accuracy:.4f}\n')
        print(f'Best Model: {self.best_model_name} with accuracy {self.best_model_accuracy:.4f}')

    def create_inference_model(self, model_name=None):
      """Creates and returns the inference model."""
      if not model_name:
          if not os.path.exists(os.path.join(self.output_dir, 'best_model.txt')):
              raise ValueError("No best model file found, train the models first.")
          
          with open(os.path.join(self.output_dir, 'best_model.txt'), 'r') as f:
             best_model_line = f.readline()
             if best_model_line:
               self.best_model_name = best_model_line.split(': ')[1].strip()
          
          model_path = os.path.join("sentiment_analysis_custom", f'{self.best_model_name}_model.h5')
          if not os.path.exists(model_path):
            raise ValueError(f"Best model {self.best_model_name} not found.")
      else:
          model_path = os.path.join("sentiment_analysis_custom", f'{model_name}_model.h5')
          if not os.path.exists(model_path):
              raise ValueError(f'Model {model_name} not found. Please train the model first.')
          self.best_model_name = model_name

      model = tf.keras.models.load_model(model_path, compile=False)
      return model
    
    def predict_sentiment(self, model, text):
      """Predicts the sentiment of the given text using the provided model."""
      if 'vectorize_layer' in st.session_state:
          vectorized_text = st.session_state['vectorize_layer'](tf.constant([text]))
          prediction = model.predict(vectorized_text, verbose = 0)[0][0]
          sentiment = "Positive" if prediction > 0.5 else "Negative"
          probability = f"{prediction:.4f}" if prediction > 0.5 else f"{1-prediction:.4f}"
          return sentiment, probability
      else:
         raise ValueError("Vectorize layer is not initialized, please train the model")
    
    def _check_existing_models(self):
        """Checks if models have already been trained and saved."""
        if not os.path.exists(self.output_dir):
            return False
        
        for model_name in ['simple_rnn', 'lstm', 'gru', 'conv1d']:
            model_path = os.path.join("sentiment_analysis_custom", f'{model_name}_model.h5')
            if not os.path.exists(model_path):
                return False
        return True
    
    def plot_training_history(self, model_name):
      """Plots the loss and accuracy history for a specific model."""
      plt.figure(figsize=(12, 6))
      
      plt.subplot(1, 2, 1)
      loss_history = self.loss_histories[model_name]
      plt.plot(loss_history, label=model_name)
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.title('Training Loss History')
      plt.legend()

      plt.subplot(1, 2, 2)
      accuracy_history = self.accuracy_histories[model_name]
      plt.plot(accuracy_history, label=model_name)
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.title('Training Accuracy History')
      plt.legend()
      plt.tight_layout()

      st.pyplot(plt)
      plt.close()

def main():
    st.title("General Sentiment Analyzer")
    
    if 'analyzer' not in st.session_state:
         st.session_state.analyzer = GeneralSentimentAnalyzer(use_pretrained_embedding = True)
    analyzer = st.session_state.analyzer
    
    user_input = st.text_area("Enter your text here:")

    model_options = ["simple_rnn", "lstm", "gru", "conv1d"]
    selected_model = st.selectbox("Select a Model", model_options)
    
    if st.button("Train Models"):
      with st.spinner('Training models...'):
         analyzer = GeneralSentimentAnalyzer(use_pretrained_embedding = True)
         analyzer.train_models(epochs=5, force_retrain=True)
         analyzer.evaluate_models()
         st.session_state.analyzer = analyzer

      st.success('Models trained!')
      st.header("Training Metrics:")
      for model_name in model_options:
          st.subheader(f"Model: {model_name}")
          analyzer.plot_training_history(model_name)
    
    try:
      inference_model = analyzer.create_inference_model(selected_model)
    except ValueError as e:
      st.error(f"Error: {e}")
      return

    if st.button("Analyze Sentiment"):
      if user_input:
        try:
           sentiment, probability = analyzer.predict_sentiment(inference_model, user_input)
           st.write(f"**Sentiment:** {sentiment}")
           st.write(f"**Probability:** {probability}")
        except ValueError as e:
             st.error(f"Error: {e}")
      else:
        st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()