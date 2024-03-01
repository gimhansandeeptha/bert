import tensorflow as tf
import tensorflow_hub as hub
import os

class Model():
  def __init__(self,bert_model, preprocess_model):
    self.preprocess_model = hub.KerasLayer(preprocess_model)
    # self.bert_model = None
    self.classifier_model = None
    self.checkpoint_dir = None
    self.save_model_path = None
    self.bert_model = hub.KerasLayer(bert_model, trainable=True, name='BERT_encoder')


  def build_classifier_model(self, n_output_classes:int,activation:str, learning_rate:float):
    # Define input layer for text data
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = self.preprocess_model(text_input)

    # Use a pre-trained BERT encoder
    outputs = self.bert_model(encoder_inputs)

    # Extract the pooled output from the BERT model
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)

    # Create the output layer with (/without) softmax activation for classification
    net = tf.keras.layers.Dense(n_output_classes, activation = activation, name='classifier')(net)

    model = tf.keras.Model(text_input, net)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model with an appropriate optimizer and loss function
    model.compile(optimizer=optimizer, loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    self.classifier_model = model

  def train(self, checkpoint_path, x_train,y_train, x_val, y_val, epochs, batch_size):
    checkpoint_path = checkpoint_path
    self.checkpoint_dir = os.path.dirname(checkpoint_path)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= self.checkpoint_dir,  # Save the model to this file
        monitor='val_loss',  # Monitor validation loss
        save_best_only=True,  # Save only the best model
        save_weights_only=True,  # Save the just weights
        mode='min',  # Save the model when the validation loss is minimized (uncomment only if save_best_model is True)
        verbose=1,  # Display messages when saving the model
        # save_freq='epoch'
    )

    # Train the model with the ModelCheckpoint callback
    history = self.classifier_model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint_callback]
    )

  def get_trained_model(self,):
    self.classifier_model.load_weights(self.checkpoint_dir)
    return self.classifier_model

