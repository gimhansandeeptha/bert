import keras
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from official.nlp import optimization
import tensorflow as tf
import tensorflow_hub as hub