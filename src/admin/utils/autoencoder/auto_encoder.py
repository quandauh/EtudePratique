#!/usr/bin/python3

import tensorflow as tf 
import numpy as np
from app.models import AppSettings

class Autoencoder:
    """
    Constructeur de l'auto encoder
    :param input_count (integer) : taille du vecteur d'entrée
    :param hidden_count (integer) : taille du vecteur encodé
    :param loss_function (function) : Fonction d'erreur de l'auto encoder
    """
    def __init__(self, input_count, hidden_count, loss_function='binary_crossentropy'):
        configT = tf.compat.v1.ConfigProto()
        configT.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=configT)

        self._setup(input_count, hidden_count, loss_function)
        self.input_count = input_count

    """
    Initialisation du réseau de neurones de l'auto encoder
    :param input_count (integer) : taille du vecteur d'entrée
    :param hidden_count (integer) : taille du vecteur encodé
    :param loss_function (function) : Fonction d'erreur de l'auto encoder
    """
    def _setup(self, input_count, hidden_count, loss_function):

        # Input layer that will receive vectors of data
        self.inputs = tf.keras.layers.Input(input_count, name="personnage")
        # Mask layer to enhance loss function
        self.mask = tf.keras.layers.Input(input_count, name="mask")

        # Encoder using ReLu activation
        self.encoding_layout = tf.keras.layers.Dense(hidden_count, activation='relu', name="encoded")(self.inputs)
        # Decoder use sigmoid (decoded datas are binary)
        self.decoding_layout = tf.keras.layers.Dense(input_count, activation='sigmoid', name="decoded")(self.encoding_layout)
        
        # Trick loss function
        self.tricked = tf.keras.layers.Multiply(name="masked_output")([self.decoding_layout, self.mask])

        # Auto encoder model
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.decoding_layout)

        # Encoder only part of auto encoder
        self.encoding_layout = tf.keras.Model(inputs=self.inputs, outputs=self.encoding_layout)

        # Auto encoder that tricks loss function
        self.tricked = tf.keras.Model(inputs=[self.inputs, self.mask], outputs=self.tricked)

        # Compile model
        self.tricked.compile(
            loss=loss_function,#tf.keras.losses.BinaryCrossentropy(), 
            optimizer=tf.keras.optimizers.experimental.SGD(learning_rate=20),
            metrics=[tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()],
        )
        
        self.tricked.summary()

    """
    Sauvegarde des pondérations des connexions du réseau dans un fichier
    :param file_name (string) : nom du fichier de sauvegarde
    """
    def save_weights(self, file_name):
        print("Saving weights to ", file_name, "...")
        self.tricked.save_weights(file_name)

    """
    Chargement des pondérations des connexions du réseau depuis un fichier
    :param file_name (string) : nom du fichier de sauvegarde
    """
    def load_weights(self, file_name):
        print('Loading weights from ', file_name, "...")
        self.tricked.load_weights(file_name)

    """
    Entrainement du réseau de neuronne
    :param X (array) : vecteurs d'entrée (personnages)
    :param Y (array) : vecteurs de sortie (personnages)
    :param batch (int) : nombre d'échantillons par mise à jour de gradient
    :param nb_epochs (integer) : nombre d'itérations pour l'entrainement

    Note : pour un auto encoder standard, X = Y, le paramètre Y est laissé en option
    pour pouvoir créer un système de suggestion inspiré des autoencodeurs débruiteurs
    """
    def fit(self, X, Y, batch=16, nb_epochs=10, callbacks=None, **fit_kwargs):
        return self.tricked.fit(
            X,
            Y,
            epochs=nb_epochs,
            batch_size=batch,
            shuffle=fit_kwargs.pop("shuffle", True),
            callbacks=callbacks,
            **fit_kwargs,
        )

    """
    Encodage d'un vecteur avec la partie encodeur
    :param X (array) : vecteur d'entrée (personnage)
    :return array : vecteur de sortie (personnage) compressé 
    """
    def transform(self, X):
        representation = self.encoding_layout.predict(X)
        return np.asarray(representation)

    """
    Passage d'un vecteur dans l'auto encodeur complet
    :param X (array) : vecteur d'entrée (personnage)
    :return array : vecteur de sortie (personnage) encodé puis décodé
    """
    def predict(self, X):
        reconstruct = self.model.predict(X)
        return np.asarray(reconstruct)  # .flatten()
    
    def export(self, filename):
        self.tricked.save(f"tricked_{filename}")
        self.model.save(filename)

    def save_compressed_matrix_db(self):
        None
