# -*- coding: utf-8 -*-
from sktime.clustering.dnn.encoders._base import AutoEncoder
from sktime.clustering.dnn.encoders._dilated_cnn import DilatedCnn
from sktime.datasets import load_UCR_UEA_dataset


def test_pipeline():
    X_train, y_train = load_UCR_UEA_dataset("CBF", split="train")
    X_test, y_test = load_UCR_UEA_dataset("CBF", split="test")

    dilated_cnn = DilatedCnn(
        nb_filters=40,
        depth=10,
        reduced_size=160,
        latent_dim=320,
        kernel_size=3,
        dropout_rate=0.0,
    )
    autoencoder = AutoEncoder(dilated_cnn, X_train)
    joe = dilated_cnn.create_encoder(X_train)
    jeff = dilated_cnn.create_decoder(X_train)
    l = ""
