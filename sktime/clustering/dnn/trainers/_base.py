# -*- coding: utf-8 -*-
"""
Define abstract class used to train autoencoders
Author:
Baptiste Lafabregue 2019.25.04
"""


import csv
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.cluster import KMeans

import sktime.clustering.dnn._utils as utils


def get_logs(log_dict):
    log = str(log_dict["acc"]) + "," + str(log_dict["max_acc"]) + ","
    log += str(log_dict["nmi"]) + "," + str(log_dict["max_nmi"]) + ","
    log += str(log_dict["ari"]) + "," + str(log_dict["max_ari"])
    return log


class Trainer(object):
    def __init__(
        self,
        dataset_name,
        classifier_name,
        encoder_model,
        batch_size,
        n_clusters,
        optimizer,
    ):

        super(Trainer, self).__init__()

        self.dataset_name = dataset_name
        self.classifier_name = classifier_name
        self.encoder_model = encoder_model
        if encoder_model is not None:
            self.encoder = encoder_model.encoder
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.optimizer = optimizer
        self.pretrain_model = True
        self.allow_pretrain = True
        self.last_logs = {
            "ari": 0.0,
            "nmi": 0.0,
            "acc": 0.0,
            "max_ari": -1.0,
            "max_nmi": -1.0,
            "max_acc": -1.0,
        }
        self.last_logs_test = {
            "ari": 0.0,
            "nmi": 0.0,
            "acc": 0.0,
            "max_ari": -1.0,
            "max_nmi": -1.0,
            "max_acc": -1.0,
        }
        self.last_logs_enc = {
            "ari": 0.0,
            "nmi": 0.0,
            "acc": 0.0,
            "max_ari": -1.0,
            "max_nmi": -1.0,
            "max_acc": -1.0,
        }
        self.last_logs_enc_test = {
            "ari": 0.0,
            "nmi": 0.0,
            "acc": 0.0,
            "max_ari": -1.0,
            "max_nmi": -1.0,
            "max_acc": -1.0,
        }

    def get_trainer_name(self):
        """
        Return the name of the training method used
        :return: method name
        """
        return self.__class__.__name__

    def initialize_model(self, x, y, ae_weights=None):
        """
        Initialize the trainer's model
        :param x: data used
        :param y: data's labels
        :param ae_weights: path of pre-trained weight if exist
        """
        pass

    def load_weights(self, weights_path):
        """
        Load model's weights
        :param weights_path: path to load weights from
        """
        pass

    def save_weights(self, weights_path):
        """
        Save model's weights
        :param weights_path: path to save weights to
        """
        pass

    def extract_features(self, x):
        """
        Extract features from the encoder (before the clustering layer)
        :param x: the data to extract features from
        :return: the encoded features
        """
        return self.encoder.predict(x)

    def reconstruct_features(self, x, already_encoded=False):
        """
        Reconstruct features from the autoencoder (encode and decode)
        :param x: the data to reconstruct features from
        :param encoding: if encoding is False (default value) it is first encoded,
        otherwise it is directly feed to the decoder
        :return: the reconstructed features, None if not supported
        """
        if self.encoder_model.autoencoder is not None:
            if already_encoded:
                return self.encoder_model.autoencoder.decoder_predict(encoding=x)
            return self.encoder_model.autoencoder.decoder_predict(x)
        return None

    def predict_clusters(self, x, seeds=None):
        """
        Predict directly cluster's labels
        :param x: the data to evaluate
        :param seeds: seeds to initialize the K-Means if needed
        :return: the predicted cluster labels, None if not supported by the method
        """
        if seeds is not None:
            seeds_enc = self.extract_features(seeds)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, init=seeds_enc)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)

        x_pred = self.extract_features(x)
        return kmeans.fit_predict(x_pred), kmeans.cluster_centers_

    def pretrain_encoder(
        self,
        x,
        nb_steps,
        log_writer,
        y=None,
        x_test=None,
        y_test=None,
        verbose=True,
        dist_matrix=None,
        noise=None,
    ):
        # varying = bool(np.isnan(np.sum(x)))
        if verbose:
            self.encoder_model.summary()

        self.last_logs_enc = {
            "ari": 0.0,
            "nmi": 0.0,
            "acc": 0.0,
            "max_ari": -np.inf,
            "max_nmi": -np.inf,
            "max_acc": -np.inf,
        }

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        train_loss = tf.keras.metrics.Mean(name="train_loss")

        # define the train function
        @tf.function
        def train_step(x_batch, noisy_batch):
            with tf.GradientTape() as tape:
                loss = self.encoder_model.loss.compute_loss(
                    x_batch, noisy_batch=noisy_batch
                )
            gradients = tape.gradient(
                loss, self.encoder_model.get_trainable_variables()
            )
            self.optimizer.apply_gradients(
                zip(gradients, self.encoder_model.get_trainable_variables())
            )

            train_loss(loss)

        if verbose:
            print("start pre-train")
        # Encoder training
        while i < nb_steps:
            train_loss.reset_states()
            train_ds = tf.data.Dataset.from_tensor_slices(x)
            train_ds = train_ds.shuffle(x.shape[0], reshuffle_each_iteration=True)
            train_ds = train_ds.batch(self.batch_size).as_numpy_iterator()
            for batch in train_ds:
                if noise is not None:
                    noisy_batch = utils.noise(batch, noise)
                else:
                    noisy_batch = batch
                train_step(batch, noisy_batch)

                i += 1
                if i >= nb_steps:
                    break

            if verbose:
                template = "Epoch {}, Loss: {}"
                print(template.format(epochs + 1, train_loss.result()))
            epochs += 1
            y_pred = self.log_stats_encoder(
                x,
                y,
                x_test,
                y_test,
                [train_loss.result(), 0, train_loss.result()],
                epochs,
                log_writer,
                "pretrain",
            )

        if verbose:
            print("end of pre-train")

    def clustering(
        self,
        x,
        y=None,
        nb_steps=50,
        nb_steps_pretrain=None,
        save_dir=None,
        save_suffix="",
        x_test=None,
        y_test=None,
        seeds=None,
        seeds_itr=None,
        verbose=True,
        only_pretrain=False,
        save_pretrain=False,
        stats_dir="./stats/",
        noise=None,
    ):

        ###############################
        # print("dist mat start")
        # log summarized stats
        dist_matrix = None
        # print("dist mat finish")
        ###############################
        if verbose:
            print("Start " + self.get_trainer_name())

        if save_dir is None:
            save_dir = "./results/" + self.get_trainer_name()

        if only_pretrain:
            save_pretrain = True

        # logging file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + "/training_log.csv", "w", newline="")
        log_writer = csv.DictWriter(
            logfile,
            fieldnames=[
                "iter",
                "acc",
                "nmi",
                "ari",
                "L",
                "Lc",
                "Lr",
                "acc_test",
                "nmi_test",
                "ari_test",
                "comment",
            ],
        )
        log_writer.writeheader()
        loss = [0, 0, 0]
        encoder_summary = None

        if self.pretrain_model and self.allow_pretrain:
            if verbose:
                print("Start encoder pre-training")
            if nb_steps_pretrain is None:
                nb_steps_pretrain = nb_steps
            start = datetime.now()
            self.pretrain_encoder(
                x,
                nb_steps_pretrain,
                log_writer,
                y=y,
                dist_matrix=dist_matrix,
                x_test=x_test,
                y_test=y_test,
                verbose=verbose,
                noise=noise,
            )
            end = datetime.now()

        # initialize cluster centers using k-means
        if verbose:
            print("Initializing cluster centers with k-means.")
        if seeds is not None:
            seeds_enc = self.extract_features(seeds)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, init=seeds_enc)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)

        x_pred = self.extract_features(x)
        y_pred = self.predict_clusters(x, seeds)
        y_pred_test, centers = None, None
        if x_test is not None:
            x_pred = self.extract_features(x)
            print("SHAPE" + str(x_pred.shape))
            y_pred_test, centers = kmeans.fit_predict(x_pred), kmeans.cluster_centers_

        # save kmeans clustering result on encoded data
        if save_pretrain:

            utils.create_directory(save_dir + "pre_train_encoder")
            np.save(
                save_dir + "pre_train_encoder/kmeans_affect_" + str(seeds_itr) + ".npy",
                y_pred_test,
            )
            np.save(
                save_dir
                + "pre_train_encoder/kmeans_clusters_"
                + str(seeds_itr)
                + ".npy",
                centers,
            )
            np.save(
                save_dir + "pre_train_encoder/x_encoded_" + str(seeds_itr) + ".npy",
                self.extract_features(x),
            )
            self.encoder_model.save_weights(
                save_dir + "pre_train_encoder/encoder_" + str(seeds_itr) + "_"
            )
            reconstruct = self.reconstruct_features(x)
            if reconstruct is not None:
                np.save(
                    save_dir
                    + "pre_train_encoder/x_reconstructed_"
                    + str(seeds_itr)
                    + ".npy",
                    reconstruct,
                )
            if x_test is not None:
                np.save(
                    save_dir
                    + "pre_train_encoder/x_test_encoded_"
                    + str(seeds_itr)
                    + ".npy",
                    self.extract_features(x_test),
                )
                reconstruct_test = self.reconstruct_features(x_test)
                if reconstruct_test is not None:
                    np.save(
                        save_dir
                        + "pre_train_encoder/x_test_reconstructed_"
                        + str(seeds_itr)
                        + ".npy",
                        reconstruct_test,
                    )

        if only_pretrain:
            ###############################
            # log summarized stats
            f = open(
                stats_dir + self.encoder_model.get_name() + "_" + self.dataset_name, "w"
            )
            f.write(get_logs(self.last_logs_enc) + "\n")
            f.write(get_logs(self.last_logs_enc_test) + "\n")
            f.write(str(start - end))
            f.close()
            ###############################
            return None

        # run the training phase of the framework
        last_epoch = self._run_training(
            x,
            y,
            x_test,
            y_test,
            nb_steps,
            seeds,
            verbose,
            log_writer,
            dist_matrix=dist_matrix,
        )
        end = datetime.now()

        # save idec model
        if verbose:
            print(
                "saving model to:",
                save_dir
                + "/"
                + self.get_trainer_name()
                + "_model_final_"
                + str(seeds_itr)
                + save_suffix,
            )
        self.save_weights(
            save_dir
            + "/"
            + self.get_trainer_name()
            + "_model_final_"
            + str(seeds_itr)
            + save_suffix
        )
        self.log_stats(x, y, x_test, y_test, loss, last_epoch, log_writer, "final")

        logfile.close()

        x_pred = self.extract_features(x)
        x_pred_test = self.extract_features(x_test)
        y_pred, centers = self.predict_clusters(x_test, seeds)

        # save idec result
        utils.create_directory(save_dir + self.get_trainer_name())
        np.save(
            save_dir
            + self.get_trainer_name()
            + "/kmeans_affect_"
            + str(seeds_itr)
            + ".npy",
            y_pred,
        )
        np.save(
            save_dir
            + self.get_trainer_name()
            + "/kmeans_clusters_"
            + str(seeds_itr)
            + ".npy",
            centers,
        )
        np.save(
            save_dir
            + self.get_trainer_name()
            + "/x_encoded_"
            + str(seeds_itr)
            + ".npy",
            x_pred,
        )
        np.save(
            save_dir
            + self.get_trainer_name()
            + "/x_test_encoded_"
            + str(seeds_itr)
            + ".npy",
            x_pred_test,
        )
        reconstruct = self.reconstruct_features(x)
        if reconstruct is not None:
            np.save(
                save_dir
                + self.get_trainer_name()
                + "/x_reconstructed_"
                + str(seeds_itr)
                + ".npy",
                reconstruct,
            )
        reconstruct_test = self.reconstruct_features(x_test)
        if reconstruct_test is not None:
            np.save(
                save_dir
                + self.get_trainer_name()
                + "/x_reconstructed_test_"
                + str(seeds_itr)
                + ".npy",
                reconstruct_test,
            )

        ###############################
        # log summarized stats
        f = open(stats_dir + self.classifier_name + "_" + self.dataset_name, "w")
        f.write(get_logs(self.last_logs) + "\n")
        f.write(get_logs(self.last_logs_test) + "\n")
        if encoder_summary is not None:
            f.write(get_logs(self.last_logs_enc) + "\n")
            f.write(get_logs(self.last_logs_enc_test) + "\n")
            f.write(str(start - end))
        f.close()
        ###############################

        return y_pred
