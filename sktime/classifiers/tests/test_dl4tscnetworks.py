import gc
import sys

import keras

from sktime.datasets import load_italy_power_demand, load_basic_motions


def test_basic_univariate(network):
    '''
    just a super basic test with gunpoint,
        load data,
        construct classifier,
        fit,
        score
    '''

    print("Start test_basic()")

    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    hist = network.fit(X_train[:10], y_train[:10])

    print(network.score(X_test[:10], y_test[:10]))
    print("End test_basic()")


def test_pipeline(network):
    '''
    slightly more generalised test with sktime pipelines
        load data,
        construct pipeline with classifier,
        fit,
        score
    '''

    print("Start test_pipeline()")

    from sktime.pipeline import Pipeline

    # just a simple (useless) pipeline for the purposes of testing
    # that the keras network is compatible with that system
    steps = [
        ('clf', network)
    ]
    clf = Pipeline(steps)

    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    hist = clf.fit(X_train[:10], y_train[:10])

    print(clf.score(X_test[:10], y_test[:10]))
    print("End test_pipeline()")


def test_highLevelsktime(network):
    '''
    truly generalised test with sktime tasks/strategies
        load data, build task
        construct classifier, build strategy
        fit,
        score
    '''

    print("start test_highLevelsktime()")

    from sktime.highlevel import TSCTask
    from sktime.highlevel import TSCStrategy
    from sklearn.metrics import accuracy_score

    train = load_italy_power_demand(split='TRAIN')
    test = load_italy_power_demand(split='TEST')
    task = TSCTask(target='class_val', metadata=train)

    strategy = TSCStrategy(network)
    strategy.fit(task, train.iloc[:10])

    y_pred = strategy.predict(test.iloc[:10])
    y_test = test.iloc[:10][task.target]
    print(accuracy_score(y_test, y_pred))

    print("End test_highLevelsktime()")


def test_basic_multivariate(network):
    '''
    just a super basic test with basicmotions,
        load data,
        construct classifier,
        fit,
        score
    '''
    print("Start test_multivariate()")

    X_train, y_train = load_basic_motions(split='TRAIN', return_X_y=True)
    X_test, y_test = load_basic_motions(split='TRAIN', return_X_y=True)

    hist = network.fit(X_train, y_train)

    print(network.score(X_test, y_test))
    print("End test_multivariate()")


def test_network(network):
    # sklearn compatibility
    # check_estimator(FCN)

    test_basic_univariate(network)
    test_basic_multivariate(network)
    test_pipeline(network)
    test_highLevelsktime(network)


def test_all_networks_all_tests():
    import sktime.contrib.deeplearning_based.dl4tsc.cnn as cnn
    import sktime.contrib.deeplearning_based.dl4tsc.encoder as encoder
    import sktime.contrib.deeplearning_based.dl4tsc.fcn as fcn
    import sktime.contrib.deeplearning_based.dl4tsc.mcdcnn as mcdcnn
    import sktime.contrib.deeplearning_based.dl4tsc.mcnn as mcnn
    import sktime.contrib.deeplearning_based.dl4tsc.mlp as mlp
    import sktime.contrib.deeplearning_based.dl4tsc.resnet as resnet
    import sktime.contrib.deeplearning_based.dl4tsc.tlenet as tlenet
    import sktime.contrib.deeplearning_based.dl4tsc.twiesn as twiesn
    import sktime.contrib.deeplearning_based.tuned_cnn as tuned_cnn

    networks = [
        cnn.CNN(),
        encoder.Encoder(),
        fcn.FCN(),
        mcdcnn.MCDCNN(),
        mcnn.MCNN(),
        mlp.MLP(),
        resnet.ResNet(),
        tlenet.TLENET(),
        twiesn.TWIESN(),
        tuned_cnn.Tuned_CNN(),
    ]

    for network in networks:
        print('\n\t\t' + network.__class__.__name__ + ' testing started')
        test_network(network)
        print('\t\t' + network.__class__.__name__ + ' testing finished')


def comparisonExperiments():
    data_dir = sys.argv[1]
    res_dir = sys.argv[2]

    complete_classifiers = [
        "dl4tsc_cnn",
        "dl4tsc_encoder",
        "dl4tsc_fcn",
        "dl4tsc_mcdcnn",
        "dl4tsc_mcnn",
        "dl4tsc_mlp",
        "dl4tsc_resnet",
        "dl4tsc_tlenet",
        "dl4tsc_twiesn",
    ]

    small_datasets = [
        "Beef",
        "Car",
        "Coffee",
        "CricketX",
        "CricketY",
        "CricketZ",
        "DiatomSizeReduction",
        "Fish",
        "GunPoint",
        "ItalyPowerDemand",
        "MoteStrain",
        "OliveOil",
        "Plane",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "SyntheticControl",
        "Trace",
        "TwoLeadECG",
    ]

    num_folds = 30

    import sktime.contrib.experiments as exp

    for f in range(num_folds):
        for d in small_datasets:
            for c in complete_classifiers:
                print(c, d, f)
                try:
                    exp.run_experiment(data_dir, res_dir, c, d, f)
                    gc.collect()
                    keras.backend.clear_session()
                except:
                    print('\n\n FAILED: ', sys.exc_info()[0], '\n\n')


if __name__ == "__main__":
    test_all_networks_all_tests()
