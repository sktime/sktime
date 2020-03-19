
# Time Series Transformers (trafo) - simple/1st degree

## Atoms

### Transforming a Single TS to a Primitive Data Frame Row (1-ts-to-df)

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
| Tabularise (UK)  | transformers.compose.Tabulariser | @mloning  |  |
| Tabularize (US)  | transformers.compose.Tabularizer | @mloning  |  |
| Auto-correlation function  | transformers.spectral_based.AutoCorrelationFourierTransformer | @jsellier  |  |
| Bin Interval segmenter  (fixed) | transformers.compose.IntervalSegmenter | @mloning  |  |
| Bin Interval segmenter (random)  | transformers.compose.RandomIntervalSegmenter | @mloning  |  |
| Cosine Transform  | transformers.spectral_based.CosineTransformer | @jsellier  |  |
| Discrete Fourier Transform  | transformers.spectral_based.DiscreteFourierTransformer | @jsellier  |  |
| Power Spectrum | transformers.spectral_based.PowerSpectrumTransformer | @jsellier  |  |
| Feature Extractor | transformers.summarise.\_tsfresh.TSFreshFeatureExtractor | @mloning @Ayushmaanseth |  |
| Relevant Feature Extractor | transformers.summarise.\_tsfresh.TSFreshRelevantFeatureExtractor | @mloning @Ayushmaanseth |  |

### Transforming a Single TS to a Single TS (1-ts-to-1-ts)

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
| Derivative Series | transformers.summarise.DeriativeSlopeTransformer | @mloning |  |
| Plateau Finder | transformers.summarise.PlateauFinder | @mloning |  |
| Random Intervals | transformers.summarise.RandomIntervalFeatureExtractor | @mloning |  |



### Transforming a Batch of TS to a Primitive Data Frame (n-ts-to-df)

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
| Tabularise (UK)  | transformers.compose.Tabulariser | @mloning  |  |
| Tabularize (US)  | transformers.compose.Tabularizer | @mloning  |  |
| Bin Interval segmenter  (fixed) | transformers.compose.IntervalSegmenter | @mloning  |  |
| Bin Interval segmenter (random)  | transformers.compose.RandomIntervalSegmenter | @mloning  |  |
| Matrix profile | transformers.matrix_profile | Claudia Rincon Sanchez | (custom implementation) |
| Principal component scores after tabularization | transformers.PCATransformer | @prockenschaub | [ Hotelling (1933) - Analysis of a complex of statistical variables into principal components](https://psycnet.apa.org/record/1934-00645-001) |
| Shapelet transform | transformers.ShapeletTransform | @jasonlines| [ Hills et al (2014) - Classification of time series by shapelet transformation](https://link.springer.com/article/10.1007/s10618-013-0322-1) |
| Shapelet transform (contracted) | transformers.ContractedShapeletTransform | @jasonlines| [ Hills et al (2014) - Classification of time series by shapelet transformation](https://link.springer.com/article/10.1007/s10618-013-0322-1) |
| Shapelet transform (random sampled) | transformers.RandomEnumerationShapeletTransform | @jasonlines| [ Hills et al (2014) - Classification of time series by shapelet transformation](https://link.springer.com/article/10.1007/s10618-013-0322-1) |

### Transforming a Batch of Multivariate TS to a Batch of Univariate TS (n-mts-to-n-uts)

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
| concatenate dimensions  | transformers.compose.ColumnConcatenator | @mloning  |  |

## Higher-order building blocks

### Composites

| components | name | sktime class | maintainer | literature
| ------| ------ | ------- | ------ | ------- |
| n-ts-to-X | Concatenate column-wise | transformers.compose.ColumnTransformer | @mloning |  |
| n-ts-to-X | Feature union | pipeline.FeatureUnion | @mloning |  |

### Reduction

| from/output | to/input | name | sktime class | maintainer | literature
| ------ | ------ | ------ | ------- | ------ | ------- |
| n-ts-to-df | 1-ts-to-df | Apply row-wise | transformers.compose.RowwiseTransformer | @mloning |  |

# Time Series Transformers - paired/2nd degree

(note: interface for 2nd degree transformers is currently under re-factoring, currently not consistent or homogenous)

## Atoms

### Distances

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
| BOSS Distance | classifiers.dictionary_based.boss.boss_distance | @MatthewMiddlehurst | [Schäfer (2014) - The BOSS is concerned with time series classification in the presence of noise](https://link.springer.com/article/10.1007/s10618-014-0377-7) |

### Kernels

(todo - this is in goastler ork)

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
|  |  |  |  |

# Time Series Classification (TSC)

## Atoms

### Univariate Time Series Classifiers

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
| BOSS Ensemble | classifiers.dictionary_based.boss.Boss_Ensemble | @MatthewMiddlehurst | [Schäfer (2014) - The BOSS is concerned with time series classification in the presence of noise](https://link.springer.com/article/10.1007/s10618-014-0377-7) |
| BOSS Atom | classifiers.dictionary_based.boss.BossIndividual | @MatthewMiddlehurst | [Schäfer (2014) - The BOSS is concerned with time series classification in the presence of noise](https://link.springer.com/article/10.1007/s10618-014-0377-7) |
| Elastic Ensemble | classifiers.distance_based.elastic_ensemble.ElasticEnsemble | @jasonlines | [Lines, Bagnall (2015) - Time Series Classification with Ensembles of Elastic Distance Measures](https://link.springer.com/article/10.1007/s10618-014-0361-2) |
| Proximity Forest | classifiers.distance_based.boss.ProximityForest | @goastler | [Lucas et al (2019) - Proximity Forest: an effective and scalable distance-based classifier for time series](https://link.springer.com/article/10.1007/s10618-019-00617-3) |
| Proximity Stump | classifiers.distance_based.boss.ProximityStump | @goastler | [Lucas et al (2019) - Proximity Forest: an effective and scalable distance-based classifier for time series](https://link.springer.com/article/10.1007/s10618-019-00617-3) |
| Random Interval Spectral Forest (RISE) | classifiers.frequency_based.rise.RandomIntervalSpectralForest | @TonyBagnall | [Lines et al (2018) - Time Series Classification with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based Ensembles](https://ieeexplore.ieee.org/document/7837946) |
| Shapelet Transform Classifier | classifiers.shapelet_based.stc.ShapeletTransformClassifier | @TonyBagnall | [Hills et al (2014) - Classification of time series by shapelet transformation](https://ieeexplore.ieee.org/document/7837946) |
| Time Series Forest | classifiers.interval_based.tsf.TimeSeriesForestClassifier | @TonyBagnall | [Deng et al (2013) - A Time Series Forest for Classification and Feature Extraction](https://www.sciencedirect.com/science/article/pii/S0020025513001473) |
| Time Series k-NN | classifiers.distance_based.time_series_neighbors.KNeighborsTimeSeriesClassifier | @jasonlines |  |
| ROCKET | transformers.rocket.Rocket | @angus924 | [Dempser et al (2019) ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels](https://arxiv.org/abs/1910.13051) |
| Mr-SEQL | classifiers.shapelet_based.MrSEQLClassifier | @lnthach | [Interpretable Time Series Classification Using Linear Models and Multi-resolution Multi-domain Symbolic Representations](https://link.springer.com/article/10.1007/s10618-019-00633-3) |

### Multivariate Time Series Classifiers

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
|  |  |  |  |

## Higher-order building blocks

### Ensembling (abstract/1st order)

(only abstract ensembles in this list - hard-coded ensembles go in one of the lists for atoms)

| of | name | sktime class | maintainer | literature
| ------ | ------ | ------ | ------- | ------ |
| univariate TSC | boosting TSC  | classifiers.compose.ensemble.TimeSeriesForestClassifier | @mloning  |  |

### Pipelines

| components | name | sktime class | maintainer | literature
| ------ | ------ | ------- | ------ | ------- |
| trafos, TSC | pipeline | sktime.pipeline.Pipeline |  |  |

### Reduction

| from/output | to/input | name | sktime class | maintainer | literature
| ------ | ------ | ------ | ------- | ------ | ------- |
| multivariate TSC| univariate TSC | column ensembler  | classifiers.compose.column_ensembler.ColumnEnsembleClassifier | @abostrom  |  |

# Time Series Regression (TSR)

## Atoms

### Univariate Time Series Regressors

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
|  |  |  |  |

### Multivariate Time Series Regressors

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
|  |  |  |  |

# Forecasting

## Atoms

### Univariate Time Series Forecasting

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
|  |  |  |  |

### Multivariate Time Series Forecasting

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
|  |  |  |  |

### Time Series Forecasting with Exogeneity

| name | sktime class | maintainer | literature
| ------ | ------- | ------ | ------- |
|  |  |  |  |

## Higher-order building blocks

### Reduction
