let tf = require('@tensorflow/tfjs-node');
let dfd = require("danfojs-node");
const _ = require('lodash');
const SMOTE = require('smote');
const correlation = require('./correlation-matrix');
const MiscUtils = require('./misc-utils');

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */
const DefaultDecimals = 5;

/**
 * 
 * @param {Numeric[]} sample Muestra a truncar
 * @param {Numeric} decimals Cantidad de decimales a redondeo
 * @returns Coleccion de datos redondeados a N decimales
 */
function truncateSampleNumerics(sample, decimals) {
    return sample.map(item => MiscUtils.trunc(item, decimals));
}

/**
 * 
 * @param {Numeric[][]} dataCollection Datos con el formato Coleccion de Coleccion [][]
 * @param {Numeric} decimals Cantidad de decimales a redondeo
 * @returns Coleccion de datos redondeados a N decimales
 */
function truncateNumerics(dataCollection, decimals) {
    return dataCollection.map(sample => truncateSampleNumerics(sample, decimals));
}

/**
 * Normalización de un arreglo de datos correspondiente a una feature
 * @param {*} data DatoSet a normalizar
 * @returns DatoSet normalizado
 */
function dataSetNormalization(data, settings) {
    let normalizedData = [];

    // let scaler = new dfd.StandardScaler()
    let scaler = new dfd.MinMaxScaler()

    const labelColumnIndex = data[0].length - 1;
    const featureStatistics = []

    tf.tidy(() => {
        let transposed = tf.transpose(tf.tensor(data)).arraySync();
        let featureCount = transposed.length;

        let featureIndex = [...Array(featureCount).keys()];

        let transposedNormalized = [];
        for (let ix in featureIndex) {
            if (ix == labelColumnIndex) {
                transposedNormalized.push(transposed[ix]);
            }
            else {
                let sf = new dfd.DataFrame(transposed[ix].map(o => [o]));
                scaler.fit(sf);

                /* Transformacion [z = (x - u) / s ] | x: valor original, u: media, s: desvio estandard */
                transposedNormalized.push(scaler.transform(sf).tensor.abs().arraySync());
                featureStatistics[ix] = {
                    min: scaler.$min.arraySync()[0],
                    max: scaler.$max.arraySync()[0]
                }
            }
        }

        normalizedData = tf.transpose(tf.tensor(transposedNormalized)).arraySync()[0];
    });

    // console.log("Normalizacion general", normalizedData[0]);

    return { normalizedData, featureStatistics };
}

/**
 * Normaliza una muestra, aplicando el mismo proceso aplicado en dataSetNormalization
 * @param {*} sample 
 * @param {*} featureStatistics 
 * @returns Muestra normalizada
 */
function sampleNormalization(sample, featureStatistics) {
    const features = sample.slice(0, sample.length);
    return featureStatistics.map((featStats, index) => {
        const feature = features[index];
        
        // /* Transformacion [z = (x - u) / s ] | x: valor original, u: media, s: desvio estandard */
        // return (feature - featStats.mean) / featStats.std;

        /* 
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min 
        */
       /* Valor normalizado */
        let X_std = (feature - featStats.min) / (featStats.max - featStats.min)
        /* Valor restaurado */
        let X_restores = X_std * (featStats.max - featStats.min) + featStats.min
        result = X_std;
        // console.log("Normalizacion individual");
        // console.log(`x: ${feature}, min: ${featStats.min}, max: ${featStats.max}, X_std: ${X_std}, result: ${result}`);
        // console.log(featureStatistics);
        // console.log(result);
        return result;
    });
}

/**
 * Genera nuevos datos a partir de una colección de datos originales
 * @param data Datos leídos de archivos
 * @param settings Settings generales para carga de archivo
 * @returns Colecciones de datos y etiquetas expandida
 */
function dataAugmentation(samples, settings) {

    // Here we generate 5 synthetic data points to bolster our training data with an balance an imbalanced data set.
    const byLabel = _.groupBy(samples, (item) => item[item.length - 1]);

    for (const label in byLabel) {
        // // // console.log("label", label);
        let labelGroup = byLabel[label].map(sample => sample.slice(0, sample.length - 1));
        const countToGenerate = settings.dataAugmentationTotal - labelGroup.length;

        if (countToGenerate <= 0) return labelGroup;

        // Pass in your real data vectors.
        const smote = new SMOTE(labelGroup);

        let newVectors;

        tf.tidy(() => {
            newVectors = smote.generate(countToGenerate).
                map(vector => {
                    // let features = vector.slice(0, maxIx);
                    /* Tipo movimiento */
                    vector.push(parseInt(label));
                    return vector;
                });
        });

        samples = samples.concat(newVectors);
    }
    return samples;
}

/**
 * Aplica Transformada de Fourier Rápida a un arreglo de datos, aplicando una normalizacion sobre el resultado
 * @param {*} dataPoints Arreglo de datos a modificar
 * @returns Arreglo datos modificados por Transformada de Fourier Rápida
 */
function fixFFT(dataPoints) {
    let real = tf.tensor1d(dataPoints);
    let imag = tf.tensor1d(dataPoints);
    let x = tf.complex(real, imag);
    let arrayFFT = tf.real(x.fft()).arraySync();
    let fftMapped = arrayFFT.map(r => (2.0 / arrayFFT.length * Math.abs(r)));
    let result = fftMapped;
    return result;
}

/**
 * Aplica Transformada de Fourier Rápida sobre matriz de datos
 * @param {*} data Matriz de datos sobre los cuales aplicar Transformada de Fourier Rápida
 * @param {*} labelColumnIndex Indice para columna de clase/label
 * @returns Matriz de datos modificados por Transformada de Fourier Rápida
 */
function applyFFT(data) {
    // console.log("applyFFT.data", data);
    const columnCount = data[0].length;
    const labelColumnIndex = columnCount - 1;
    let result = [];

    tf.tidy(() => {
        // let transposed = tf.transpose(tf.tensor2d(data)).arraySync();
        // let partialResult = [];
        // for (let colIx = 0; colIx < columnCount; colIx++) {
        //     let featureCollection = transposed[colIx];
        //     partialResult[colIx] = labelColumnIndex == colIx ?
        //         featureCollection : fixFFT(featureCollection);
        // }
        // result = tf.transpose(tf.tensor2d(partialResult)).arraySync();

        for (let i = 0; i < data.length; i++) {
            let originalSample = data[i];
            let resampled = [
                ...fixFFT(originalSample.slice(0, originalSample.length - 1)), 
                ...originalSample.slice(originalSample.length - 1)
            ];
            result.push(resampled);
        }
    });

    return result;
}

function correlateSample(currentData, includesClass, featureStats) {
    let resampled = [];
    let meanValue = null, varianceValue = null, stdDevValue = null;
    tf.tidy(() => {
        const featureLength = currentData.length - (includesClass ? 1 : 0);
        const label = currentData.slice(featureLength)[0];
        const sample = currentData.slice(0, featureLength);

        const { mean, variance } = tf.moments(tf.tensor1d(sample));
        meanValue = mean.arraySync();
        varianceValue = variance.arraySync();
        stdDevValue = tf.sqrt(variance).arraySync();

        for (let i = 0; i < sample.length; i++) {
            let currFeature = sample[i];
            let currMoments = featureStats[i];

            /* Respecto de todo el dataset */
            resampled.push(currFeature - currMoments.mean);
            resampled.push(currFeature - currMoments.std);

            // /* Rspecto de la muestra */
            // resampled.push(currFeature - meanValue);
            // resampled.push(currFeature - stdDevValue);
        }

        resampled.push(meanValue);
        resampled.push(stdDevValue);

        if (!MiscUtils.isNullOrUndef(label))
            resampled.push(label);
    });

    return { sample: resampled, meanValue, varianceValue, stdDevValue };
}

function deviationMatrix(data, featureNames, settings) {
    let samples = data.map(o => o.slice(0, featureNames.length));

    /* Media, Varianza y Desvio Estandar de cada colecion completa de features */
    let featureStats = null;
    tf.tidy(() => {
        featureStats = tf.transpose(samples).arraySync().
            map((sample, i) => {
                let moments = tf.moments(tf.tensor1d(sample));
                return {
                    feature: featureNames[i],
                    mean: moments.mean.arraySync(),
                    variance: moments.variance.arraySync(),
                    std: tf.sqrt(moments.variance).arraySync()
                };
            });
    });

    /* Media, Varianza y Desvio Estandar de cada muestra y su remuestreo */
    let sampleStats = [];
    let devMatrix = data.map(row => {
        let correlationResult = correlateSample(row, true, featureStats);
        sampleStats.push({
            mean: correlationResult.meanValue,
            variance: correlationResult.meanValue,
            std: correlationResult.stdDevValue
        });
        return correlationResult.sample;
    })

    let newFeatureNames = [];
    const featureCount = data[0].length - 1;
    for (let i = 0; i < featureCount; i++) {
        const featureName = featureNames[i];
        newFeatureNames.push(`${featureName}_m`);
        newFeatureNames.push(`${featureName}_std`);
    }
    newFeatureNames.push(`mean`);
    newFeatureNames.push(`std`);

    let gralMean = _.sum(sampleStats.map(o => o.meanValue)) / sampleStats.length;
    let gralStd = Math.sqrt(_.sum(sampleStats.map(o => Math.pow(o.stdDevValue, 2))) / sampleStats.length);

    return { devMatrix, newFeatureNames, gralMean, gralStd, featureStats };
}

function filterCorrelationMatrix(data, featureNames) {
    const features = data.map(o => o.slice(0, o.length - 1));
    const correlationResult = correlation.calculateCorrelation(features);
    // console.log("correlationResult", correlationResult);

    let newResult = [];
    let newFeatureNames = [];
    tf.tidy(() => {
        const avgMap = correlationResult.map(item => {
            return (_.sum(item) - 1) / (item.length - 1);
        });

        const { mean } = tf.moments(avgMap);
        let indexToKeep = [];
        avgMap.forEach((currentValue, featureIndex) => {
            if (currentValue >= mean.arraySync()) indexToKeep.push(featureIndex);
        });

        newResult = data.map(sample => {
            let resampled = indexToKeep.map(index => sample[index]);
            resampled.push(sample[sample.length - 1]);
            return resampled;
        });

        newFeatureNames = indexToKeep.map(index => featureNames[index]);
    });

    return { updatedSamples: newResult, updatedFeatureNames: newFeatureNames };
}

function shuffle(samples) {
    let indexedSamples = samples.map((s, i) => { return { index: i, sample: s } });
    const shuffledData = _.shuffle(indexedSamples);
    const result = {
        data: shuffledData.map(s => s.sample),
        indexing: shuffledData.map((s, i) => { return { index: i, original: s.index } })
    }

    return result;
}

function removeLower(samples, featureNames) {
    let result = samples;
    tf.tidy(() => {
        let featureMoments = featureNames.map((f, i) => {
            let values = samples.map(s => s[i]);
            let orderedValues = _.orderBy(values);

            // console.log("values", values);

            const { mean, variance } = tf.moments(tf.tensor(values));
            return {
                mean: mean.arraySync(),
                std: tf.sqrt(variance).arraySync(),
                min: _.min(values),
                max: _.max(values),
                median: orderedValues[Math.floor(orderedValues.length / 2)],
            }
        });


        result = samples.filter(o => {
            return o.slice(0, featureNames.length).filter((f, i) => f > featureMoments[i].median).length > 0
        });

        // // console.log("result.length", result.length);

    });
    return result;
}

/**
 * Normaliza datos y aplica Transformada de Fourier Rápida
 * @param {*} data Matriz de datos a pre procesar para aplicar a modelos
 * @param {*} labelColumnIndex Indice para columna de clase/label
 * @returns Matriz de datos Pre Procesados
 */
function preProcess(data, dataFeatureNames, settings) {
    let localSettings = Object.assign({}, {
        filter: false,
        normalization: true,
        fourier: true,
        deviationMatrix: true,
        truncate: true,
        decimals: DefaultDecimals
    }, settings);

    let result = data;
    let normalizationFeatStats = null;
    let devMatrixStats = null;

    if (localSettings.selectFeatures) {
        let { updatedSamples, updatedFeatureNames } = filterCorrelationMatrix(result, dataFeatureNames);
        result = updatedSamples;
        dataFeatureNames = updatedFeatureNames;
    }

    if (localSettings.fourier || localSettings.normalization) {
        const normalizationResult = dataSetNormalization(result, localSettings);
        result = normalizationResult.normalizedData;
        normalizationFeatStats = normalizationResult.featureStatistics;
        // console.log("normalization");
        console.log(result[0]);
    }

    if (localSettings.filter) {
        result = removeLower(result, dataFeatureNames);
        // console.log("fourier");
        console.log(result[0]);
    }

    if (localSettings.fourier) {
        result = applyFFT(result, localSettings);
        // console.log("result.fourier", result[0]);
    }

    if (localSettings.dataAugmentation) {
        result = dataAugmentation(result, localSettings);
    }

    if (localSettings.deviationMatrix) {
        let { devMatrix, newFeatureNames, gralMean, gralStd, featureStats } = deviationMatrix(result, dataFeatureNames, localSettings);
        result = devMatrix;
        dataFeatureNames = newFeatureNames;
        devMatrixStats = { gralMean, gralStd, featureStats };
        // console.log("result.deviationMatrix", result[0]);
    }

    if (localSettings.truncate) {
        result = truncateNumerics(result, localSettings.decimals);
    }

    let shuffleIndex = null;
    if (localSettings.shuffle) {
        const shuffleResult = shuffle(result);
        result = shuffleResult.data;
        shuffleIndex = shuffleResult.indexing;
    }

    return {
        data: result,
        indexing: shuffleIndex,
        featureNames: dataFeatureNames,
        stats: {
            normalizationFeat: normalizationFeatStats,
            devMatrix: devMatrixStats,
        },
        trainingSettings: {
            selectedFeatures: dataFeatureNames,
            truncateDecimals: localSettings.decimals,
            selectFeatures: localSettings.selectFeatures,
            filter: localSettings.filter,
            fourier: localSettings.fourier,
            normalization: localSettings.normalization,
            dataAugmentation: localSettings.dataAugmentation,
            deviationMatrix: localSettings.deviationMatrix,
            truncate: localSettings.truncate,
        }
    };
}

function refactorSample(sample, preProcessData, dataSet) {
    let result = sample;

    const normalizationFeatStats = preProcessData.stats.normalizationFeat;
    const fourierStats = preProcessData.stats.fourier;
    const devMatrixStats = preProcessData.stats.devMatrix;
    const trainingSettings = preProcessData.trainingSettings;

    if (trainingSettings.selectFeatures) {
        let { updatedSamples, updatedFeatureNames } = filterCorrelationMatrix(result, dataFeatureNames);
        result = updatedSamples;
        dataFeatureNames = updatedFeatureNames;
    }

    if (trainingSettings.normalization) {
        result = sampleNormalization(result, normalizationFeatStats);
        // console.log("normalization");
        // console.log(result[0]);
    }

    if (trainingSettings.fourier) {
        let newDataSet = JSON.parse(JSON.stringify(dataSet));
        result.push(-1); /* Se agrega una clase -1 para respetar el shape */
        newDataSet.push(result);
        newDataSet = applyFFT(newDataSet);
        result = newDataSet[newDataSet.length - 1];
        result = result.slice(0, result.length - 1); /* Se quita la clase -1 agregada */
        // console.log("fourier");
        // console.log(result[0]);

        // // console.log("fourier", result);
        // result.push(-1); /* Se agrega una clase -1 para respetar el shape */
        // result = fixFFT(result, fourierStats);
        // result = result.slice(0, result.length - 1); /* Se quita la clase -1 agregada */
    }

    if (trainingSettings.deviationMatrix) {
        // console.log("deviationMatrix");
        let correlationResult = correlateSample(result, false, devMatrixStats.featureStats);
        result = correlationResult.sample;
        // console.log(result);
    }

    if (trainingSettings.truncate) {
        // console.log("truncate");
        result = truncateSampleNumerics(result, trainingSettings.truncateDecimals);
        // // console.log("Truncate", result);
    }

    return result;
}

function confusionMatrix(pedictionLabels, realLabels) {
    let result = null;
    tf.tidy(() => {
        /* Las label son 0-based */
        const labels = tf.tensor1d(realLabels.map(o => o - 1), 'int32');
        const predictions = tf.tensor1d(pedictionLabels.map(o => o - 1), 'int32');
        const numClasses = 4;
        let out = tf.math.confusionMatrix(labels, predictions, numClasses);
        out = out.arraySync();

        let matrix = [['', '|', ...out[0].map((v, i) => i + 1)]];

        for (let i = 0; i < out.length; i++) {
            matrix.push([i + 1, '|', ...out[i]]);
        }

        result = {
            matrix: matrix,
            predictionCount: pedictionLabels.length,
            realLabelCount: realLabels.length,
            precision: _.sum(out.map((row, i) => row[i])) / pedictionLabels.length,
        }
    });

    return result;
}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

module.exports = {
    preProcess,
    refactorSample,
    confusionMatrix
};