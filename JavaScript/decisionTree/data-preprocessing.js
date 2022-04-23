let tf = require('@tensorflow/tfjs-node');
let dfd = require("danfojs-node");
const { transform } = require('lodash');
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

    let scaler = new dfd.MinMaxScaler();
    //let scaler = new dfd.StandardScaler()

    const labelColumnIndex = data[0].length - 1;
    const featureStatistics = []

    tf.tidy(() => {
        let transposed = tf.transpose(tf.tensor2d(data)).arraySync();
        let featureCount = transposed.length;

        let featureIndex = [...Array(featureCount).keys()];

        let transposedNormalized = [];
        for (let ix in featureIndex) {
            if (ix == labelColumnIndex) {
                transposedNormalized.push(transposed[ix]);
            }
            else {
                // let sf = new dfd.Series(transposed[ix]);
                let sf = new dfd.DataFrame(transposed[ix].map(o => [o]));

                scaler.fit(sf);

                /* Transformacion [z = (x - u) / s ] | x: valor original, u: media, s: desvio estandard */
                // transposedNormalized.push(scaler.transform(sf).tensor.abs().arraySync());
                transposedNormalized.push(_.flatten(scaler.transform(sf).tensor.arraySync()));

                featureStatistics[ix] = {
                    mean: scaler.$min.arraySync(), // scaler.$mean.arraySync(),
                    std: scaler.$max.arraySync() // scaler.$std.arraySync()
                }

                // console.log("transposedNormalized[" + ix + "]", transposedNormalized[ix][0], "|" + !isNaN(transposedNormalized[ix][0]) + "|");
                // console.log("tansposedNormalized[" + ix + "]", tansposedNormalized[ix]);
                // console.log("featureStatistics[ix]", featureStatistics[ix]);
            }
        }

        normalizedData = tf.transpose(tf.tensor2d(transposedNormalized)).arraySync();
    })

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
        /* Transformacion [z = (x - u) / s ] | x: valor original, u: media, s: desvio estandard */
        return (feature - featStats.mean) / featStats.std;
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
        // // console.log("label", label);
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
function fixFFT(dataPoints, settings) {
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
function applyFFT(data, settings) {

    const columnCount = data[0].length;
    const labelColumnIndex = columnCount - 1;
    let result = [];
    tf.tidy(() => {
        let transposed = tf.transpose(tf.tensor2d(data)).arraySync();
        let partialResult = [];
        for (let colIx = 0; colIx < columnCount; colIx++) {
            let featureCollection = transposed[colIx];
            partialResult[colIx] = labelColumnIndex == colIx ? featureCollection : fixFFT(featureCollection, settings);
            // console.log(
            //     "featureCollection", featureCollection.length, 
            //     "partialResult[colIx]", partialResult[colIx].length
            //     );

            // const decimals = 4;
            // let diff = featureCollection.map((v, i) => {
            //     const original = MiscUtils.trunc(v, decimals);
            //     const transColl = MiscUtils.trunc(partialResult[colIx][i], decimals);
            //     const diff = MiscUtils.trunc(v - partialResult[colIx][i], decimals);
            //     const transSingle = MiscUtils.trunc(fixFFT([v], settings)[0], decimals);
            //     const transSingleHalf = MiscUtils.trunc(transSingle / 2, decimals);
            //     const transCollDiff = MiscUtils.trunc(original + diff, decimals);

            //     const min = _.min(featureCollection);
            //     const max = _.max(featureCollection);

            //     const transSingle2 = MiscUtils.trunc(fixFFT(_.orderBy([min, v, max]), settings)[0], decimals);

            //     return [original, transColl, diff, transCollDiff, transSingle];
            // });

            // console.log("[ original, transColl, diff, transCollDiff, transSingle ]");
            // console.log(diff);
            // console.log("[ original, transColl, diff, transCollDiff, transSingle ]");

            // console.log(
            //     "FFT Individual", MiscUtils.trunc(fixFFT([featureCollection[15]], settings)[0], 5), 
            //     "FFT Calc", MiscUtils.trunc(partialResult[colIx][15], 5)
            //     );
        }

        result = tf.transpose(tf.tensor2d(partialResult)).arraySync();
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

        const sampleReformater = (sampleFeatures, label, meanValue, stdDevValue, featureStats) => {
            let result = [];
            for (let i = 0; i < sampleFeatures.length; i++) {
                let currFeature = sampleFeatures[i];
                let currMoments = featureStats[i];
                result.push(currFeature - currMoments.mean);
                result.push(currFeature - currMoments.std);
            }

            result.push(meanValue);
            result.push(stdDevValue);

            return result;
        };

        const { mean, variance } = tf.moments(tf.tensor1d(sample));

        meanValue = mean.arraySync();
        varianceValue = variance.arraySync();
        stdDevValue = tf.sqrt(variance).arraySync();
        resampled = sampleReformater(sample, null, meanValue, stdDevValue, featureStats);

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

function filterDeviation(data) {
    let result = data;
    tf.tidy(() => {
        let transposed = tf.transpose(tf.tensor(data)).arraySync();
        let indexesToRemove = [];
        for (let i = 0; i < transposed.length - 1; i++) {
            const featureArray = transposed[i];
            let oneFeature = tf.tensor1d(featureArray);
            oneFeature = oneFeature.add(1);
            const { mean, variance } = tf.moments(oneFeature);
            const stdDeviation = tf.sqrt(variance);
            const stdDeviationVal = stdDeviation.arraySync();

            featureArray.forEach((o, index) => {
                let devDiff = Math.abs(o - stdDeviationVal);
                let deviations = MiscUtils.trunc(devDiff / stdDeviationVal, 2);
                result.push([deviations, devDiff, index]);
                if (deviations > 2) indexesToRemove.push(index);
            });
        }

        indexesToRemove = [...new Set(indexesToRemove)];

        result = indexesToRemove.map(i => data[i]);
    });

    return result;
}

function remapClass(data) {
    let result = data;
    tf.tidy(() => {
        let transposed = tf.transpose(tf.tensor(data)).arraySync();
        let stdDeviations = transposed.map(featureArray => {
            let oneFeature = tf.tensor1d(featureArray);
            oneFeature = oneFeature.add(1);
            const { mean, variance } = tf.moments(oneFeature);
            const stdDeviation = tf.sqrt(variance);
            const stdDeviationVal = stdDeviation.arraySync();
            return stdDeviationVal;
        });

        result = data.map(sample => {
            const zeroClass = stdDeviations.filter((o, i) => sample[i] < o).length == sample.length - 1;
            if (zeroClass) sample[sample.length - 1] = 0;
            return sample;
        });
    });

    return result;
}

function filterNone(result) {
    return result.filter(o => o[o.length - 1] > 0);
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
    return _.shuffle(samples);
}

function removeLower(samples, featureNames) {
    let result = samples;
    tf.tidy(() => {
        let featureMoments = featureNames.map((f, i) => {
            let values = samples.map(s => s[i]);
            let orderedValues = _.orderBy(values);

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
            return o.slice(0, featureNames.length).filter((f, i) => {
                let moments = featureMoments[i];
                return f > moments.median;
                // return f > (moments.median + moments.std);
                // return f > (moments.median + (moments.std * 2));
            }).length > 0
        });

        // console.log("result.length", result.length);

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
    let localSettings = Object.assign({
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

    // if (localSettings.filter) {
    //     // result = filterDeviation(result);
    //     // result = remapClass(result);
    //     result = filterNone(result);
    // }

    if (localSettings.fourier || localSettings.normalization) {
        const normalizationResult = dataSetNormalization(result, localSettings);
        result = normalizationResult.normalizedData;
        normalizationFeatStats = normalizationResult.featureStatistics;
    }

    if (localSettings.filter) {
        result = removeLower(result, dataFeatureNames);
    }

    if (localSettings.fourier) {
        result = applyFFT(result, localSettings);
    }

    if (localSettings.dataAugmentation) {
        result = dataAugmentation(result, localSettings);
    }

    if (localSettings.deviationMatrix) {
        let { devMatrix, newFeatureNames, gralMean, gralStd, featureStats } = deviationMatrix(result, dataFeatureNames, localSettings);
        result = devMatrix;
        dataFeatureNames = newFeatureNames;
        devMatrixStats = { gralMean, gralStd, featureStats };

        // console.log("grouped", _.groupBy(result, (o) => o[0]+'-'+ o[1]));
        // console.log(result[0]);
    }

    if (localSettings.truncate) {
        result = truncateNumerics(result, localSettings.decimals);
    }

    if (localSettings.shuffle) {
        result = shuffle(result);
    }

    return {
        data: result,
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

function refactorSample(sample, preProcessData) {
    let result = sample;

    const normalizationFeatStats = preProcessData.stats.normalizationFeat;
    const fourierStats = preProcessData.stats.fourier;
    const devMatrixStats = preProcessData.stats.devMatrix;
    const trainingSettings = preProcessData.trainingSettings;

    if (trainingSettings.fourier || trainingSettings.normalization) {
        result = sampleNormalization(result, normalizationFeatStats);
    }

    if (trainingSettings.fourier) {
        /* TODO Como? */
        // result = applyFFT(result, localSettings);
    }

    if (trainingSettings.deviationMatrix) {
        let correlationResult = correlateSample(result, false, devMatrixStats.featureStats);
        result = correlationResult.sample;
    }

    if (trainingSettings.truncate) {
        result = truncateSampleNumerics(result, trainingSettings.truncateDecimals);
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
        const out = tf.math.confusionMatrix(labels, predictions, numClasses);
        const matrix = out.arraySync();
        result = {
            matrix: matrix,
            predictionCount: pedictionLabels.length,
            realLabelCount: realLabels.length,
            precision: _.sum(matrix.map((row, i) => row[i])) / pedictionLabels.length,
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