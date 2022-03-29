let tf = require('@tensorflow/tfjs-node');
let dfd = require("danfojs-node");

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

let defaultDecimals = 5;

function isNullOrUndef(value) { return typeof value == 'undefined' || value == null; }

function trunc(value, decimals) {
    if (isNullOrUndef(decimals)) {
        decimals = defaultDecimals;
    }

    let fixed = value.toFixed(decimals);
    let fixedFloat = parseFloat(fixed);

    return fixedFloat;
}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

/**
 * 
 * @param {Numeric[][]} dataCollection Datos con el formato Coleccion de Coleccion [][]
 * @param {Numeric} decimals Cantidad de decimales a redondeo
 * @returns Coleccion de datos redondeados a N decimales
 */
function truncateNumerics(dataCollection, decimals) {
    return dataCollection.map(itemCollection => itemCollection.map(item => trunc(item, decimals)));
}


/**
 * Normalización de un arreglo de datos correspondiente a una feature
 * @param {*} data Datos a normalizar
 * @returns Datos normalizados
 */
function normalization(data, labelColumnIndex, settings) {
    let result = [];

    // let scaler = new dfd.MinMaxScaler();
    let scaler = new dfd.StandardScaler()

    tf.tidy(() => {
        let transposed = tf.transpose(tf.tensor2d(data)).arraySync();
        let featureCount = transposed.length;

        let featureIndex = [...Array(featureCount).keys()];

        let tansposedNormalized = [];
        for (let ix in featureIndex) {
            if (ix == labelColumnIndex) {
                tansposedNormalized.push(transposed[ix]);
            }
            else {
                let sf = new dfd.Series(transposed[ix])
                scaler.fit(sf)
                tansposedNormalized.push(scaler.transform(sf).tensor.abs().arraySync());
            }
        }

        result = tf.transpose(tf.tensor2d(tansposedNormalized)).arraySync();
    })

    return result;
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
function applyFFT(data, labelColumnIndex, settings) {
    let columnCount = data[0].length;
    let result = [];
    tf.tidy(() => {
        let transposed = tf.transpose(tf.tensor2d(data)).arraySync();
        let partialResult = [];
        for (let colIx = 0; colIx < columnCount; colIx++) {
            partialResult[colIx] = labelColumnIndex == colIx ? transposed[colIx] : fixFFT(transposed[colIx], settings);
        }

        result = tf.transpose(tf.tensor2d(partialResult)).arraySync();
    });

    return result;
}

function deviationMatrix(data, settings) {
    let devMatrix = [];

    tf.tidy(() => {
        let transposed = tf.transpose(tf.tensor(data)).arraySync();
        let features = tf.tensor(transposed.slice(0, transposed.length - 1));
        let labels = transposed.slice(transposed.length - 1);
        let featuresLength = features.shape[0];
        let trasposedDevMatrix = [];
        let featuresArray = features.arraySync();
        for (let i = 0; i < featuresLength; i++) {
            let currentFeatures = tf.tensor(featuresArray[i]);

            let { mean, variance } = tf.moments(currentFeatures);
            let stdDeviation = tf.sqrt(variance);

            let featuresMeanDiff = currentFeatures.sub(mean).abs().arraySync();
            let featuresStdDevDiff = currentFeatures.sub(stdDeviation).abs().arraySync();

            trasposedDevMatrix.push(featuresMeanDiff);
            trasposedDevMatrix.push(featuresStdDevDiff);
        }
        trasposedDevMatrix.push(labels);
        devMatrix = tf.transpose(tf.tensor(trasposedDevMatrix)).arraySync();
    });

    return devMatrix;
}

/**
 * Normaliza datos y aplica Transformada de Fourier Rápida
 * @param {*} data Matriz de datos a pre procesar para aplicar a modelos
 * @param {*} labelColumnIndex Indice para columna de clase/label
 * @returns Matriz de datos Pre Procesados
 */
function preProcess(data, labelColumnIndex, settings) {

    let localSettings = Object.assign({
        normalization: true,
        fourier: true,
        deviationMatrix: true,
        truncate: true,
        decimals: defaultDecimals
    }, settings);

    let localData = data;
    let result = null;
    if (localSettings.fourier || localSettings.normalization) {
        result = normalization(localData, labelColumnIndex, localSettings);
    }

    if (localSettings.fourier) {
        result = applyFFT(result, labelColumnIndex, localSettings);
    }

    result = truncateNumerics(result, localSettings.decimals)

    if (localSettings.deviationMatrix) {
        result = deviationMatrix(result, localSettings);
    }

    if (localSettings.truncate) {
        result = truncateNumerics(result, localSettings.decimals);
    }

    return result;
}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

module.exports = {
    preProcess
};