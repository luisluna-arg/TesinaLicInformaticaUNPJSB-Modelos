const tf = require('@tensorflow/tfjs-node');

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

/**
 * Normalización de un arreglo de datos correspondiente a una feature
 * @param {*} data Datos a normalizar
 * @returns Datos normalizados
 */
function normalization(data, labelColumnIndex) {
    let result = [];

    tf.tidy(() => {
        let transposed = tf.transpose(tf.tensor2d(data)).arraySync();
        let sampleCount = transposed[0].length;
        let featureCount = transposed.length;

        let featureIndex = [...Array(featureCount).keys()];
        let normalizedFeatures = new Array(featureCount).fill(new Array(sampleCount).fill(0.0));

        for (let ix in featureIndex) {
            if (ix == labelColumnIndex) {
                normalizedFeatures[ix] = transposed[ix];
            }
            else {
                let featureArray = transposed[ix];
                let max = Math.max(...featureArray);
                let min = Math.min(...featureArray);
                normalizedFeatures[ix] = featureArray.map(sample => (sample - min) / (max - min));
            }
        }

        result = tf.transpose(normalizedFeatures).arraySync();
    })

    return result;
}

/**
 * Aplica Transformada de Fourier Rápida a un arreglo de datos, aplicando una normalizacion sobre el resultado
 * @param {*} dataPoints Arreglo de datos a modificar
 * @returns Arreglo datos modificados por Transformada de Fourier Rápida
 */
function fixFFT(dataPoints) {
    const real = tf.tensor1d(dataPoints);
    const imag = tf.tensor1d(dataPoints);
    const x = tf.complex(real, imag);
    let arrayFFT = tf.real(x.fft()).arraySync();
    return arrayFFT.map(r => parseFloat((2.0 / arrayFFT.length * Math.abs(r)).toFixed(8)));
}

/**
 * Aplica Transformada de Fourier Rápida sobre matriz de datos
 * @param {*} data Matriz de datos sobre los cuales aplicar Transformada de Fourier Rápida
 * @param {*} labelColumnIndex Indice para columna de clase/label
 * @returns Matriz de datos modificados por Transformada de Fourier Rápida
 */
function applyFFT(data, labelColumnIndex) {
    let columnCount = data[0].length;
    let result = [];
    tf.tidy(() => {
        let transposed = tf.transpose(tf.tensor2d(data)).arraySync();
        let partialResult = [];
        for (let colIx = 0; colIx < columnCount; colIx++) {
            partialResult[colIx] = labelColumnIndex == colIx ? transposed[colIx] : fixFFT(transposed[colIx]);
        }
        result = tf.transpose(tf.tensor2d(partialResult)).arraySync();
    });

    return result;
}

/**
 * Normaliza datos y aplica Transformada de Fourier Rápida
 * @param {*} data Matriz de datos a pre procesar para aplicar a modelos
 * @param {*} labelColumnIndex Indice para columna de clase/label
 * @returns Matriz de datos Pre Procesados
 */
function preProcess(data, labelColumnIndex) {
    return applyFFT(normalization(data, labelColumnIndex), labelColumnIndex);
}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

module.exports = {
    preProcess
};