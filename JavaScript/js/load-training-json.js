const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const _ = require('lodash');
const SMOTE = require('smote');
const { preProcess } = require('./data-preprocessing');

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

const defaultDecimals = 5;

function getArrayShape(array) {
  return [array.length, array.length > 0 ? Array.isArray(array[0]) ? array[0].length : 1 : 1];
}

function isNullOrUndef(value) {
  return typeof value == 'undefined' || value == null;
}

function featureNamesForDevMatrix(featureNames) {
  let newNames = [];
  for (let a = 0; a < featureNames.length; a++) {
    const featureName = featureNames[a];
    newNames.push(featureName + '_m');
    newNames.push(featureName + '_std');
  }
  return newNames;
}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

const MOVE_TYPE = {
  NONE: 0,
  DOWN: 1,
  UP: 2,
  LEFT: 3,
  RIGHT: 4
};

class ReadData {

  constructor(samples, maxes, labels = null) {
    this.samples = samples;
    this.labels = labels;
    this.dataMax = maxes;

    const _FEATURE_NAMES = [
      "delta",
      "theta",
      "lowAlpha",
      "highAlpha",
      "lowBeta",
      "highBeta",
      "lowGamma",
      "highGamma"
    ];

    this.FEATURE_NAMES = _FEATURE_NAMES;
  }

  concat(data2) {
    if (isNullOrUndef(data2)) return;

    let data1 = this;
    data1.samples = data1.samples.concat(data2.samples);

    if (isNullOrUndef(data1.labels) != isNullOrUndef(data2.labels)) {
      throw 'No se puede concatenar, uno de los dataset esta dividido en muestras y etiquetas'
    }

    if (!isNullOrUndef(data1.labels) && !isNullOrUndef(data2.labels)) {
      data1.labels = data1.labels.concat(data2.labels);
    }

    if (isNullOrUndef(data1.dataMax) || isNullOrUndef(data2.dataMax)) {
      this.calculateDataMaxes();
    }
    else {
      for (let i = 0; i < data1.dataMax.length; i++) {
        if (data1.dataMax[i] < data2.dataMax[i]) {
          data1.dataMax[i] = data2.dataMax[i];
        }
      }
    }
  }

  /**
   * Calcula los valores maximos para la coleccion de muestras de la instancia, por columnas
   */
  calculateDataMaxes(labelColumnCount = null) {

    let localLabelColCount = labelColumnCount;

    if (localLabelColCount == null && Array.isArray(this.labels) && this.labels.length > 0) {
      let labelExtraction = this.labels[0];
      localLabelColCount = Array.isArray(labelExtraction) ? labelExtraction.length : 1;
    }
    localLabelColCount = (localLabelColCount != null ? localLabelColCount : 0);

    let featureColumnCount = this.samples[0].length - localLabelColCount;
    let maxes = new Array(featureColumnCount).fill(0);

    for (let i = 0; i < this.samples.length; i++) {
      const currentSample = this.samples[i];
      for (let j = 0; j < featureColumnCount; j++) {
        if (maxes[j] < currentSample[j]) maxes[j] = currentSample[j];
      }
    }

    this.dataMax = maxes;
  }

  summary() {
    console.log("DATA SUMMARY");
    console.log("============");

    const samplesShape = getArrayShape(this.samples);
    let maxes = {};
    for (let i = 0; i < this.FEATURE_NAMES.length; i++) {
      maxes[this.FEATURE_NAMES[i]] = this.dataMax[i];
    }

    if (this.labels != null && Array.isArray(this.labels)) {
      const labelsShape = getArrayShape(this.labels);
      console.log("Shapes Samples|Labels:", samplesShape, labelsShape);

      console.log("Feature maxes:", maxes);
    }
    else {
      console.log("Shapes Samples:", samplesShape);
      console.log("Feature maxes:", maxes);
    }
  }

}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */



/**
 * Lee archivos pasados por parametro para luego convertirlos en objectos manejables y etiquetados
 * @param fileData Datos leídos de archivos
 * @param settings Settings generales para carga de archivo
 * @returns Datos leídos. Separados en features y labels según settings.
 */
function loadJSON(fileData, settings) {

  const localSettings = Object.assign({
    shuffle: false,
    minTolerance: 0,
    split: false,
    dataAugmentation: false,
    dataAugmentationTotal: 1000,
    normalization: true,
    fourier: true,
    deviationMatrix: false,
    decimals: defaultDecimals
  }, settings);

  /* Maximos: Para validar la tolerancia de las lecturas */
  const dataMax = new Array(8);
  dataMax.fill(0);

  let samples = [];
  _.each(fileData, (fileSettings) => {
    /* Read lecture as JSON */
    let data = JSON.parse(
      fs.readFileSync(fileSettings.file, { encoding: 'utf-8' })
    ).filter(o => o.poorSignalLevel === 0);

    let newData = data.
      filter(o => o.poorSignalLevel == 0).
      map((record) => {
        /* Crea el array de la muestra */
        let dataItem = [
          record.eegPower.delta, /* delta */
          record.eegPower.theta, /* theta */
          record.eegPower.lowAlpha, /* lowAlpha */
          record.eegPower.highAlpha, /* highAlpha */
          record.eegPower.lowBeta, /* lowBeta */
          record.eegPower.highBeta, /* highBeta */
          record.eegPower.lowGamma, /* lowGamma */
          record.eegPower.highGamma, /* highGamma */
          /* Tipo movimiento */
          fileSettings.moveType, /* moveType */
        ];

        /* Actualiza maximo encontrado */
        for (let i = 0; i < dataItem.length - 2; i++) {
          if (dataMax[i] < dataItem[i]) dataMax[i] = dataItem[i];
        }

        return dataItem;
      });

    samples = _.concat(samples, newData);
  });

  let readData = new ReadData(samples, dataMax);

  const columnCount = 1;

  if (localSettings.fourier || localSettings.normalization) {
    labelColumn = 8;
    readData.samples = preProcess(readData.samples, labelColumn, localSettings);
    readData.calculateDataMaxes(columnCount);
  }

  if (localSettings.minTolerance > 0) {
    readData = filterData(readData, localSettings);
  }

  if (localSettings.dataAugmentation) {
    readData = dataAugmentation(readData, localSettings);
    readData.calculateDataMaxes(columnCount);
  }

  if (localSettings.deviationMatrix) {
    readData.FEATURE_NAMES = featureNamesForDevMatrix(readData.FEATURE_NAMES);
    readData.calculateDataMaxes(columnCount);
  }

  if (localSettings.split) {
    return splitData(readData, localSettings);
  }

  return readData;
}

function filterData(data, settings) {
  const localSettings = Object.assign({
    minTolerance: 0,
  }, settings);

  let finalSamples = []
  const columnCount = 1;

  /* Se normaliza cada variable de medicion en un rango de 0 a 1, 
    *  usando su porcentaje respecto del maximo valor observado en toda la muestra
    *  Si el valor observado supero el % de tolerancia, se lo considera valor aceptable y
    *  se usan sus labels.
    *  Si no se supera ese % de tolerancia minimo, se desactivan las labels (array de 0)
    *  para descartar la muestra
    */
  for (let i = 0; i < data.samples.length; i++) {
    const sample = data.samples[i];
    const features = sample.slice(0, sample.length - columnCount); /* Sample array */

    for (let featureIndex = 0; featureIndex < features.length; featureIndex++) {
      let featureMax = data.dataMax[featureIndex];
      let featureValue = features[featureIndex];
      if (featureMax > 0 && featureValue / featureMax > localSettings.minTolerance) {
        finalSamples.push(sample);
        break;
      }
    }
  }

  data.samples = finalSamples;

  return data;

}

/**
 * Aleatoriza los datos y los separa en arreglos de muestras y etiquetas
 * Tambien ajusta las etiquetas de acuerdo al nivel de tolerancia fijado en options
 * @param data Datos leídos de archivos
 * @param settings Settings generales para carga de archivo
 * @returns Data separada en muestras y etiquetas
 */
function splitData(data, settings) {
  const localSettings = Object.assign({
    shuffle: false,
    minTolerance: 0
  }, settings);

  // Step 1. Shuffle the data
  let dataSamples = data.samples;
  if (localSettings.shuffle) { dataSamples = _.shuffle(dataSamples); }

  // Step 2. Split into samples and labels
  let finalSamples = [];
  let finalLabels = [];
  const labelColumnCount = 1;

  _.each(dataSamples, (dataArray) => {
    const featureCount = dataArray.length - labelColumnCount;
    finalSamples.push(dataArray.slice(0, featureCount));
    finalLabels.push(dataArray.slice(featureCount));
  });

  data.labels = finalLabels;
  data.samples = finalSamples;

  return data;
}

/**
 * Genera nuevos datos a partir de una colección de datos originales
 * @param data Datos leídos de archivos
 * @param settings Settings generales para carga de archivo
 * @returns Colecciones de datos y etiquetas expandida
 */
function dataAugmentation(readData, settings) {

  // Here we generate 5 synthetic data points to bolster our training data with an balance an imbalanced data set.
  const countToGenerate = settings.dataAugmentationTotal - readData.samples.length;

  if (countToGenerate <= 0) return readData;

  // Pass in your real data vectors.
  const smote = new SMOTE(readData.samples);

  let newVectors;
  const labelColumnCount = 1;

  tf.tidy(() => {
    newVectors = smote.generate(countToGenerate).
      map(vector => {
        let maxIx = vector.length - labelColumnCount;
        let features = vector.slice(0, maxIx).map(o => Math.floor(o));
        let resultLabel = Math.floor(tf.tensor(vector.slice(maxIx)).dataSync());

        /* Tipo movimiento */
        features.push(resultLabel);

        /* Actualiza maximo encontrado */
        for (let i = 0; i < features.length - 1; i++) {
          if (readData.dataMax[i] < features[i]) {
            readData.dataMax[i] = features[i];
          }
        }

        return features;
      });
  });

  readData.samples = readData.samples.concat(newVectors);

  return readData;
}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

module.exports = {
  MOVE_TYPE,
  loadJSON,
  splitData
};