const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const _ = require('lodash');
const SMOTE = require('smote');
const { preProcess } = require('./data-preprocessing');

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

/* ****************************************************** */

/**
 * Calcula los valores maximos para una coleccion de muestras, por columnas
 * @param {*} samples Muestras a analizar
 * @returns Arreglo de maximos de igual dimension que las muestras analizadas
 */
function calculateDataMaxes(samples) {
  let columnCount = samples[0].length;
  let maxes = new Array(columnCount-1).fill(0);

  for (let i = 0; i < samples.length; i++) {
    const currentSample = samples[i];
    for (let j = 0; j < columnCount - 1; j++) {
      if (maxes[j] < currentSample[j]) maxes[j] = currentSample[j];
    }
  }

  return maxes;
}

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
    applyFFT: true,
    dataAugmentationTotal: 1000,
  }, settings);

  /* Maximos: Para validar la tolerancia de las lecturas */
  const dataMax = [];
  dataMax.fill(0, 0, 8);

  let samples = [];
  _.each(fileData, (fileSettings) => {
    /* Read lecture as JSON */
    let data = JSON.parse(
      fs.readFileSync(fileSettings.file, { encoding: 'utf-8' })
    ).filter(o => o.poorSignalLevel === 0);

    let newData = data.
      filter(o => o.poorSignalLevel == 0).
      map((record) => {
        /* Crea el objeto de datos */
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
        for(let i = 0; i < dataItem.length - 2; i++) {
          if (dataMax[i] < dataItem[i]) dataMax[i] = dataItem[i];
        }

        return dataItem;
      });

    samples = _.concat(samples, newData);
  });

  let readData = {
    samples,
    dataMax
  };

  if (localSettings.dataAugmentation) {
    readData = dataAugmentation(readData, localSettings);
  }

  if (localSettings.applyFFT) {
    labelColumn = 8;
    readData.samples = preProcess(readData.samples, labelColumn);
    readData.dataMax = calculateDataMaxes(readData.samples);
  }

  if (localSettings.split) {
    return splitData(readData, localSettings);
  }
  else {
    return readData;
  }

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
    minTolerance: 0,
  }, settings);

  // Step 1. Shuffle the data
  let dataSamples = data.samples;
  if (localSettings.shuffle) { dataSamples = _.shuffle(dataSamples); }

  // Step 2. Split into samples and labels
  const finalSamples = [];
  const finalLabels = [];
  const labelColumnCount = 1;

  _.each(dataSamples, (dataArray) => {
    let regularSample = dataArray.slice(0, dataArray.length - labelColumnCount);

    /* Se normaliza cada variable de medicion en un rango de 0 a 1, 
    *  usando su porcentaje respecto del maximo valor observado en toda la muestra
    *  Si el valor observado supero el % de tolerancia, se lo considera valor aceptable y
    *  se usan sus labels.
    *  Si no se supera ese % de tolerancia minimo, se desactivan las labels (array de 0)
    *  para descartar la muestra
    */
    for(let i = 0; i < regularSample.length; i++) {
      let propertyMax = data.dataMax[i];
      let propertyValue = regularSample[i];
      
      if (propertyMax > 0 && propertyValue / propertyMax > localSettings.minTolerance) {
        finalSamples.push(regularSample);
        finalLabels.push(dataArray[dataArray.length - 1]);
        break;
      }
    }

  });

  return {
    samples: finalSamples,
    labels: finalLabels
  };
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