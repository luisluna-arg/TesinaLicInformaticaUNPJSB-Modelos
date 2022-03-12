const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const _ = require('lodash');
const SMOTE = require('smote');

const MOVE_TYPE = {
  NONE: 0,
  DOWN: 1,
  UP: 2,
  LEFT: 3,
  RIGHT: 4
};

/**
 * Lee archivos pasados por parametro para luego convertirlos en objectos manejables y etiquetados
 * @param {*} fileData 
 * @param {*} Settings
 * @returns 
 */
function loadJSON(fileData, settings) {

  const localSettings = Object.assign({
    shuffle: false,
    minTolerance: 0,
    split: false,
    dataAugmentation: false,
    dataAugmentationTotal: 25000,
  }, settings);

  /* Maximos: Para validar la tolerancia de las lecturas */
  const dataMax = {
    delta: 0,
    theta: 0,
    lowAlpha: 0,
    highAlpha: 0,
    lowBeta: 0,
    highBeta: 0,
    lowGamma: 0,
    highGamma: 0,
  };

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
        let dataItem = {
          delta: record.eegPower.delta,
          theta: record.eegPower.theta,
          lowAlpha: record.eegPower.lowAlpha,
          highAlpha: record.eegPower.highAlpha,
          lowBeta: record.eegPower.lowBeta,
          highBeta: record.eegPower.highBeta,
          lowGamma: record.eegPower.lowGamma,
          highGamma: record.eegPower.highGamma,
          /* Tipo movimiento */
          down: fileSettings.moveType == MOVE_TYPE.DOWN ? 1 : 0,
          up: fileSettings.moveType == MOVE_TYPE.UP ? 1 : 0,
          right: fileSettings.moveType == MOVE_TYPE.RIGHT ? 1 : 0,
          left: fileSettings.moveType == MOVE_TYPE.LEFT ? 1 : 0
        }

        /* Actualiza maximo encontrado */
        for (let property in dataItem) {
          if (dataMax.hasOwnProperty(property) && dataMax[property] < dataItem[property])
            dataMax[property] = dataItem[property];
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
 * @param {*} data 
 * @param {*} settings
 * @returns 
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

  _.each(dataSamples, (dataItem) => {
    let regularSample = [
      dataItem.delta,
      dataItem.theta,
      dataItem.lowAlpha,
      dataItem.highAlpha,
      dataItem.lowBeta,
      dataItem.highBeta,
      dataItem.lowGamma,
      dataItem.highGamma
    ];

    /* Se normaliza cada variable de medicion en un rango de 0 a 1, 
    *  usando su porcentaje respecto del maximo valor observado en toda la muestra
    *  Si el valor observado supero el % de tolerancia, se lo considera valor aceptable y
    *  se usan sus labels.
    *  Si no se supera ese % de tolerancia minimo, se desactivan las labels (array de 0)
    *  para descartar la muestra
    */
    for (var property in dataItem) {
      let propertyMax = data.dataMax[property];
      let propertyValue = dataItem[property];
      if (propertyMax > 0 && propertyValue / propertyMax > localSettings.minTolerance
      ) {
        finalSamples.push(regularSample);
        finalLabels.push([dataItem.down, dataItem.up, dataItem.right, dataItem.left]);

        break;
      }
    }
  });

  return {
    samples: finalSamples,
    labels: finalLabels
  };
}

function dataAugmentation(readData, settings) {

  // Here we generate 5 synthetic data points to bolster our training data with an balance an imbalanced data set.
  const countToGenerate = settings.dataAugmentationTotal - readData.samples.length;

  if (countToGenerate <= 0) return readData;

  let vectorizedSamples = readData.samples.map((dataItem) => {
    let result = [
      dataItem.delta,
      dataItem.theta,
      dataItem.lowAlpha,
      dataItem.highAlpha,
      dataItem.lowBeta,
      dataItem.highBeta,
      dataItem.lowGamma,
      dataItem.highGamma,
      /* Tipo movimiento */
      dataItem.down,
      dataItem.up,
      dataItem.right,
      dataItem.left
    ];

    return result;
  });


  // Pass in your real data vectors.
  const smote = new SMOTE(vectorizedSamples);

  let newVectors;

  tf.tidy(() => {
    newVectors = smote.generate(countToGenerate).
      map(vector => {
        let maxIx = vector.length - 4;
        let features = vector.slice(0, maxIx).map(o => Math.floor(o));
        let activatedLabelIndex = tf.tensor(vector.slice(maxIx)).softmax().argMax().dataSync();
        let resultLabels = [0, 0, 0, 0];
        resultLabels[activatedLabelIndex] = 1;

        let propIndex = 0;
        let labelIndex = 0;

        let dataItem = {
          delta: Math.floor(features[propIndex++]),
          theta: Math.floor(features[propIndex++]),
          lowAlpha: Math.floor(features[propIndex++]),
          highAlpha: Math.floor(features[propIndex++]),
          lowBeta: Math.floor(features[propIndex++]),
          highBeta: Math.floor(features[propIndex++]),
          lowGamma: Math.floor(features[propIndex++]),
          highGamma: Math.floor(features[propIndex++]),
          /* Tipo movimiento */
          down: Math.floor(resultLabels[labelIndex++]),
          up: Math.floor(resultLabels[labelIndex++]),
          right: Math.floor(resultLabels[labelIndex++]),
          left: Math.floor(resultLabels[labelIndex++])
        };


        /* Actualiza maximo encontrado */
        for (let property in dataItem) {
          if (readData.hasOwnProperty(property) && readData[property] < readData[property])
            readData.dataMax[property] = dataItem[property];
        }

        return dataItem;
      });
  });

  readData.samples = readData.samples.concat(newVectors);

  return readData
}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

module.exports = {
  MOVE_TYPE,
  loadJSON,
  splitData
};