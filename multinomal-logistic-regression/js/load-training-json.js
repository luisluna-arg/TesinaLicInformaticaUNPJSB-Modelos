const fs = require('fs');
const _ = require('lodash');

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
    tolerance: 1,
    split: false
  }, settings);

  /* Maximos: Para validar la tolerancia de las lecturas */
  const dataMax = {
    attention: 0,
    meditation: 0,
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
          attention: record.eSense.attention,
          meditation: record.eSense.meditation,
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

  if (localSettings.split){
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
    tolerance: 1
  }, settings);

  // Step 1. Shuffle the data
  let samples = data.samples;
  if (localSettings.shuffle) { samples = _.shuffle(samples); }

  // Step 2. Split into features and labels
  const features = [];
  const labels = [];

  _.each(samples, (dataItem) => {
    let regularSample = [
      dataItem.attention,
      dataItem.meditation,
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
      if (data.dataMax[property] > 0 && dataItem[property] / data.dataMax[property] > localSettings.tolerance) {
        features.push(regularSample);
        labels.push([dataItem.down, dataItem.up, dataItem.right, dataItem.left]);
        break;
      }
    }
  });

  return {
    features,
    labels
  };
}


/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

module.exports = {
  MOVE_TYPE,
  loadJSON,
  splitData
};