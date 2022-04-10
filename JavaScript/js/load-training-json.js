const fs = require('fs');
const _ = require('lodash');
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

  getLabelCount = function(data) {
    const labelGetter = (labelContainer) => Array.isArray(labelContainer) ? labelContainer[labelContainer.length - 1] : labelContainer;
    let labelCount = {};
    data.forEach(item => { 
      const label = labelGetter(item);
      if (isNullOrUndef(labelCount[label])) labelCount[label] = 0;
      labelCount[label]++;
    });
    return labelCount;
  };

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
      let labelCount = this.getLabelCount(this.labels);

      console.log("Shapes Samples|Labels:", samplesShape, labelsShape);
      console.log("Feature maxes:", maxes);
      // console.log("Label count:", labelCount);
    }
    else {
      let labelCount = this.getLabelCount(this.samples);
      console.log("Shapes Samples:", samplesShape);
      console.log("Feature maxes:", maxes);
      // console.log("Label count:", labelCount);
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

  readData = dataPreProcessing(readData, localSettings);

  if (localSettings.split) {
    return splitData(readData);
  }

  return readData;
}

function dataPreProcessing(readData, localSettings) {
  const columnCount = 1;

  console.log("dataPreProcessing.readData", readData.samples.length);
  console.log("dataPreProcessing.readData.samples[0]", readData.samples[0].length);
  console.log("dataPreProcessing.readData.FEATURE_NAMES", readData.FEATURE_NAMES.length);

  // if (localSettings.minTolerance > 0) {
  //   readData = filterData(readData, localSettings);
  // }
  
  let { result, featureNames } = preProcess(readData.samples, readData.FEATURE_NAMES, localSettings);

  readData.samples = result;
  readData.FEATURE_NAMES = featureNames;

  if (localSettings.deviationMatrix) {
    readData.FEATURE_NAMES = featureNamesForDevMatrix(readData.FEATURE_NAMES);
  }

  readData.calculateDataMaxes(columnCount);

  return readData;
}

// function filterData(data, settings) {
//   const localSettings = Object.assign({
//     minTolerance: 0,
//   }, settings);

//   let finalSamples = []
//   const columnCount = 1;

//   /* Se normaliza cada variable de medicion en un rango de 0 a 1, 
//     *  usando su porcentaje respecto del maximo valor observado en toda la muestra
//     *  Si el valor observado supero el % de tolerancia, se lo considera valor aceptable y
//     *  se usan sus labels.
//     *  Si no se supera ese % de tolerancia minimo, se desactivan las labels (array de 0)
//     *  para descartar la muestra
//     */
//   for (let i = 0; i < data.samples.length; i++) {
//     const sample = data.samples[i];
//     const features = sample.slice(0, sample.length - columnCount); /* Sample array */

//     for (let featureIndex = 0; featureIndex < features.length; featureIndex++) {
//       let featureMax = data.dataMax[featureIndex];
//       let featureValue = features[featureIndex];
//       if (featureMax > 0 && featureValue / featureMax > localSettings.minTolerance) {
//         finalSamples.push(sample);
//         break;
//       }
//     }
//   }

//   data.samples = finalSamples;

//   return data;

// }

/**
 * Aleatoriza los datos y los separa en arreglos de muestras y etiquetas
 * Tambien ajusta las etiquetas de acuerdo al nivel de tolerancia fijado en options
 * @param data Datos leídos de archivos
 * @param settings Settings generales para carga de archivo
 * @returns Data separada en muestras y etiquetas
 */
function splitData(data) {
  // Split into samples and labels
  let finalSamples = [];
  let finalLabels = [];
  const labelColumnCount = 1;

  _.each(data.samples, (dataArray) => {
    const featureCount = dataArray.length - labelColumnCount;
    let features = dataArray.slice(0, featureCount);
    finalSamples.push(features);
    let labels = dataArray.slice(featureCount);
    
    if (Array.isArray(labels)) {
      finalLabels.push(labels.length == 1 ? labels[0] : labels);
    }
    else {
      finalLabels.push(labels);
    }
  });

  data.labels = finalLabels;
  data.samples = finalSamples;

  return data;
}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

module.exports = {
  MOVE_TYPE,
  loadJSON,
  splitData,
  dataPreProcessing
};