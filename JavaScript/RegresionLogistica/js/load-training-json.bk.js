const fs = require('fs');
const shuffleSeed = require('shuffle-seed');
const _ = require('lodash');
const tf = require('@tensorflow/tfjs');
const { forEach } = require('lodash');

module.exports = function readJSON(
  filename,
  {
    shuffle = false,
    splitTest = false,
    waveType = {
      DELTA_THETA: 0,
      ALPHA: 1,
      BETA: 2,
      GAMMA: 3
    }
  }
) {
  let data = JSON.parse(fs.readFileSync(filename, { encoding: 'utf-8' }));
  data = data.filter(o => o.poorSignalLevel === 0);

  let labelNames = [];
  labelNames[waveType.DELTA_THETA] = "LEFT";
  labelNames[waveType.ALPHA] = "TOP";
  labelNames[waveType.BETA] = "RIGHT";
  labelNames[waveType.GAMMA] = "BOTTOM";

  let labels = [];
  let scaledData = [];
  let combinedData = [];
  let features = [
    /* attention */
    /* meditation */
    /* delta */
    /* theta */
    /* lowAlpha */
    /* highAlpha */
    /* lowBeta */
    /* highBeta */
    /* lowGamma */
    /* highGamma */
  ];

  /* Filter no lecture records */
  let localData = data.filter(o => o.poorSignalLevel == 0);

  let dataIndexes = localData.map((record, index) => index);

  if (shuffle) {
    dataIndexes => shuffleSeed.shuffle(dataIndexes, 'phrase');
  }

  let minValues = [0, 0, 0, 0];
  let maxValues = [0, 0, 0, 0];

  dataIndexes.forEach(index => {
    let record = localData[index];

    /* Map features to a simpler object */
    features.push([
      record.eSense.attention,
      record.eSense.meditation,
      record.eegPower.delta,
      record.eegPower.theta,
      record.eegPower.lowAlpha,
      record.eegPower.highAlpha,
      record.eegPower.lowBeta,
      record.eegPower.highBeta,
      record.eegPower.lowGamma,
      record.eegPower.highGamma
    ]);

    /* Map data to a accumulated values to determine classification */
    let localCombinedData = [
      record.eegPower.delta + record.eegPower.theta,
      record.eegPower.lowAlpha + record.eegPower.highAlpha,
      record.eegPower.lowBeta + record.eegPower.highBeta,
      record.eegPower.lowGamma + record.eegPower.highGamma
    ];

    for (let i = 0; i < localCombinedData.length; i++) {
      if (localCombinedData[i] < minValues[i]) minValues[i] = localCombinedData[i];
      if (localCombinedData[i] > maxValues[i]) maxValues[i] = localCombinedData[i];
    }

    combinedData.push(localCombinedData);
  });

  /*
  * Normalize values
  * Normalization formula: normalized_value = (value − min_value) / (max_value − min_value)
  */
  dataIndexes.forEach(index => {
    let localScaled = {};

    let localCombinedData = combinedData[index];

    for (let i = 0; i < localCombinedData.length; i++) {
      let minValue = minValues[i];
      let maxValue = maxValues[i];
      let value = localCombinedData[i];
      let difMaxMin = maxValue - minValue;
      localScaled[i] = difMaxMin > 0 ? ((value - minValue) / (difMaxMin)) : 0;
    }

    scaledData.push(localScaled);

    let currentLabel = _.orderBy([
      [waveType.DELTA_THETA, localScaled.deltaTheta],
      [waveType.ALPHA, localScaled.alpha],
      [waveType.BETA, localScaled.beta],
      [waveType.GAMMA, localScaled.gamma]
    ], o => o[1]).reverse()[0];

    labels.push(currentLabel[0]);
  });

  if (splitTest) {
    const trainSize = _.isNumber(splitTest)
      ? splitTest
      : Math.floor(data.length / 2);

    return {
      features: features.slice(trainSize),
      labels: labels.slice(trainSize),
      testFeatures: features.slice(0, trainSize),
      testLabels: labels.slice(0, trainSize),
      combinedData,
      scaledData
    };
  } else {
    return { features, labels };
  }
};
