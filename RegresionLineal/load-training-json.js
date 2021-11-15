const fs = require('fs');
const shuffleSeed = require('shuffle-seed');
const _ = require('lodash');

module.exports = function loadJSON(
  filename,
  {
    shuffle = false,
    splitTest = false,
  }
) {
  let data = JSON.parse(fs.readFileSync(filename, { encoding: 'utf-8' }));
  data = data.filter(o => o.poorSignalLevel === 0);

  let waveType = {
    DELTA_THETA: 0,
    ALPHA: 1,
    BETA: 2,
    GAMMA: 3
  };

  let labelNames = [];
  labelNames[waveType.DELTA_THETA] = "LEFT";
  labelNames[waveType.ALPHA] = "TOP";
  labelNames[waveType.BETA] = "RIGHT";
  labelNames[waveType.GAMMA] = "BOTTOM";

  let labels = [];
  let features = [
    [] /* attention */
    , [] /* meditation */
    , [] /* delta */
    , [] /* theta */
    , [] /* lowAlpha */
    , [] /* highAlpha */
    , [] /* lowBeta */
    , [] /* highBeta */
    , [] /* lowGamma */
    , [] /* highGamma */
  ];

  /* Filter no lecture records */
  let localData = data.filter(o => o.poorSignalLevel == 0);

  let dataIndexes = localData.map((record, index) => index);

  if (shuffle) {
    dataIndexes => shuffleSeed.shuffle(dataIndexes, 'phrase');
  }

  dataIndexes.forEach(index => {
    let record = localData[index];

    let recordIX = 0;

    features[recordIX++].push(record.eSense.attention);
    features[recordIX++].push(record.eSense.meditation);

    features[recordIX++].push(record.eegPower.delta);
    features[recordIX++].push(record.eegPower.theta);

    features[recordIX++].push(record.eegPower.lowAlpha);
    features[recordIX++].push(record.eegPower.highAlpha);

    features[recordIX++].push(record.eegPower.lowBeta);
    features[recordIX++].push(record.eegPower.highBeta);

    features[recordIX++].push(record.eegPower.lowGamma);
    features[recordIX++].push(record.eegPower.highGamma);

    let deltaTheta = record.eegPower.delta + record.eegPower.theta;
    let alpha = record.eegPower.lowAlpha + record.eegPower.highAlpha;
    let beta = record.eegPower.lowBeta + record.eegPower.highBeta;
    let gamma = record.eegPower.lowGamma + record.eegPower.highGamma;

    let values = _.orderBy([
      [waveType.DELTA_THETA, deltaTheta],
      [waveType.ALPHA, alpha],
      [waveType.BETA, beta],
      [waveType.GAMMA, gamma],
    ], o => o[1]).pop();

    labels.push(labelNames[values[0]]);
  });

  if (splitTest) {
    const trainSize = _.isNumber(splitTest)
      ? splitTest
      : Math.floor(data.length / 2);

    let splittedFeatures = [];
    let splittedTestFeatures = [];
    for (let i = 0; i < features.length; i++) {
      splittedTestFeatures.push(features[i].slice(0, trainSize));
      splittedFeatures.push(features[i].slice(trainSize));
    }

    return {
      features: splittedFeatures,
      labels: labels.slice(trainSize),
      testFeatures: splittedTestFeatures,
      testLabels: labels.slice(0, trainSize)
    };
  } else {
    return { features: features, labels };
  }
};


//   function standarize(features) {
//     const { mean, variance } = tf.moments(features, 0);
//     this.mean = mean;
//     this.variance = variance;
//     return features.sub(mean).div(variance.pow(0.5));
//   }
//   labels = tf.tensor(labels);
//   splitted = tf.tensor(splitted);
