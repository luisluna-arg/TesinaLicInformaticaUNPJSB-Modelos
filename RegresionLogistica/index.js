require('@tensorflow/tfjs-node');
const loadJSON = require('./load-training-json');
const LogisticRegression = require('./logistic-regression');
const _ = require('lodash');

const jsonFileName = 'reunion_formatted.json';

let { features, labels, testFeatures, testLabels, scaledData, combinedData } = loadJSON('./data/' + jsonFileName, {
    shuffle: true,
    splitTest: 50
});

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 40  // With batchsize of 1 it turns into StochasticGradientDescent
});

// console.log("labels", labels);
// console.log("testLabels", testLabels);

regression.train();
const precision = regression.test(testFeatures, testLabels);
console.log("Precisión: ", precision);

console.log("Predicción");

const sampleToPredict = 15;
const featuresToPredict = _.slice(features, 0, sampleToPredict);
// const scaledDataToPredict = _.slice(scaledData, 0, sampleToPredict);
// const labelsToPredict = _.slice(labels, 0, sampleToPredict);

// console.log("featuresToPredict", featuresToPredict);
// console.log("scaledDataToPredict", scaledDataToPredict);
// console.log("labelsToPredict", labelsToPredict);

regression.predict(
    /* attention, meditation, delta, theta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma, highGamma */
    featuresToPredict
).print();