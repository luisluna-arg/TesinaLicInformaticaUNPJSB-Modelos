const { MOVE_TYPE, loadJSON } = require('./js/load-training-json');
const { LogisticRegression } = require('./js/logistic-regression-model');
const _ = require('lodash');
const { forEach } = require('lodash');

const filePath = './data/';

let data = _.shuffle(
    loadJSON(filePath + '/ABAJO.json', MOVE_TYPE.DOWN)
        .concat(loadJSON(filePath + '/ARRIBA.json', MOVE_TYPE.UP))
        .concat(loadJSON(filePath + '/IZQUIERDA.json', MOVE_TYPE.LEFT))
        .concat(loadJSON(filePath + '/DERECHA.json', MOVE_TYPE.RIGHT))
    );

const testPercentage = 15;
const trainingLength = data.length - (data.length * testPercentage / 100);

const trainingData = data.slice(0, trainingLength);
const testData = data.slice(trainingLength);

const regression = new LogisticRegression(trainingData, {
    learningRate: 0.5,
    iterations: 1000,
    batchSize: 100, // With batchsize of 1 it turns into StochasticGradientDescent
    tolerance: 0.8
});

// // console.log("regression", regression);
// // console.log("labels", labels);
// // console.log("testLabels", testLabels);

regression.train().then((result) => {
    // const precision = regression.test(testFeatures, testLabels);
    // console.log("Precisión: ", precision);

    // console.log("Predicción");

    /* attention, meditation, delta, theta, lowAlpha, highAlpha, lowBeta, highBeta, lowGamma, highGamma */
    regression.test(testData);

});

const outputs = regression.model.layers.map(o => o.output)

for (let i = 0; i < outputs.length; i++) {
    console.log("output " + i);
    console.log(outputs[i]);
}
