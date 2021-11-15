require('@tensorflow/tfjs-node');
const loadJSON = require('./load-training-json');
const LinearRegression = require('./linear-regression');
// const plot = require('node-remote-plot');

const jsonFileName = 'reunion_formatted.json';

let { features, labels, testFeatures, testLabels } = loadJSON('./data/' + jsonFileName, {
    shuffle: true,
    splitTest: 50
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10  // With batchsize of 1 it turns into StochasticGradientDescent
});

regression.train();
const R2 = regression.test(testFeatures, testLabels);

alert("R2: ", R2);

// plot({
//     x: regression.mseHistory.reverse(),
//     xLabel: 'Iteration #',
//     yLabel: 'Mean Squared Error',
//     name: 'plotMSE'
// }).then(function () {
//     plot({
//         x: regression.bHistory,
//         y: regression.mseHistory.reverse(),
//         xLabel: 'Value of B',
//         yLabel: 'Mean Squared Error',
//         name: 'plotMSEvsB'
//     })
// });

regression.predict([
    [50], /* attention */
    [51],  /* meditation */
    [1582550], /* delta */
    [25431], /* theta */
    [5550], /* lowAlpha */
    [3759], /* highAlpha */
    [2067], /* lowBeta */
    [1922], /* highBeta */
    [457], /* lowGamma */
    [54222] /* highGamma */
]).print();