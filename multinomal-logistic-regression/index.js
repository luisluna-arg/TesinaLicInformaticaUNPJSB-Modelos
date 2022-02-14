const models = {
    TF_COREAPI: 0,
    TF_LAYERMODEL: 1,
}

/* 
 * Ejecutar local con "node --inspect index.js"
 * Cambiar ModelType para alternar entre modelos manual y de capas
 */

/* Definir modelo a usar */
const ModelType = models.TF_COREAPI;

const { MOVE_TYPE, loadJSON, splitData } = require('./js/load-training-json');
const _ = require('lodash');

// const { LogisticRegression } = require('./js/logistic-regression');
const { LogisticRegression } = (() => {
    switch (ModelType) {
        case models.TF_COREAPI: {
            console.log("TIPO MODELO: TF_COREAPI");
            console.log("==================");
            console.log("");
            return require('./js/logistic-regression');
        }
        default: {
            console.log("TIPO MODELO: TF_LAYERMODEL");
            console.log("=========================");
            console.log("");
            return require('./js/logistic-regression-model');
        }
    }
})();

const fileBasePath = './data/';

const filesToLoad = [
    { file: fileBasePath + '/ABAJO.json', moveType: MOVE_TYPE.DOWN },
    { file: fileBasePath + '/ARRIBA.json', moveType: MOVE_TYPE.UP },
    { file: fileBasePath + '/IZQUIERDA.json', moveType: MOVE_TYPE.LEFT },
    { file: fileBasePath + '/DERECHA.json', moveType: MOVE_TYPE.RIGHT }
];

const trainingCount = 25;

let lodedData = loadJSON(filesToLoad, { shuffle: false, split: false });

console.log("");
console.log("Testings");
console.log("========");
console.log("");

const batchSize = 0.5; /* Porcentaje del tamaño de la muestra */

let regressionSettings = {
    learningRate: 0.75,
    iterations: 60,
    batchSize: Math.floor(lodedData.samples.length * batchSize),
    useReLu: false,
    shuffle: false
};

let splitDataSettings = {
    shuffle: true,
    tolerance: 0.35
};

console.log("Regression Settings: " + JSON.stringify(regressionSettings));
console.log("Split Data Settings: " + JSON.stringify(splitDataSettings));
console.log("");

let promises = [];
let testings = [];
for (let i = 0; i < trainingCount; i++) {
    let { features, labels } = splitData(lodedData, splitDataSettings);

    const testPercentage = 15;
    const trainingLength = features.length - (features.length * testPercentage / 100);
    //const trainingLength = features.length - 20;

    const trainingData = features.slice(0, trainingLength);
    const trainingLabels = labels.slice(0, trainingLength);

    const testData = features.slice(trainingLength);
    const testLabels = labels.slice(trainingLength);

    const regression = new LogisticRegression(trainingData, trainingLabels, regressionSettings);

    if (ModelType == models.TF_LAYERMODEL) {
        promises.push(
            regression.train(function(logs) {
                let result = regression.test(testData, testLabels);
                testings.push(result);
                console.log("Precision [" + (i + 1) + "]", result);
            }))
            ;
    }
    else {
        regression.train();
        let result = regression.test(testData, testLabels);
        testings.push(result);
        console.log("Precision [" + (i + 1) + "]", result);
    }
}

const rounding = 2;

function showResults(currentResults) {
    console.log("Promedio: " + _.round((_.sum(currentResults) / currentResults.length), rounding) +
        ", Min: " + _.round(_.min(currentResults), rounding) +
        ", Max: " + _.round(_.max(currentResults), rounding)
    );
}

if (ModelType == models.TF_LAYERMODEL) {
    Promise.all(promises).then(results => { showResults(testings); });
} else {
    showResults(testings);
}


