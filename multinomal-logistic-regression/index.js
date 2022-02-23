const methods = {
    TF_COREAPI: 0,
    TF_LAYERMODEL: 1,
    NAIVE_BAYES: 2
}

/* 
 * Ejecutar local con "node --inspect index.js"
 * Cambiar ModelType para alternar entre modelos manual y de capas
 */

/* Definir modelo a usar */
const ModelType = methods.NAIVE_BAYES;

const { MOVE_TYPE, loadJSON, splitData } = require('./js/load-training-json');
const _ = require('lodash');

const { Naivebayes } = (() => {
    switch (ModelType) {
        case methods.NAIVE_BAYES: {
            console.log("");
            console.log("TIPO MODELO: NAIVE_BAYES");
            console.log("==================");
            console.log("");
            return require('./js/naive-bayes');
        }
        default: {
            return { Naivebayes: null };
        }
    }
})();

const { LogisticRegression } = (() => {
    switch (ModelType) {
        case methods.TF_COREAPI: {
            console.log("");
            console.log("TIPO MODELO: TF_COREAPI");
            console.log("==================");
            console.log("");
            return require('./js/logistic-regression');
        }
        case methods.TF_LAYERMODEL: {
            console.log("");
            console.log("TIPO MODELO: TF_LAYERMODEL");
            console.log("=========================");
            console.log("");
            return require('./js/logistic-regression-model');
        }
        default: {
            return { Naivebayes: null };
        }
    }
})();

const fileBasePath = './data/Full';
//const fileBasePath = './data/2022.2.22';

const filesToLoad = [
    { file: fileBasePath + '/ABAJO.json', moveType: MOVE_TYPE.DOWN },
    { file: fileBasePath + '/ARRIBA.json', moveType: MOVE_TYPE.UP },
    { file: fileBasePath + '/IZQUIERDA.json', moveType: MOVE_TYPE.LEFT },
    { file: fileBasePath + '/DERECHA.json', moveType: MOVE_TYPE.RIGHT }
];

let lodedData = loadJSON(filesToLoad, { shuffle: false, split: false });

console.log("");
console.log("Testings");
console.log("========");
console.log("");

const batchSize = 0.5; /* Porcentaje del tamaño de la muestra */

const regressionSettings = {
    learningRate: 0.5,
    iterations: 200,
    batchSize: Math.floor(lodedData.samples.length * batchSize),
    useReLu: false,
    shuffle: false
};

const naiveBayesSettings = {
    MoveTypeEnum: MOVE_TYPE,
    verbose: false,
    normalize: true,
    useReLu: true
};

const splitDataSettings = {
    shuffle: true,
    tolerance: 0.7
};

const testPercentage = 5;
const trainingCount = 25;

console.log("Regression Settings: " + JSON.stringify(regressionSettings));
console.log("Split Data Settings: " + JSON.stringify(splitDataSettings));
console.log("");

let promises = [];
let testings = [];
for (let i = 0; i < trainingCount; i++) {
    let { features, labels } = splitData(lodedData, splitDataSettings);

    const trainingLength = features.length - (features.length * testPercentage / 100);
    //const trainingLength = features.length - 20;

    const trainingData = features.slice(0, trainingLength);
    const trainingLabels = labels.slice(0, trainingLength);

    const testData = features.slice(trainingLength);
    const testLabels = labels.slice(trainingLength);


    switch  (ModelType) {
        case methods.TF_LAYERMODEL: {
            const regression = new LogisticRegression(trainingData, trainingLabels, regressionSettings);
            promises.push(
                regression.train(function(logs) {
                    let result = regression.test(testData, testLabels);
                    testings.push(result);
                    console.log("Precision [" + (i + 1) + "]", result);
                })
                );
            break;
        }
        case methods.TF_COREAPI: {
            const regression = new LogisticRegression(trainingData, trainingLabels, regressionSettings);
            regression.train();
            let result = regression.test(testData, testLabels);
            testings.push(result);
            console.log("Precision [" + (i + 1) + "]", result);
            break;
        }
        case methods.NAIVE_BAYES: {
            const naiveBayes = new Naivebayes(trainingData, trainingLabels, naiveBayesSettings);
            naiveBayes.train();
            let result = naiveBayes.test(testData, testLabels);
            testings.push(result);
            console.log("Precision [" + (i + 1) + "]", result);

            //initialize our vocabulary and its size
            naiveBayes.vocabulary = {}
            naiveBayes.vocabularySize = 0

            // naiveBayes.summary();
            // naiveBayes.printTrainingData();

            break;
        }
    }
}

const rounding = 2;

function showResults(currentResults) {
    console.log("Promedio: " + _.round((_.sum(currentResults) / currentResults.length), rounding) +
        ", Min: " + _.round(_.min(currentResults), rounding) +
        ", Max: " + _.round(_.max(currentResults), rounding)
    );
}

if (ModelType == methods.TF_LAYERMODEL) {
    Promise.all(promises).then(results => { showResults(testings); });
} else {
    showResults(testings);
}


