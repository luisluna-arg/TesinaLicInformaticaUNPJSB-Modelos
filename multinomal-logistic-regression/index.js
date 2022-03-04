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
const ModelType = methods.TF_LAYERMODEL;

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

const fileBasePath = './data/generated';
// const fileBasePath = './data/Full';
//const fileBasePath = './data/2022.2.22';

const filesToLoad = [
    { file: fileBasePath + '/ABAJO.json', moveType: MOVE_TYPE.DOWN },
    { file: fileBasePath + '/ARRIBA.json', moveType: MOVE_TYPE.UP },
    { file: fileBasePath + '/IZQUIERDA.json', moveType: MOVE_TYPE.LEFT },
    { file: fileBasePath + '/DERECHA.json', moveType: MOVE_TYPE.RIGHT }
];

let loadedData = loadJSON(filesToLoad, { 
    shuffle: false, split: false, dataAugmentation: false,
    dataAugmentationTotal: 50000
});

console.log("");
console.log("Testings");
console.log("========");
console.log("");


// const batchSize = 0.5; /* Porcentaje del tamaño de la muestra */

const regressionSettingsCoreModel = {
    learningRate: 0.5, //0.5, 
    iterations: 2000,
    batchSize: 20000,//Math.floor(loadedData.samples.length * batchSize),
    useReLu: false,
    shuffle: false,
    verbose: true
};

const regressionSettingsLayerModel = {
    learningRate: 0.00005, //0.5, 
    iterations: 800,
    batchSize: 10000,//Math.floor(loadedData.samples.length * batchSize),
    useReLu: false,
    shuffle: false,
    verbose: true
};

const naiveBayesSettings = {
    MoveTypeEnum: MOVE_TYPE,
    verbose: false,
    normalize: true,
    useReLu: false
};

const splitDataSettings = {
    shuffle: true,
    minTolerance: 0.0
};

const testPercentage = 2;
const trainingExerciseCount = 1;

// console.log("Regression Settings (Core): " + JSON.stringify(regressionSettingsCoreModel));
console.log("Split Data Settings: " + JSON.stringify(splitDataSettings));
console.log("");

let promises = [];
let testings = [];

console.log("Inicio", new Date());

for (let i = 0; i < trainingExerciseCount; i++) {
    let { samples, labels } = splitData(loadedData, splitDataSettings);

    // regressionSettings.batchSize = Math.floor(samples.length * batchSize);

    // const trainingLength = samples.length - (samples.length * testPercentage / 100);
    const trainingLength = samples.length - 20;

    const trainingData = samples.slice(0, trainingLength);
    const trainingLabels = labels.slice(0, trainingLength);

    const testData = samples.slice(trainingLength);
    const testLabels = labels.slice(trainingLength);


    switch  (ModelType) {
        case methods.TF_LAYERMODEL: {
            const regression = new LogisticRegression(trainingData, trainingLabels, regressionSettingsLayerModel);
            promises.push(
                regression.train(function(logs) {
                    console.log("train logs", logs);
                    let result = regression.test(testData, testLabels);
                    testings.push(result);
                    console.log("Sample count: ", regression.samples.shape[0], ", Precision [" + (i + 1) + "]", result);
                    regression.summary();
                })
                );
            break;
        }
        case methods.TF_COREAPI: {
            const regression = new LogisticRegression(trainingData, trainingLabels, regressionSettingsCoreModel);
            regression.train();
            let result = regression.test(testData, testLabels);
            testings.push(result);
            console.log("Precision [" + (i + 1) + "]", result);
            // regression.summary();
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
    console.log("Fin", new Date());
}

if (ModelType == methods.TF_LAYERMODEL) {
    Promise.all(promises).then(results => { showResults(testings); });
} else {
    showResults(testings);
}
