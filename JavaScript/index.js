const methods = {
    TF_COREAPI: 0,
    TF_LAYERMODEL: 1,
    NAIVE_BAYES: 2,
    DECISION_TREE: 3
}

/* 
 * Ejecutar local con "node --inspect index.js"
 * Cambiar ModelType para alternar entre modelos manual y de capas
 */

/* Definir modelo a usar */
const ModelType = methods.DECISION_TREE;

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

const { DirectionDecisionTree } = (() => {
    switch (ModelType) {
        case methods.DECISION_TREE: {
            console.log("");
            console.log("TIPO MODELO: DECISION_TREE");
            console.log("==================");
            console.log("");
            return require('./js/decision-tree');
        }
        default: {
            return { DirectionDecisionTree: null };
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

//const fileBasePath = './data/generated';
const fileBasePath = './data/Full';
//const fileBasePath = './data/2022.2.22';

const filesToLoad = [
    { file: fileBasePath + '/ABAJO.json', moveType: MOVE_TYPE.DOWN },
    { file: fileBasePath + '/ARRIBA.json', moveType: MOVE_TYPE.UP },
    { file: fileBasePath + '/IZQUIERDA.json', moveType: MOVE_TYPE.LEFT },
    { file: fileBasePath + '/DERECHA.json', moveType: MOVE_TYPE.RIGHT }
];

let loadingSettings = {
    shuffle: true,
    split: false,
    dataAugmentation: false,
    dataAugmentationTotal: 50000,
    minTolerance: 0.0 /* 0 para que traiga todo */
};

let loadedData = loadJSON(filesToLoad, loadingSettings);

/* ************************************************************** */

console.log("");
console.log("Testings");
console.log("========");
console.log("");

const regressionSettingsCoreModel = {
    learningRate: 0.5,
    iterations: 2000,
    batchSize: 20000,
    useReLu: false,
    shuffle: !loadingSettings.shuffle,
    verbose: true
};

const regressionSettingsLayerModel = {
    learningRate: 0.000005, //0.5, 
    iterations: 5000,
    batchSize: 500,//Math.floor(loadedData.samples.length * batchSize),
    useReLu: false,
    shuffle: !loadingSettings.shuffle,
    verbose: true
};

const naiveBayesSettings = {
    MoveTypeEnum: MOVE_TYPE,
    verbose: false,
    useReLu: false
};

const decisionTreeSettings = {
    MoveTypeEnum: MOVE_TYPE,
    verbose: false,
    useReLu: false
};

const trainingExerciseCount = 1;

// console.log("Regression Settings (Core): " + JSON.stringify(regressionSettingsCoreModel));
console.log("Split Data Settings: " + JSON.stringify(loadingSettings));
console.log("");

let promises = [];
let testings = [];

// console.log("Inicio", new Date());

for (let i = 0; i < trainingExerciseCount; i++) {
    let { samples, labels } = splitData(loadedData, loadingSettings);

    /* Data de entrenamiento */
    const trainingLength = Math.floor(samples.length * 0.7);
    const trainingData = samples.slice(0, trainingLength);
    const trainingLabels = labels.slice(0, trainingLength);

    /* Data de pruebas */
    const testLength = samples.length - trainingLength;
    const testData = samples.slice(testLength);
    const testLabels = labels.slice(testLength);

    switch (ModelType) {
        case methods.TF_LAYERMODEL: {
            const regression = new LogisticRegression(trainingData, trainingLabels, regressionSettingsLayerModel);

            // promises.push(
            //     regression.train(function (logs) {
            //         console.log("train logs", logs);
            //         let result = regression.test(testData, testLabels);
            //         testings.push(result);
            //         console.log("Sample count: ", regression.samples.shape[0], ", Precision [" + (i + 1) + "]", result);
            //         regression.summary();
            //     })
            // );

            regression.testModel(testData, testLabels);

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
        case methods.DECISION_TREE: {
            console.log("Crear Arbol mediante muestras");
            const decisionTree = new DirectionDecisionTree(trainingData, trainingLabels, decisionTreeSettings);
            decisionTree.train();
            let result = decisionTree.test(testData, testLabels);
            testings.push(result);
            console.log("Precision [" + (i + 1) + "]", result);

            let localJson = decisionTree.toJSON();

            console.log("");
            console.log("Crear Arbol mediante JSON");
            const decisionTreeJSON = new DirectionDecisionTree(localJson);
            decisionTreeJSON.train();
            let JSONResult = decisionTreeJSON.test(testData, testLabels);
            console.log("Precision JSON [" + (i + 1) + "]", JSONResult);

            decisionTree.summary();
            // decisionTree.printTrainingData();

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
    // console.log("Fin", new Date());
}

if (ModelType == methods.TF_LAYERMODEL) {
    Promise.all(promises).then(results => { showResults(testings); });
} else {
    showResults(testings);
}
