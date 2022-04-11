/* 
 * Ejecutar local con "node --inspect index.js"
 * Cambiar ModelType para alternar entre modelos manual y de capas
 */

const { MOVE_TYPE, loadJSON, splitData, dataPreProcessing } = require('./js/load-training-json');
const _ = require('lodash');

/* ************************************************************** */
/* ************************************************************** */
/* ************************************************************** */

let enumValue = 0;
const methods = {
    TF_LAYERMODEL: enumValue++,
    NAIVE_BAYES: enumValue++,
    DECISION_TREE: enumValue++,
    /* ************************* */
    NEURO: enumValue++,
    CONV_NET_LSTM: enumValue++,
    NET_LSTM: enumValue++,
    /* ************************* */
    TF_COREAPI: enumValue++, 
}


const printHeader = function (header) {
    console.log("");
    console.log(header);
    console.log(new Array(header.length + 1).fill("=").join(''));
    console.log("");
}

const printSubHeader = function (subheader) {
    console.log("");
    console.log("> " + subheader);
}

const loadModel = (modelType) => {
    const FILE_BASE_PATH = './js/';
    let fileName = null;
    let modelName = null;
    
    switch (modelType) {
        case methods.CONV_NET_LSTM: {
            modelName = 'CONV_NET_LSTM';
            fileName = FILE_BASE_PATH + 'ConvNetLSTM';
            break;
        }
        case methods.NAIVE_BAYES: {
            modelName = 'NAIVE_BAYES';
            fileName = FILE_BASE_PATH + 'naive-bayes';
            break;
        }
        case methods.DECISION_TREE: {
            modelName = 'DECISION_TREE'
            fileName = FILE_BASE_PATH + 'decision-tree';
            break;
        }
        case methods.NEURO: {
            modelName = "NEURO";
            fileName = FILE_BASE_PATH + 'neuronalNetwork';
            break;
        }
        case methods.TF_COREAPI: {
            modelName = 'TF_COREAPI';
            fileName = FILE_BASE_PATH + 'logistic-regression';
            break;
        }
        case methods.TF_LAYERMODEL: {
            modelName = 'TF_LAYERMODEL';
            fileName = FILE_BASE_PATH + 'logistic-regression-model';
            break;
        }
        case methods.NET_LSTM: {
            modelName = 'NET_LSTM';
            fileName = FILE_BASE_PATH + 'NetWithLSTM';
            break;
        }
    }

    if (fileName == null) return null;

    printHeader("TIPO MODELO: " + modelName);
    return require(fileName).model;
}


/* Definir modelo a usar */
const ModelType = methods.NAIVE_BAYES;
const ModelClass = loadModel(ModelType)


/* ************************************************************** */
/* ************************************************************** */
/* ************************************************************** */

let defaultSettings = {
    filter: false,
    shuffle: true,
    split: false,
    truncate: false,
    decimals: 2,
    normalization: false,
    fourier: false,
    deviationMatrix: false,
    dataAugmentation: false,
    selectFeatures: false,
    dataAugmentationTotal: 40000, /* Muestras totales cada vez que un un archivo o lista de archivos es aumentado */
    minTolerance: 0.0 /* entre 0 y 1, 0 para que traiga todo */
};

let loadingSettings = {};
Object.assign(loadingSettings, defaultSettings);

switch (ModelType) {
    case methods.NAIVE_BAYES: {
        Object.assign(loadingSettings, {
            truncate: false,
            decimals: 4,
            normalization: true,
            fourier: false,
            deviationMatrix: true
        });
        break;
    }
    case methods.DECISION_TREE: {
        Object.assign(loadingSettings, {
            selectFeatures: false,
            filter: false,
            truncate: true,
            decimals: 4,
            normalization: true,
            fourier: true,
            deviationMatrix: true,
            dataAugmentation: false,
            dataAugmentationTotal: 5000 /* Muestras totales por grupo de clase aumentado */
            });
        break;
    }
    case methods.TF_COREAPI: {
        Object.assign(loadingSettings, {
            truncate: true,
            decimals: 4,
            normalization: true,
            fourier: true,
            deviationMatrix: true,
            dataAugmentation: true,
            dataAugmentationTotal: 50000, /* Muestras totales cada vez que un un archivo o lista de archivos es aumentado */
            });
        break;
    }
}

printHeader("Lectura de datos" + (loadingSettings.dataAugmentation ? " (Con Data Augmentation)" : ""));

const fileBasePath = './data/Full';
// const fileBasePath = './data/generated';
// const fileBasePath = './data/2022.2.22';
// const fileBasePath = './data/2022.3.19_Brazos';
// const fileBasePath = './data/2022.3.19_Cabeza';

let loadedData = null;

/* Carga de datos sin ningun tipo de procesamiento */
function loadData(moveType) {
    let fileNames = {};
    fileNames[MOVE_TYPE.DOWN] = '/ABAJO.json';
    fileNames[MOVE_TYPE.UP] = '/ARRIBA.json';
    fileNames[MOVE_TYPE.LEFT] = '/IZQUIERDA.json';
    fileNames[MOVE_TYPE.RIGHT] = '/DERECHA.json';

    let filePath = fileBasePath + fileNames[moveType];

    return loadJSON([{ file: filePath, moveType }], defaultSettings);
}

loadedData = loadData(MOVE_TYPE.DOWN);
loadedData.concat(loadData(MOVE_TYPE.UP));
loadedData.concat(loadData(MOVE_TYPE.LEFT));
loadedData.concat(loadData(MOVE_TYPE.RIGHT));

// loadedData = dataAugmentation(loadJSON([{ file: fileBasePath + '/ABAJO.json', moveType: MOVE_TYPE.DOWN }], defaultSettings), defaultSettings);
// loadedData.concat(dataAugmentation(loadJSON([{ file: fileBasePath + '/ARRIBA.json', moveType: MOVE_TYPE.UP }], defaultSettings), defaultSettings));
// loadedData.concat(dataAugmentation(loadJSON([{ file: fileBasePath + '/IZQUIERDA.json', moveType: MOVE_TYPE.LEFT }], defaultSettings), defaultSettings));
// loadedData.concat(dataAugmentation(loadJSON([{ file: fileBasePath + '/DERECHA.json', moveType: MOVE_TYPE.RIGHT }], defaultSettings), defaultSettings));
console.log("");
loadedData.calculateDataMaxes();
loadedData.summary();

// printHeader("Samples")
// console.log(loadedData.samples);

/* Una vez cargados, aplicar preprocesamientos, excepto data augmentation */
printHeader("Preprocesamiento de datos");
loadedData = dataPreProcessing(loadedData, loadingSettings);

console.log("");
console.log("Loading Settings", loadingSettings);

let start = loadedData.samples.length - 2005;
start = start < 0 ? 0 : start;

/* ************************************************************** */

printHeader("Testings");

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
    verbose: false
};

const neuroSettings = {
    MoveTypeEnum: MOVE_TYPE,
    verbose: true,
    epochs: 5,
    stepsPerEpoch: 20,
    validationSteps: 100,
    learningRate: 0.00000005
};

const NetLSTMSettings = {
    epochs: 10,
    batchSize: 2,
    verbose: false,
};

const trainingExerciseCount = 1;

let promises = [];
let testings = [];

for (let i = 0; i < trainingExerciseCount; i++) {
    const { samples, labels } = splitData(loadedData);

    /* Data de entrenamiento */
    const trainingLength = Math.floor(samples.length * 0.7);
    const trainingData = samples.slice(0, trainingLength);
    const trainingLabels = labels.slice(0, trainingLength);

    /* Data de pruebas */
    const testData = samples.slice(trainingLength);
    const testLabels = labels.slice(trainingLength);

    switch (ModelType) {
        case methods.TF_LAYERMODEL: {
            // LogisticRegression
            const regression = new ModelClass(trainingData, trainingLabels, regressionSettingsLayerModel);

            promises.push(
                regression.train(function (logs) {
                    let result = regression.test(testData, testLabels);
                    testings.push(result);
                    regression.summary();
                })
            );

            break;
        }
        case methods.TF_COREAPI: {
            // LogisticRegression
            const regression = new ModelClass(trainingData, trainingLabels, regressionSettingsCoreModel);
            regression.train();
            console.log("train");
            let result = regression.test(testData, testLabels);
            testings.push(result);
            console.log("Precision [" + (i + 1) + "]", result);

            regression.summary();
            break;
        }
        case methods.NAIVE_BAYES: {
            // NaiveBayes
            const naiveBayes = new ModelClass(trainingData, trainingLabels, naiveBayesSettings);
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
            printSubHeader("Crear Arbol mediante muestras");

            console.log("Training.length", trainingData.length);
            console.log("Training", trainingData[0]);
            console.log("Test.length", testData.length);
            console.log("Test", testData[0]);
            loadedData.summary();

            const decisionTree = new ModelClass(trainingData, trainingLabels, loadedData.FEATURE_NAMES, decisionTreeSettings);
            decisionTree.train();
            let result = decisionTree.test(testData, testLabels);
            testings.push(result);
            console.log("Precision [" + (i + 1) + "]", parseFloat(result.precision.toFixed(8)));
            
            // let localJson = decisionTree.toJSON();

            // printSubHeader("Crear Arbol mediante JSON");
            // const decisionTreeJSON = new ModelClass(localJson, loadedData.FEATURE_NAMES);
            // decisionTreeJSON.train();
            // let JSONResult = decisionTreeJSON.test(testData, testLabels);
            // console.log("Precision JSON [" + (i + 1) + "]", parseFloat(JSONResult.precision.toFixed(8)));

            decisionTree.summary();

            break;
        }
        case methods.NEURO: {
            printSubHeader("Red neuronal");
            // NeuronalNetwork
            const neuro = new ModelClass(trainingData, trainingLabels, neuroSettings);
            promises.push(
                neuro.train(function (logs) {
                    let result = neuro.test(testData, testLabels);
                    testings.push(result);
                    neuro.summary();
                })
            );
            break;
        }
        case methods.CONV_NET_LSTM: {
            printSubHeader("Red CONV_NET_LSTM");
            // ConvNetLSTM
            const net = new ModelClass(trainingData, trainingLabels, neuroSettings);
            // net.testModel(testData, testLabels);
            net.train();
            let result = net.test(testData, testLabels);
            testings.push(result);
            // console.log("Precision [" + (i + 1) + "]", result);
            break;
        }
        case methods.NET_LSTM: {
            printSubHeader("Red NET_LSTM");
            // NET_LSTM
            const net = new ModelClass(trainingData, trainingLabels, NetLSTMSettings);
            net.train();
            let result = net.test(testData, testLabels);
            testings.push(result);
            // console.log("Precision [" + (i + 1) + "]", result);
            break;
        }
        
        
    }
}

const rounding = 2;

function showResults(currentResults) {
    let precision = currentResults.map(o => o.precision);

    console.log("Promedio: " + _.round((_.sum(precision) / precision.length), rounding) +
        ", Min: " + _.round(_.min(precision), rounding) +
        ", Max: " + _.round(_.max(precision), rounding)
    );
}

let promiseModels = [methods.TF_LAYERMODEL, methods.NEURO, methods.TF_LAYERMODEL];
if (promiseModels.filter(o => o == ModelType)) {
    Promise.all(promises).then(results => { showResults(testings); });
} else {
    showResults(testings);
}
