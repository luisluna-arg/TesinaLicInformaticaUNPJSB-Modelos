const { MOVE_TYPE, loadJSON, splitData, dataPreProcessing } = require('./load-training-json');
const { DecisionTreeModel } = require('./decision-tree-model');

/* HELPER FUNCTIONS */
/* //////////////// */

const printHeader = function (header) {
    console.log("");
    console.log(header);
    console.log(new Array(header.length + 1).fill("=").join(''));
}

const printSubHeader = function (subheader) {
    console.log("");
    console.log("> " + subheader);
}

/* SETTINGS */
/* //////// */

const fileBasePath = './data';
let loadedData = null;

const loadingSettings = {
    preProcess: false,
    filter: false,
    shuffle: true,
    split: false,
    truncate: true,
    decimals: 4,
    normalization: true,
    fourier: true,
    deviationMatrix: true,
    dataAugmentation: false,
    selectFeatures: false,
    dataAugmentationTotal: 40000, /* Muestras totales cada vez que un un archivo o lista de archivos es aumentado */
    minTolerance: 0.0 /* entre 0 y 1, 0 para que traiga todo */
};

const modelSettings = {
    MoveTypeEnum: MOVE_TYPE,
    verbose: false
};

/* CARGA DE ARCHIVOS */
/* ///////////////// */

/* Carga de datos sin ningun tipo de procesamiento */
function loadData(moveType) {
    let fileNames = {};
    fileNames[MOVE_TYPE.DOWN] = '/ABAJO.json';
    fileNames[MOVE_TYPE.UP] = '/ARRIBA.json';
    fileNames[MOVE_TYPE.LEFT] = '/IZQUIERDA.json';
    fileNames[MOVE_TYPE.RIGHT] = '/DERECHA.json';

    let filePath = fileBasePath + fileNames[moveType];

    return loadJSON([{ file: filePath, moveType }], loadingSettings);
}

loadedData = loadData(MOVE_TYPE.DOWN);
loadedData.concat(loadData(MOVE_TYPE.UP));
loadedData.concat(loadData(MOVE_TYPE.LEFT));
loadedData.concat(loadData(MOVE_TYPE.RIGHT));

/* PREPROCESADO DE DATOS */
/* ///////////////////// */

/* Una vez cargados, aplicar preprocesamientos, excepto data augmentation */
printHeader("Preprocesamiento de datos");
loadingSettings.preProcess = true;
loadedData = dataPreProcessing(loadedData, loadingSettings);

/* ENTRENAMIENTO */
/* ///////////// */
splitData(loadedData);

const samples = loadedData.getSamples();
const labels = loadedData.getLabels();

// decisionTree: {
//     MoveTypeEnum: MOVE_TYPE,
//     verbose: false
// },

/* Data de entrenamiento */
const trainingLength = Math.floor(samples.length * 0.7);
const trainingData = samples.slice(0, trainingLength);
const trainingLabels = labels.slice(0, trainingLength);

/* Data de pruebas */
const testData = samples.slice(trainingLength);
const testLabels = labels.slice(trainingLength);

// loadedData.summary();

printHeader("Preprocesamiento y entrenamiento");
const decisionTree = new DecisionTreeModel(trainingData, trainingLabels, loadedData.getFeatureNames(), modelSettings);

/* TEST */
/* //// */
let correct = 0;
for (let i = 0; i < testData.length; i++) {
    const sample = testData[i];
    const prediction = decisionTree.predict(sample);
    const equals = prediction == testLabels[i];
    correct += equals ? 1 : 0;
    // console.log(`Prediction: ${prediction}`, `Expected: ${testLabels[i]}`, equals);
}

printHeader("Resultados");
console.log(`Precision: ${correct / testData.length}`, `Correct: ${correct} | Total: ${testData.length}`);
