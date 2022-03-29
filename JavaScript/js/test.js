const { MOVE_TYPE, loadJSON, splitData } = require('./load-training-json');
const _ = require('lodash');
const DecisionTree = require('decision-tree');

/* ************************************************************** */
/* ************************************************************** */
/* ************************************************************** */

let loadingSettings = {
    shuffle: true,
    split: false,
    decimals: 4,
    normalization: true,
    fourier: true,
    deviationMatrix: true,
    truncate: false,
    dataAugmentation: false,
    dataAugmentationTotal: 4000, /* Muestras totales cada vez que un un archivo o lista de archivos es aumentado */
    minTolerance: 0.0 /* entre 0 y 1, 0 para que traiga todo */
};

let message = "Lectura de datos" + (loadingSettings.dataAugmentation ? " (Con Data Augmentation)" : "")
// console.log(message);
// console.log("");

const fileBasePath = '../data/Full';
// const fileBasePath = './data/generated';
// const fileBasePath = '../data/2022.1.14';
// const fileBasePath = './data/2022.2.22';
//const fileBasePath = '../data/2022.3.19_Brazos';
// const fileBasePath = './data/2022.3.19_Cabeza';

let loadedData = loadJSON([{ file: fileBasePath + '/ABAJO.json', moveType: MOVE_TYPE.DOWN }], loadingSettings);
loadedData.concat(loadJSON([{ file: fileBasePath + '/ARRIBA.json', moveType: MOVE_TYPE.UP }], loadingSettings));
loadedData.concat(loadJSON([{ file: fileBasePath + '/IZQUIERDA.json', moveType: MOVE_TYPE.LEFT }], loadingSettings));
loadedData.concat(loadJSON([{ file: fileBasePath + '/DERECHA.json', moveType: MOVE_TYPE.RIGHT }], loadingSettings));

/* ************************************************************** */

console.log("Testings");
console.log("========");
console.log("");

const CLASS_NAME = 'class';

console.log("loadedData.samples[0]", loadedData.samples[0]);

let fullSamples = loadedData.samples.map(sample => {
    let obj = {};
    let i;
    for (i = 0; i < sample.length-1; i++) {
        obj[loadedData.FEATURE_NAMES[i]] = sample[i];
    }
    obj['class'] = sample[i];

    return obj;
});

fullSamples = _.shuffle(fullSamples);

const trainingLength = Math.floor(fullSamples.length * 0.75);
let trainSamples = fullSamples.slice(0, trainingLength);
let testSamples = fullSamples.slice(trainingLength);

// console.log("trainSamples.slice(0, 5)", trainSamples.slice(0, 5));
// console.log("trainSamples.length", trainSamples.length, "trainSamples[0]", trainSamples[0]);
// console.log("testSamples.slice(0, 5)", testSamples.slice(0, 5));
// console.log("testSamples.length", testSamples.length, "testSamples[0]", testSamples[0]);

let decisionTree = new DecisionTree(CLASS_NAME, loadedData.FEATURE_NAMES);
decisionTree.train(trainSamples);

let trainAccuracy = decisionTree.evaluate(trainSamples);
let testAccuracy = decisionTree.evaluate(testSamples);

let labels = testSamples.map(o => o[CLASS_NAME]);
let predictions = testSamples.map(o => decisionTree.predict(o));

loadedData.summary()

console.log("trainAccuracy", trainAccuracy);
console.log("testAccuracy", testAccuracy);
console.log("predictions", predictions);
console.log("labels", labels);