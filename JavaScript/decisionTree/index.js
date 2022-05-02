const MiscUtils = require('./misc-utils');
const { DecisionTreeModel, confusionMatrix } = require('./decision-tree-model');
const _ = require('lodash');
const { preProcess } = require('./data-preprocessing');

/* ////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////// */

/* Descomentar esta seccion para reentrenar el modelo */

// MiscUtils.printHeader("Carga, preprocesamiento de datos y entrenamiento");
// const decisionTree = new DecisionTreeModel('./data');

// /* Test data contiene datos preprocesados */
// const testData = decisionTree.getTestData();
// const dataSet = decisionTree.getDataSet();

// /* TEST - El modelo se prueba con datos preprocesados */
// /* ////////////////////////////////////////////////// */
// let startTime = new Date();
// console.log(`Inicio: ${startTime.toLocaleString()}`);
// let correct = 0;
// let pedictionLabels = [];
// let realLabels = [];
// let testDataCount = 30;

// for (let i = 0; i < testDataCount; i++) {
//     const sample = testData.samples[i];
//     const prediction = decisionTree.predictPreProcessed(sample);
//     const realLabel = testData.labels[i];
//     pedictionLabels.push(prediction);
//     realLabels.push(realLabel);
//     const equals = prediction == realLabel;
//     correct += equals ? 1 : 0;
// }

// MiscUtils.printHeader("Resultados Test");
// console.log(`Correct: ${correct} | Total: ${testData.samples.length}`);
// decisionTree.summary();
// MiscUtils.printHeader("Matriz de confusion");
// console.log(confusionMatrix(pedictionLabels, realLabels));
// decisionTree.exportDataSet();
// decisionTree.exportSettings();

// console.log(`Inicio: ${startTime.toLocaleString()}`);
// console.log(`Fin: ${new Date().toLocaleString()}`);



/* TEST JSON - El modelo, reconstruido, se prueba con datos no preprocesados */
/* ///////////////////////////////////////////////////////////////////////// */
startTime = new Date();
console.log(`Inicio: ${startTime.toLocaleString()}`);

const trainingJSONLoaded = MiscUtils.readJSON('./data/decisiontree-settings.json');
const decisionTreeJSON = new DecisionTreeModel(trainingJSONLoaded);
correct = 0;

let fileDataSet = decisionTreeJSON.getDataSet();
testDataSet = _.shuffle(fileDataSet).slice(Math.floor(fileDataSet.length * 0.7));
testDataCount = 30;// testDataSet.length;

let pedictionLabelsJSON = [];
let realLabelsJSON = [];

let sampleStartTime;
for (let i = 0; i < testDataCount; i++) {
    sampleStartTime = new Date().getTime()
    const originalSample = testDataSet[i];
    // console.log("originalSample", originalSample);
    const sample = originalSample.slice(0, originalSample.length - 1);
    const prediction = decisionTreeJSON.predict(sample);
    const realLabel = originalSample[originalSample.length - 1];

    pedictionLabelsJSON.push(prediction);
    realLabelsJSON.push(realLabel);

    const equals = prediction == realLabel;
    correct += equals ? 1 : 0;
    // console.log(`Duracion [${i}]: ${new Date(new Date().getTime() - sampleStartTime).getMilliseconds()} ms, 
    //     Label: ${realLabel}, Prediction: ${prediction}`);
}

MiscUtils.printHeader("Resultados Test JSON");
console.log(`Correct: ${correct} | Total: ${testDataCount} | Precision: ${correct / testDataCount * 100}%`);
MiscUtils.printHeader("Matriz de confusion JSON");
console.log(confusionMatrix(pedictionLabelsJSON, realLabelsJSON));

console.log(`Inicio: ${startTime.toLocaleString()}`);
console.log(`Fin: ${new Date().toLocaleString()}`);

