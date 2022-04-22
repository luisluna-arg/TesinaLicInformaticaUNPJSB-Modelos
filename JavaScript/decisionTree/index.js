const MiscUtils = require('./misc-utils');
const { DecisionTreeModel, confusionMatrix } = require('./decision-tree-model');
const _ = require('lodash');

/* ////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////// */

MiscUtils.printHeader("Carga, preprocesamiento de datos y entrenamiento");
const decisionTree = new DecisionTreeModel('./data');

/* Test data contiene datos preprocesados */
const testData = decisionTree.getTestData();
const trainingJSON = decisionTree.toJSON();
const dataSet = decisionTree.getDataSet();

const testDataSet = _.shuffle(dataSet).slice((dataSet.length * 0.7));

/* TEST - El modelo se prueba con datos preprocesados */
/* ////////////////////////////////////////////////// */
let correct = 0;
let pedictionLabels = [];
let realLabels = [];
for (let i = 0; i < testData.samples.length; i++) {
    const sample = testData.samples[i];
    const prediction = decisionTree.predictPreProcessed(sample);
    const realLabel = testData.labels[i];
    pedictionLabels.push(prediction);
    realLabels.push(realLabel);
    const equals = prediction == realLabel;
    correct += equals ? 1 : 0;
}

MiscUtils.printHeader("Resultados Test");
console.log(`Correct: ${correct} | Total: ${testData.samples.length}`);
decisionTree.summary();
MiscUtils.printHeader("Matriz de confusion");
console.log(confusionMatrix(pedictionLabels, realLabels));



/* TEST JSON - El modelo, reconstruido, se prueba con datos no preprocesados */

const decisionTreeJSON = new DecisionTreeModel(trainingJSON);
correct = 0;

let pedictionLabelsJSON = [];
let realLabelsJSON = [];
for (let i = 0; i < testDataSet.length; i++) {
    const originalSample = testDataSet[i];
    const sample = originalSample.slice(0, originalSample.length - 1);
    const prediction = decisionTreeJSON.predict(sample, i == testDataSet.length - 1);
    const realLabel = originalSample[originalSample.length - 1];
    
    pedictionLabelsJSON.push(prediction);
    realLabelsJSON.push(realLabel);

    const equals = prediction == realLabel;
    correct += equals ? 1 : 0;
}

MiscUtils.printHeader("Resultados Test JSON");
console.log(`Correct: ${correct} | Total: ${testDataSet.length} | Precision: ${correct / testDataSet.length * 100}%`);
MiscUtils.printHeader("Matriz de confusion JSON");
console.log(confusionMatrix(pedictionLabelsJSON, realLabelsJSON));