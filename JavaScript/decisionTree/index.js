const MiscUtils = require('./misc-utils');
const { DecisionTreeModel } = require('./decision-tree-model');
const _ = require('lodash');

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
for (let i = 0; i < testData.samples.length; i++) {
    const sample = testData.samples[i];
    const prediction = decisionTree.predictPreProcessed(sample);
    const equals = prediction == testData.labels[i];
    correct += equals ? 1 : 0;
}

MiscUtils.printHeader("Resultados Test");
console.log(`Correct: ${correct} | Total: ${testData.samples.length}`);
decisionTree.summary();

// console.log(decisionTree.getFeatureNames());
// console.log(decisionTree.getTrainingData().samples[0]);
// console.log(decisionTree.getTestData().samples[0]);

/* TEST JSON - El modelo, reconstruido, se prueba con datos no preprocesados */

const decisionTreeJSON = new DecisionTreeModel(trainingJSON);
correct = 0;
for (let i = 0; i < testDataSet.length; i++) {
    const originalSample = testDataSet[i];
    const sample = originalSample.slice(0, originalSample.length - 1);
    const prediction = decisionTreeJSON.predict(sample);
    // console.log("sample", sample, "prediction", prediction, "label", originalSample[originalSample.length - 1]);
    const equals = prediction == originalSample[originalSample.length - 1];
    correct += equals ? 1 : 0;
}

MiscUtils.printHeader("Resultados Test JSON");
console.log(`Correct: ${correct} | Total: ${testDataSet.length} | Precision: ${correct / testDataSet.length * 100}%`);
