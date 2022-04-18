const MiscUtils = require('./misc-utils');
const { NaiveBayesModel } = require('./naive-bayes-model');
const _ = require('lodash');

/* ////////////////////////////////////////////////// */

MiscUtils.printHeader("Carga, preprocesamiento de datos y entrenamiento");
const decisionTree = new NaiveBayesModel('./data');

/* Test data contiene datos preprocesados */
const trainingData = decisionTree.getTrainingData();
const testData = decisionTree.getTestData();
const dataSet = decisionTree.getDataSet();

const trainingJSON = decisionTree.toJSON();

const testDataSet = _.shuffle(dataSet).slice((dataSet.length * 0.7));

/* TEST - El modelo se prueba con datos preprocesados */
/* ////////////////////////////////////////////////// */

decisionTree.whenTrained(() => {
    let correct = 0;
    let collection = [];
    let results = [];
    for (let i = 0; i < testData.samples.length; i++) {
        const sample = testData.samples[i];
        // console.log("sample", sample);
        const label = testData.labels[i];
        const prediction = decisionTree.predictPreProcessed(sample);
        results.push([label, prediction]);
    }

    let counter = { correcto: 0, total: testData.samples.length }
    for (let i = 0; i < results.length; i++) {
        results[i][1].then((resolution) => {
            const expected = results[i][0];
            const predicted = parseInt(resolution);
            // console.log("expected", expected, "predicted", predicted);
            // console.log("results[" + i + "][0]", results[i][0], "resolutionValue", resolutionValue);
            counter.correcto += (expected == predicted) ? 1 : 0;
        });
    }

    Promise.all(results.map(o => o[1])).then(() => {
        counter.precision = MiscUtils.trunc(counter.correcto / counter.total * 100, 2);
        MiscUtils.printHeader("Resultados Test");
        console.log(`Correct: ${counter.correcto} | Total: ${counter.total} | Precisión: ${counter.precision}`);
        decisionTree.summary();
    });

    // // console.log(decisionTree.getFeatureNames());
    // // console.log(decisionTree.getTrainingData().samples[0]);
    // // console.log(decisionTree.getTestData().samples[0]);

    /* TEST JSON - El modelo, reconstruido, se prueba con datos no preprocesados */

    let jsonResult = [];
    const decisionTreeJSON = new NaiveBayesModel(trainingJSON);
    correct = 0;
    for (let i = 0; i < testDataSet.length; i++) {
        const originalSample = testDataSet[i];
        const sample = originalSample.slice(0, originalSample.length - 1);
        // console.log("sample", sample);
        const expectedLabel = originalSample.slice(originalSample.length - 1)[0];
        const prediction = decisionTreeJSON.predict(sample);
        jsonResult.push([expectedLabel, prediction]);
    }

    let counterJson = { correcto: 0, total: testDataSet.length }
    for (let i = 0; i < jsonResult.length; i++) {
        jsonResult[i][1].then((resolution) => {
            const expected = jsonResult[i][0];
            const predicted = parseInt(resolution);
            // console.log("expected", expected, "predicted", predicted);
            // console.log("jsonResult[" + i + "][0]", jsonResult[i][0], "resolutionValue", resolutionValue);
            counterJson.correcto += (expected == predicted) ? 1 : 0;
        });
    }

    Promise.all(jsonResult.map(o => o[1])).then(() => {
        counterJson.precision = MiscUtils.trunc(counterJson.correcto / counterJson.total * 100, 2);
        MiscUtils.printHeader("Resultados Test Json");
        console.log(`Correct: ${counterJson.correcto} | Total: ${counterJson.total} | Precisión: ${counterJson.precision}`);
        decisionTree.summary();
    });
});




