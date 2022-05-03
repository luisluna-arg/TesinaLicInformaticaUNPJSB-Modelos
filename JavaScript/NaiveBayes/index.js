const MiscUtils = require('./misc-utils');
const { NaiveBayesModel, confusionMatrix } = require('./naive-bayes-model');
const _ = require('lodash');

/* ////////////////////////////////////////////////// */

MiscUtils.printHeader("Carga, preprocesamiento de datos y entrenamiento");

let startTime = new Date();
console.log(`Inicio: ${startTime.toLocaleString()}`);

const naiveBayes = new NaiveBayesModel('./data');

/* Test data contiene datos preprocesados */
const testData = naiveBayes.getTestData();
const dataSet = naiveBayes.getDataSet();

// const testDataSet = _.shuffle(dataSet).slice((dataSet.length * 0.7));

/* TEST - El modelo se prueba con datos preprocesados */
/* ////////////////////////////////////////////////// */

const jsonTest = function () {
    /* TEST JSON - El modelo, reconstruido, se prueba con datos no preprocesados */
    /* ///////////////////////////////////////////////////////////////////////// */
    startTime = new Date();
    console.log(`Inicio: ${startTime.toLocaleString()}`);

    const trainingJSONLoaded = MiscUtils.readJSON('./data/naivebayes-settings.json');
    const naiveBayesJSON = new NaiveBayesModel(trainingJSONLoaded);
    correct = 0;

    let fileDataSet = naiveBayesJSON.getDataSet();
    let testDataSet = _.shuffle(fileDataSet).slice(Math.floor(fileDataSet.length * 0.7));
    testDataCount = 30;// testDataSet.length;

    let predictionLabelsJSON = [];
    let realLabelsJSON = [];

    let jsonResult = [];
    let sampleStartTime;
    for (let i = 0; i < testDataCount; i++) {
        sampleStartTime = new Date().getTime()
        const originalSample = testDataSet[i];
        const sample = originalSample.slice(0, originalSample.length - 1);
        const expectedLabel = originalSample.slice(originalSample.length - 1)[0];
        const prediction = naiveBayesJSON.predict(sample);
        jsonResult.push([expectedLabel, prediction]);
    }

    let counterJson = { correcto: 0, total: testDataSet.length }
    for (let i = 0; i < jsonResult.length; i++) {
        jsonResult[i][1].then((resolution) => {
            const expected = jsonResult[i][0];
            const predicted = parseInt(resolution);

            predictionLabelsJSON.push(predicted);
            realLabelsJSON.push(expected);

            const equals = expected == predicted;
            counterJson.correcto += equals ? 1 : 0;
        });
    }

    Promise.all(jsonResult.map(o => o[1])).then(() => {
        MiscUtils.printHeader("Resultados Test JSON");
        console.log(`Correct: ${correct} | Total: ${testDataCount} | Precision: ${counterJson.correcto / counterJson.total * 100}%`);
        MiscUtils.printHeader("Matriz de confusion JSON");
        console.log(confusionMatrix(predictionLabelsJSON, realLabelsJSON));

        console.log(`Inicio: ${startTime.toLocaleString()}`);
        console.log(`Fin: ${new Date().toLocaleString()}`);
    });
}

naiveBayes.whenTrained(() => {
    let correct = 0;
    let results = [];
    for (let i = 0; i < testData.samples.length; i++) {
        const sample = testData.samples[i];
        // console.log("sample", sample);
        const label = testData.labels[i];
        const prediction = naiveBayes.predictPreProcessed(sample);
        results.push([label, prediction]);
    }

    let counter = { correcto: 0, total: testData.samples.length }
    let predictionLabels = [];
    let realLabels = [];
    for (let i = 0; i < results.length; i++) {
        results[i][1].then((resolution) => {
            const realLabel = results[i][0];
            const predicted = parseInt(resolution);

            realLabels.push(realLabel);
            predictionLabels.push(predicted);

            counter.correcto += (realLabel == predicted) ? 1 : 0;
        });
    }

    Promise.all(results.map(o => o[1])).then(() => {


        counter.precision = MiscUtils.trunc(counter.correcto / counter.total * 100, 2);
        MiscUtils.printHeader("Resultados Test");
        console.log(`Correct: ${counter.correcto} | Total: ${counter.total} | Precisión: ${counter.precision}`);
        naiveBayes.summary();
        MiscUtils.printHeader("Matriz de confusion");
        console.log(confusionMatrix(predictionLabels, realLabels));
        naiveBayes.exportDataSet();
        naiveBayes.exportSettings();

        console.log(`Inicio: ${startTime.toLocaleString()}`);
        console.log(`Fin: ${new Date().toLocaleString()}`);


        jsonTest();
    });




});




