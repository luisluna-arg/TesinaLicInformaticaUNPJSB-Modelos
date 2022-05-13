const MiscUtils = require('./misc-utils');
const { NaiveBayesModel, confusionMatrix } = require('./naive-bayes-model');
const _ = require('lodash');
const { sampleNormalization } = require('./data-preprocessing');

/* ////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////// */

const blankPad = (num, places) => String(num).padStart(places, ' ');

function printConfusionMatrix(predictionLabels, realLabels) {
    let confMat = { matrix: null };
    MiscUtils.printHeader("Matriz de confusión");
    confMat = confusionMatrix(predictionLabels, realLabels);

    let zeroPadCount = _.max(_.flatten(confMat.matrix).filter(o => !isNaN(o))).toString().length;

    for (let i = 0; i < confMat.matrix.length; i++) {
        console.log(confMat.matrix[i].
            map((m, j) => {
                if (m == '|') return m;
                if (i == 0 && j == 0) return new Array(zeroPadCount + 1).join(' ');
                return blankPad(m, zeroPadCount)
            }).join(' '));
    }
    console.log(``);
    console.log(`Listas real/predicción: ${confMat.realLabelCount}/${confMat.predictionCount}`);
    console.log(`Precisión: ${MiscUtils.trunc(confMat.precision * 100, 2)}%`);
    console.log(``);
}

// MiscUtils.printHeader("Carga, preprocesamiento de datos y entrenamiento");

// let startTime = new Date();
// console.log(`Inicio: ${startTime.toLocaleString()}`);

// const naiveBayes = new NaiveBayesModel('./data');

// /* Test data contiene datos preprocesados */
// const testData = naiveBayes.getTestData();
// const dataSet = naiveBayes.getDataSet();

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

    let filterStats = trainingJSONLoaded.preProcess.stats.filter;
    let normalizationFeat = trainingJSONLoaded.preProcess.stats.normalizationFeat;
    let featureNames = [
        'delta',
        'theta',
        'lowAlpha',
        'highAlpha',
        'lowBeta',
        'highBeta',
        'lowGamma',
        'highGamma',
    ];

    function remapLower(sample, featureNames, featureMoments) {
        const featValues = sample.slice(0, featureNames.length);
        const doRemap = featValues.filter((feat, i) => feat > featureMoments[i].median).length > 0;
        return [...featValues, doRemap ? 0 : sample[featureNames.length]];
    }

    let fileDataSet = naiveBayesJSON.getDataSet();
    testDataSet = _.shuffle(fileDataSet).slice(Math.floor(fileDataSet.length * 0.7))
        .map(s => {
            let norm = sampleNormalization(s, normalizationFeat);
            let remappedNorm = remapLower([...norm, s[s.length - 1]], featureNames, filterStats);
            return [...s.slice(0, s.length), remappedNorm[remappedNorm.length - 1]]
        });

    testDataCount = 500;// testDataSet.length;

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

        printConfusionMatrix(predictionLabelsJSON, realLabelsJSON);

        console.log(`Inicio: ${startTime.toLocaleString()} | Fin: ${new Date().toLocaleString()}`);
    });
}

// naiveBayes.whenTrained(() => {
//     let correct = 0;
//     let results = [];
//     for (let i = 0; i < testData.samples.length; i++) {
//         const sample = testData.samples[i];
//         // console.log("sample", sample);
//         const label = testData.labels[i];
//         const prediction = naiveBayes.predictPreProcessed(sample);
//         results.push([label, prediction]);
//     }

//     let counter = { correcto: 0, total: testData.samples.length }
//     let predictionLabels = [];
//     let realLabels = [];
//     for (let i = 0; i < results.length; i++) {
//         results[i][1].then((resolution) => {
//             const realLabel = results[i][0];
//             const predicted = parseInt(resolution);

//             realLabels.push(realLabel);
//             predictionLabels.push(predicted);

//             counter.correcto += (realLabel == predicted) ? 1 : 0;
//         });
//     }

//     Promise.all(results.map(o => o[1])).then(() => {
//         counter.precision = counter.correcto / counter.total;
//         naiveBayes.setTestAccuracy(counter.precision);
//         console.log(`Precision ${counter.precision}%`);
//         MiscUtils.printHeader("Resultados Test");
//         console.log(`Correct: ${counter.correcto} | Total: ${counter.total} | Precisión: ${counter.precision}`);
//         naiveBayes.summary();

//         printConfusionMatrix(predictionLabels, realLabels);

//         naiveBayes.exportDataSet();
//         naiveBayes.exportPreProcessDataSet();
//         naiveBayes.exportSettings();

//         console.log(`Inicio: ${startTime.toLocaleString()} | Fin: ${new Date().toLocaleString()}`);
//         console.log("");

//         jsonTest();
//     });

// });


jsonTest();


