const MiscUtils = require('./misc-utils');
const { LinearRegressionModel, confusionMatrix } = require('./linear-regression-model');
const _ = require('lodash');
const { sampleNormalization } = require('./data-preprocessing');

/* ////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////// */

const logFilePath = './Logs.txt';

const blankPad = (num, places) => String(num).padStart(places, ' ');

const printLogs = (logLine, logType) => {
    switch (logType) {
        case 1: 
        {
            MiscUtils.printHeader(logLine); 
            MiscUtils.writeTextFileHeader(logFilePath, logLine);
            break;
        }
        case 2: {
            MiscUtils.printSubHeader(logLine); 
            MiscUtils.writeTextFileSubHeader(logFilePath, logLine);
            break;
        }
        default: {
            console.log(logLine); 
            MiscUtils.writeTextFile(logFilePath, logLine);
            break;
        }
    }
};

function printConfusionMatrix(predictionLabels, realLabels) {
    let confMat = { matrix: null };
    printLogs("Matriz de confusión", 1);
    confMat = confusionMatrix(predictionLabels, realLabels);

    let zeroPadCount = _.max(_.flatten(confMat.matrix).filter(o => !isNaN(o))).toString().length;

    for (let i = 0; i < confMat.matrix.length; i++) {
        printLogs(confMat.matrix[i].
            map((m, j) => {
                if (m == '|') return m;
                if (i == 0 && j == 0) return new Array(zeroPadCount + 1).join(' ');
                return blankPad(m, zeroPadCount)
            }).join(' '));
    }
    printLogs(``);
    printLogs(`Listas real/predicción: ${confMat.realLabelCount}/${confMat.predictionCount}`);
    printLogs(`Precisión: ${MiscUtils.trunc(confMat.precision * 100, 2)}%`);
    printLogs(``);
}

function trainModel(loadingSettings) {
    let startTime = new Date();

    printLogs(``);
    printLogs(``);
    printLogs(`*************************************************`);

    printLogs(`Inicio de entrenamiento: ${startTime.toLocaleString()}`, 1);

    printLogs("Carga, preprocesamiento de datos y entrenamiento", 1);
    const linearRegressionModel = new LinearRegressionModel('./data', loadingSettings);

    /* Test data contiene datos preprocesados */
    const testData = linearRegressionModel.getTestData();
    let testDataSamples = testData.samples.filter((s, i) => testData.labels[i] > 0);
    let testDataLabels = testData.labels.filter((s, i) => testData.labels[i] > 0);

    /* TEST - El modelo se prueba con datos preprocesados */
    /* ////////////////////////////////////////////////// */
    let correct = 0;
    let predictionLabels = [];
    let realLabels = [];
    let testDataCount = testDataSamples.length; // 30;

    for (let i = 0; i < testDataCount; i++) {
        const sample = testDataSamples[i];
        const prediction = linearRegressionModel.predictPreProcessed(sample);
        const realLabel = testDataLabels[i];
        predictionLabels.push(prediction);
        realLabels.push(realLabel);
        correct += prediction == realLabel ? 1 : 0;
    }

    printLogs("Resultados Test", 1);
    printLogs(`Correct: ${correct} | Total: ${testDataSamples.length}`, 0);

    linearRegressionModel.summary(printLogs);

    printConfusionMatrix(predictionLabels, realLabels);

    linearRegressionModel.exportDataSet();
    linearRegressionModel.exportPreProcessDataSet();
    linearRegressionModel.exportSettings();

    printLogs(`Inicio: ${startTime.toLocaleString()} | Fin: ${new Date().toLocaleString()}`, 0);

    return correct / testDataCount * 100;
}

function predictWithJson() {
    /* TEST JSON - El modelo, reconstruido, se prueba con datos no preprocesados */
    /* ///////////////////////////////////////////////////////////////////////// */
    MiscUtils.printHeader("Test JSON");

    startTime = new Date();

    const trainingJSONLoaded = MiscUtils.readJSON('./data/decisiontree-settings.json');
    const decisionTreeJSON = new LinearRegressionModel(trainingJSONLoaded);
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
        const doRemap = featValues.filter((feat, i) => feat > featureMoments[i].q3).length > 0;
        return [...featValues, doRemap ? 0 : sample[featureNames.length]];
    }

    let fileDataSet = decisionTreeJSON.getDataSet();
    testDataSet = _.shuffle(fileDataSet).slice(Math.floor(fileDataSet.length * 0.7))
        .map(s => {
            let norm = sampleNormalization(s, normalizationFeat);
            let remappedNorm = remapLower([...norm, s[s.length - 1]], featureNames, filterStats);
            return [...s.slice(0, s.length), remappedNorm[remappedNorm.length - 1]]
        });


    testDataCount = 500;// testDataSet.length;
    let pedictionLabelsJSON = [];
    let realLabelsJSON = [];

    let sampleStartTime;
    for (let i = 0; i < testDataCount; i++) {
        sampleStartTime = new Date().getTime()
        const originalSample = testDataSet[i];
        const sample = originalSample.slice(0, originalSample.length - 1);
        const prediction = decisionTreeJSON.predict(sample);
        const realLabel = originalSample[originalSample.length - 1];

        pedictionLabelsJSON.push(parseInt(prediction));
        realLabelsJSON.push(realLabel);

        correct += prediction == realLabel ? 1 : 0;
        // console.log(`Duracion [${i}]: ${new Date(new Date().getTime() - sampleStartTime).getMilliseconds()} ms, 
        //     Label: ${realLabel}, Prediction: ${prediction}`);
    }

    MiscUtils.printSubHeader("Resultados");
    console.log(`Datos entrenamiento | Test: ${MiscUtils.trunc(trainingJSONLoaded.testAccuracy * 100, 2)}% | Precisión: ${MiscUtils.trunc(trainingJSONLoaded.trainAccuracy * 100, 2)}%`);
    console.log(`Correctos/Total: ${correct}/${testDataCount} | Precisión test JSON: ${MiscUtils.trunc(correct / testDataCount * 100, 2)}%`);

    printConfusionMatrix(pedictionLabelsJSON, realLabelsJSON);

    console.log(`Inicio: ${startTime.toLocaleString()} | Fin: ${new Date().toLocaleString()}`);
}






/* Comentar esta seccion para no entrenar el modelo */
let samplesPerLabel = 2000;
const maxPrecision = 85;
let precision = 0;
while(precision <= maxPrecision) {
    precision = trainModel({ dataAugmentationTotal: samplesPerLabel, dataAugmentation: true });
    samplesPerLabel += 5000;
}

// predictWithJson()




