const MiscUtils = require('./misc-utils');
const { MOVE_TYPE, loadJSON, splitData, dataPreProcessing } = require('./load-training-json');
const { refactorSample } = require('./data-preprocessing');
const Bayes = require('bayes');

const CLASS_NAME = "moveType";

const MoveTypeEnum = {
    NONE: 0,
    DOWN: 1,
    UP: 2,
    LEFT: 3,
    RIGHT: 4
}

const MoveTypeTokens = [
    'NONE',
    'DOWN',
    'UP',
    'LEFT',
    'RIGHT'
];

/* SETTINGS */
/* //////// */

const DataLoadingSettings = {
    preProcess: false,
    filter: false,
    shuffle: true,
    split: false,
    truncate: true,
    decimals: 4,
    normalization: true,
    fourier: true,
    deviationMatrix: true,
    selectFeatures: false,
    dataAugmentation: false,
    dataAugmentationTotal: 10000, /* Muestras totales cada vez que un un archivo o lista de archivos es aumentado */
    minTolerance: 0.0 /* entre 0 y 1, 0 para que traiga todo */
};

const ModelTrainingSettings = {
    MoveTypeEnum: MOVE_TYPE,
    verbose: false
};

function loadData(fileBasePath) {
    fileBasePath = !MiscUtils.isNullOrUndef(fileBasePath) ? fileBasePath : './data';
    let loadedData = null;

    /* CARGA DE ARCHIVOS */
    /* ///////////////// */

    /* Carga de datos sin ningun tipo de procesamiento */
    function loadFiles(moveType) {
        let fileNames = {};
        fileNames[MOVE_TYPE.DOWN] = '/ABAJO.json';
        fileNames[MOVE_TYPE.UP] = '/ARRIBA.json';
        fileNames[MOVE_TYPE.LEFT] = '/IZQUIERDA.json';
        fileNames[MOVE_TYPE.RIGHT] = '/DERECHA.json';

        let filePath = fileBasePath + fileNames[moveType];

        return loadJSON([{ file: filePath, moveType }], DataLoadingSettings);
    }

    loadedData = loadFiles(MOVE_TYPE.DOWN);
    loadedData.concat(loadFiles(MOVE_TYPE.UP));
    loadedData.concat(loadFiles(MOVE_TYPE.LEFT));
    loadedData.concat(loadFiles(MOVE_TYPE.RIGHT));

    const dataSet = loadedData.getSamples();

    /* PREPROCESADO DE DATOS */
    /* ///////////////////// */

    /* Una vez cargados, aplicar preprocesamientos, excepto data augmentation */
    MiscUtils.printHeader("Preprocesamiento de datos");
    DataLoadingSettings.preProcess = true;
    const preProcessResult = dataPreProcessing(loadedData, DataLoadingSettings);
    loadedData = preProcessResult.data;

    /* ENTRENAMIENTO */
    /* ///////////// */
    splitData(loadedData);

    const samples = loadedData.getSamples();
    const labels = loadedData.getLabels();

    /* Data de entrenamiento */
    const trainingLength = Math.floor(samples.length * 0.7);
    const trainingData = samples.slice(0, trainingLength);
    const trainingLabels = labels.slice(0, trainingLength);

    /* Data de pruebas */
    const testData = samples.slice(trainingLength);
    const testLabels = labels.slice(trainingLength);

    // loadedData.summary();

    return {
        featureNames: loadedData.getFeatureNames(),
        dataSet: dataSet,
        training: {
            samples: trainingData,
            labels: trainingLabels
        },
        test: {
            samples: testData,
            labels: testLabels
        },
        preProcess: {
            normalizationFeatStats: preProcessResult.normalizationFeatStats,
            deviationMatrixStats: preProcessResult.deviationMatrixStats,
            fourierStats: preProcessResult.fourierStats,
            trainingSettings: preProcessResult.trainingSettings
        }
    };
}

class Sample {
    constructor(sampleValues, featureNames, label) {
        for (let i = 0; i < sampleValues.length; i++) {
            this[featureNames[i]] = sampleValues[i].toString();
        }

        if (label) {
            this[CLASS_NAME] = label;
        }
    }
}

class NaiveBayesModel {

    #JSONTrained = false;
    #trainAccuracy = 0;
    #testAccuracy = 0;
    #classifier = null;
    #featureNames = null;
    #preProcess = null;
    #testData = {};
    #dataSet = {};
    #trainingData = {};
    #options = null;
    #trainingFinished = null;
    #trainingPromise = null;

    constructor(arg) {
        if (typeof arg == 'string') {
            /* Recibe el path de archivos de entrenamiento de entrenamiento */
            let { training, test, featureNames, preProcess, dataSet } = loadData(arg);
            this.#testData = test;
            this.#preProcess = preProcess;
            this.#dataSet = dataSet;
            this.#createModel(training.samples, training.labels, featureNames, ModelTrainingSettings);
        }
        else {
            /* Recibe el JSON para reconstruir */
            this.#rebuildModel(arg);
        }
    }

    /* Metodos publicos */
    /* **************** */

    async predictPreProcessed(predictionSample) {
        return this.#classifier.categorize(this.#createWord(predictionSample));
    }

    predict(predictionSample) {
        let localSample = predictionSample;
        /* Toda muestra ajena al dataset original debe replantearse */
        localSample = refactorSample(localSample, this.#preProcess);
        return this.predictPreProcessed(localSample);
    }

    getFeatureNames() {
        return this.#featureNames;
    }

    getDataSet() {
        return this.#dataSet;
    }

    getTrainingData() {
        return this.#trainingData;
    }

    getTestData() {
        return this.#testData;
    }

    whenTrained(callback) {
        console.log("whenTrained(callback)");
        this.#trainingFinished.then(callback);
    }

    toJSON() {
        return {
            modelRebuildSettings: this.#classifier.toJson(),
            trainAccuracy: this.#trainAccuracy,
            preProcess: this.#preProcess,
            featureNames: this.#featureNames
        };
    }

    summary() {
        MiscUtils.printHeader("Resultados de modelo")
        console.log(`Muestras de entrenamiento: ${this.#trainingData.samples.length}`);
        console.log(`Muestras de test: ${this.#testData.samples.length}`);
        // console.log(`Precision de entrenamiento: ${MiscUtils.trunc(this.#trainAccuracy * 100, 2)} % de acierto`);
        // console.log(`Precision de test: ${MiscUtils.trunc(this.#testAccuracy * 100, 2)} % de acierto`);
    }

    /* Metodos privados */
    /* **************** */
    #createModel(...args) {
        let argIndex = 0;
        this.#trainingData.samples = args[argIndex++];
        this.#trainingData.labels = args[argIndex++];
        this.#featureNames = args[argIndex++];
        this.#options = args[argIndex++];

        if (typeof this.#trainingData.samples == 'undefined' || this.#trainingData.samples == null || this.#trainingData.samples.length == 0) {
            throw 'Coleccion features no valida';
        }

        let firstSample = this.#trainingData.samples[0];
        if (!Array.isArray(firstSample) || firstSample.length != this.#featureNames.length) {
            throw 'La forma de las muestras no coincide con la esperada. Muestras: [,' +
            this.#featureNames.length +
            '], Forma recibida: [' +
            this.#trainingData.samples.length +
            ',' +
            firstSample.length +
            ']';
        }

        if (typeof this.#trainingData.labels == 'undefined' || this.#trainingData.labels == null || this.#trainingData.labels.length == 0) {
            throw 'Coleccion labels no valida';
        }

        let self = this;
        let trainingPromise = {};

        this.#classifier = Bayes()
        this.#trainingFinished = new Promise(function (resolve, reject) {
            trainingPromise.resolve = resolve;
            trainingPromise.reject = reject;
        });
        this.#trainingPromise = trainingPromise;
        this.#train();
    }

    #rebuildModel(...args) {
        let argIndex = 0;
        const jsonSettings = args[argIndex++];

        this.#JSONTrained = true;

        let modelRebuildJson = jsonSettings.modelRebuildSettings;
        this.#featureNames = jsonSettings.featureNames;
        this.#trainAccuracy = jsonSettings.trainAccuracy;
        this.#preProcess = jsonSettings.preProcess;

        this.#classifier = Bayes.fromJson(modelRebuildJson);
    }

    async #train() {
        let self = this;
        const trainingLabels = this.#trainingData.labels.map(o => o.toString());
        const trainingSamples = this.#formatSamples(this.#trainingData.samples, trainingLabels);

        let promises = [];
        for (let i = 0; i < trainingSamples.length; i++) {
            const pair = trainingSamples[i];
            promises.push(this.#classifier.learn(pair.word, pair.label));
        }

        console.log("#train");
        let trainingSolver = this.#trainingPromise.resolve;
        Promise.all(promises).then(() => {
            trainingSolver();
        });

        // this.#trainAccuracy = this.#classifier.evaluate(trainingSamples);

        // const testLabelsForEval = this.#testData.labels.map(o => o.toString());
        // const testSamplesForEval = this.#formatSamples(this.#testData.samples, testLabelsForEval);
        // this.#testAccuracy = this.#classifier.evaluate(testSamplesForEval);
    }

    #createWord(sample) {
        return sample.map(feat => feat.toString()).join(',');
    }

    #formatSamples(samplesToFormat, sampleLabels) {
        const words = samplesToFormat.map((sample, index) => {
            const word = this.#createWord(sample);
            const label = sampleLabels[index].toString();

            return { word, label };
        });
        return words;
    }

}

module.exports = {
    NaiveBayesModel
}