const MiscUtils = require('./misc-utils');
const { MOVE_TYPE, loadJSON, splitData, dataPreProcessing } = require('./load-training-json');
const { refactorSample, confusionMatrix } = require('./data-preprocessing');
const DecisionTree = require('decision-tree');

const CLASS_NAME = "moveType";

/* SETTINGS */
/* //////// */

const ExportBasePath = './data/';
const DataSetExportPath = ExportBasePath + 'decisiontree-data.csv';
const SettingsExportPath = ExportBasePath + 'decisiontree-settings.json';

const DataLoadingSettings = {
    preProcess: false,
    filter: true,
    shuffle: true,
    split: false,
    truncate: true,
    decimals: 4,
    normalization: true,
    fourier: true,
    deviationMatrix: true,
    selectFeatures: false,
    dataSetExportPath: DataSetExportPath,
    settingsExportPath: SettingsExportPath,
    minTolerance: 0.0, /* entre 0 y 1, 0 para que traiga todo */
    dataAugmentationTotal: 170000, /* Muestras totales cada vez que un un archivo o lista de archivos es aumentado */
    dataAugmentation: true
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
    const indexing = preProcessResult.indexing;

    /* ENTRENAMIENTO */
    /* ///////////// */
    splitData(loadedData);

    const samples = loadedData.getSamples();
    const labels = loadedData.getLabels();

    /* Data de entrenamiento */
    const trainingLength = Math.floor(samples.length * 0.7);
    const trainingData = samples.slice(0, trainingLength);
    const trainingLabels = labels.slice(0, trainingLength);
    const trainingIndexing = indexing.slice(0, trainingLength);

    /* Data de pruebas */
    const testData = samples.slice(trainingLength);
    const testLabels = labels.slice(trainingLength);
    const testIndexing = indexing.slice(trainingLength);

    // loadedData.summary();

    return {
        featureNames: loadedData.getFeatureNames(),
        dataSet: dataSet,
        indexing: indexing,
        training: {
            samples: trainingData,
            labels: trainingLabels,
            indexing: trainingIndexing
        },
        test: {
            samples: testData,
            labels: testLabels,
            indexing: testIndexing
        },
        preProcess: {
            stats: preProcessResult.stats,
            trainingSettings: preProcessResult.trainingSettings
        }
    };
}

function loadDataSet(fileBasePath) {

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

class DecisionTreeModel {

    #JSONTrained = false;
    #trainAccuracy = 0;
    #testAccuracy = 0;
    #decisionTree = null;
    #featureNames = null;
    #preProcess = null;
    #testData = {};
    #dataSet = {};
    #indexing = null;
    #trainingData = {};
    #options = {

    };

    constructor(arg) {
        if (typeof arg == 'string') {
            /* Recibe el path de archivos de entrenamiento de entrenamiento */
            let { training, test, featureNames, preProcess, dataSet, indexing } = loadData(arg);

            this.#testData = test;
            this.#preProcess = preProcess;
            this.#dataSet = dataSet;
            this.#indexing = indexing;
            this.#createModel(training.samples, training.labels, featureNames, ModelTrainingSettings);
        }
        else {
            /* Recibe el JSON para reconstruir */
            this.#rebuildModel(arg);
        }
    }

    /* Metodos publicos */
    /* **************** */

    predictPreProcessed(predictionSample) {
        let localSample = predictionSample;
        return this.#decisionTree.predict(new Sample(localSample, this.#featureNames));
    }

    predict(predictionSample) {
        let resampled = predictionSample;

        let refactorSettings = Object.assign({}, this.#preProcess);
        refactorSettings.trainingSettings = Object.assign(
            {}, refactorSettings.trainingSettings
            , {
                normalization: true,
                fourier: true,
                dataAugmentation: false,
                shuffle: false
            }
        );

        /* Toda muestra ajena al dataset original debe replantearse */
        resampled = refactorSample(resampled, refactorSettings, this.#dataSet);

        // let newDataSet = this.#dataSet;

        // let newPredictinSample = predictionSample;
        // newPredictinSample.push(-1); // Agregar una clase -1 para que sea compatible con el modelo
        // newDataSet = newDataSet.slice(0, newDataSet.length - 1);
        // newDataSet.push(predictionSample)
        // let dataSet = preProcess(newDataSet, this.#featureNames, refactorSettings.trainingSettings);

        // // let localSettings = Object.assign({}, {
        // //     filter: false, normalization: true, fourier: true,
        // //     deviationMatrix: true, truncate: true, decimals: 8
        // // }, refactorSettings.trainingSettings);
        // // console.log("refactorSettings.trainingSettings", localSettings);
        
        // resampled = dataSet.data[dataSet.data.length - 1];


        // console.log("resampled", resampled);

        return this.#decisionTree.predict(new Sample(resampled, this.#featureNames));
    }

    getFeatureNames() {
        return this.#featureNames;
    }

    getDataSet() {
        return this.#dataSet;
    }

    getDataSetIndexing() {
        return this.#indexing;
    }

    getTrainingData() {
        return this.#trainingData;
    }

    getTestData() {
        return this.#testData;
    }

    toJSON() {
        return {
            modelRebuildSettings: this.#decisionTree.toJSON(),
            trainAccuracy: this.#trainAccuracy,
            testAccuracy: this.#testAccuracy,
            preProcess: this.#preProcess,
            featureNames: this.#featureNames,
            options: this.#options
        };
    }

    summary() {
        MiscUtils.printHeader("Resultados de modelo")
        console.log(`Muestras de entrenamiento: ${this.#trainingData.samples.length}`);
        console.log(`Muestras de test: ${this.#testData.samples.length}`);
        console.log(`Precision de entrenamiento: ${MiscUtils.trunc(this.#trainAccuracy * 100, 2)} % de acierto`);
        console.log(`Precision de test: ${MiscUtils.trunc(this.#testAccuracy * 100, 2)} % de acierto`);
    }

    /**
     * Exporta el dataset del modelo a un CSV.
     * Tiene como fin dejar el dataset manipulable para poder calcular fourier 
     * al aplicar sobre una unica muestra.
     */
    exportDataSet() {
        let localSettings = Object.assign({}, DataLoadingSettings, {
            // normalization: true,
            // fourier: false,
            // deviationMatrix: false,
            dataAugmentation: false,
            // truncate: false
        });

        const path = !MiscUtils.isNullOrUndef(localSettings.dataSetExportPath) ?
            localSettings.dataSetExportPath : DataSetExportPath;

        // let dataSetNoClass = this.#dataSet.map(o => o.slice(0, o.length - 1));
        // dataSetNoClass.push();

        // let preProcessedData = preProcess(this.#dataSet, this.#featureNames, localSettings);

        // MiscUtils.writeDataSetCSV(path, preProcessedData.data);
        MiscUtils.writeDataSetCSV(path, this.#dataSet);
    }

    exportSettings() {
        let localSettings = Object.assign({}, DataLoadingSettings, {
        });
        const path = !MiscUtils.isNullOrUndef(localSettings.dataSetExportPath) ?
            localSettings.settingsExportPath : SettingsExportPath;
        MiscUtils.writeJSON(path, this.toJSON());
    }

    /* **************************************************************************** */


    /* Metodos privados */
    /* **************** */
    #createModel(...args) {
        let argIndex = 0;
        this.#trainingData.samples = args[argIndex++];
        this.#trainingData.labels = args[argIndex++];
        this.#featureNames = args[argIndex++];

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

        this.#decisionTree = new DecisionTree(CLASS_NAME, this.#featureNames);
        // this.#trainingData.labels = this.#trainingData.labels.map(o => o.toString());
        // this.#trainingData.samples = this.#formatSamples(this.#trainingData.samples, this.#trainingData.labels);
        this.#train();
    }

    #rebuildModel(jsonSettings) {
        this.#JSONTrained = true;

        let modelRebuildJson = jsonSettings.modelRebuildSettings;
        this.#featureNames = jsonSettings.featureNames;
        this.#trainAccuracy = jsonSettings.trainAccuracy;
        this.#preProcess = jsonSettings.preProcess;
        this.#options = jsonSettings.options;

        if (!MiscUtils.isNullOrUndef(DataLoadingSettings.dataSetExportPath)) {
            this.#dataSet = MiscUtils.readDataSetCSV(DataLoadingSettings.dataSetExportPath);
        }

        this.#decisionTree = new DecisionTree(CLASS_NAME, this.#featureNames);
        this.#decisionTree.import(modelRebuildJson);
    }

    #train() {
        const trainingLabels = this.#trainingData.labels.map(o => o.toString());
        const trainingSamples = this.#formatSamples(this.#trainingData.samples, trainingLabels);
        this.#decisionTree.train(trainingSamples);
        this.#trainAccuracy = this.#decisionTree.evaluate(trainingSamples);

        const testLabelsForEval = this.#testData.labels.map(o => o.toString());
        const testSamplesForEval = this.#formatSamples(this.#testData.samples, testLabelsForEval);
        this.#testAccuracy = this.#decisionTree.evaluate(testSamplesForEval);
    }

    #formatSamples(samplesToFormat, sampleLabels) {
        return this.#createSamples(samplesToFormat, sampleLabels);
    }

    #createSamples(sampleValues, sampleLabels) {
        let results = [];
        for (let i = 0; i < sampleValues.length; i++) {
            let values = sampleValues[i];
            let label = sampleLabels[i];
            let sampleInstance = new Sample(values, this.#featureNames, label);
            results.push(sampleInstance);
        }
        return results;
    }

}

module.exports = {
    DecisionTreeModel,
    confusionMatrix
}