const MiscUtils = require('./misc-utils');
const { MOVE_TYPE, loadJSON, splitData, dataPreProcessing } = require('./load-training-json');
const { refactorSample, confusionMatrix } = require('./data-preprocessing');
const tf = require('@tensorflow/tfjs-node');

/* SETTINGS */
/* //////// */

const ExportBasePath = './data/';
const DataSetExportPath = ExportBasePath + 'linear-regression-data.csv';
const PreProcessedDataSetExportPath = ExportBasePath + 'linear-regression-preprocessed-data.csv';
const SettingsExportPath = ExportBasePath + 'linear-regression-settings.json';

let DataLoadingSettings = {
    preProcess: false,
    classRemap: true, /* Determina si se remapea una clase segun un criterio de limpieza de ruido */
    shuffle: true,
    split: false,
    truncate: true,
    decimals: 4,
    normalization: true,
    fourier: true,
    deviationMatrix: true,
    dataSetExportPath: DataSetExportPath,
    preProcessedDataSetExportPath: PreProcessedDataSetExportPath,
    settingsExportPath: SettingsExportPath,
    minTolerance: 0.0, /* entre 0 y 1, 0 para que traiga todo */
    dataAugmentationTotal: 30000, /* Muestras totales cada vez que un un archivo o lista de archivos es aumentado */
    dataAugmentation: true
};

const ModelTrainingSettings = {
    MoveTypeEnum: MOVE_TYPE,
    verbose: false
};

function loadData(fileBasePath, loadingSettings) {
    if (loadingSettings != null) {
        DataLoadingSettings = Object.assign({}, DataLoadingSettings, loadingSettings);
    }

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
    preProcessedDataSet = loadedData.getSamples();

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
        preProcessedDataSet: preProcessedDataSet,
        training: {
            samples: trainingData,
            labels: trainingLabels
        },
        test: {
            samples: testData,
            labels: testLabels
        },
        preProcess: {
            stats: preProcessResult.stats,
            trainingSettings: preProcessResult.trainingSettings
        }
    };
}

class LinearRegressionModel {

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
    #preProcessedDataSet = {};
    #costHistory = null;
    #learningRateHistory = null;
    #compileSettings = null;
    #JSONTrained = false;

    constructor(...args) {
        if (args.length > 1 && typeof args[0] == 'string') {
            /* Recibe el path de archivos de entrenamiento */
            let filePath = args[0];
            let loadingSettings = args.length == 2 ? args[1] : null;

            let { training, test, featureNames, preProcess, dataSet, preProcessedDataSet } = loadData(filePath, loadingSettings);

            this.#testData = test;
            this.#preProcess = preProcess;
            this.#dataSet = dataSet;
            this.#preProcessedDataSet = preProcessedDataSet;
            this.#createModel(training.samples, training.labels, featureNames, ModelTrainingSettings);
        }
        else {
            /* Recibe el JSON para reconstruir */
            this.#rebuildModel(args[0]);
        }
    }

    /* Metodos publicos */
    /* **************** */

    predictPreProcessed(predictionSample) {
        return this.#remapLabelArray(this.#classifier.predict(tf.tensor([predictionSample])));
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
        resampled = refactorSample(resampled, refactorSettings);

        return this.predictPreProcessed(resampled);
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

    setTestAccuracy(accuracy) {
        this.#testAccuracy = accuracy;
    }

    whenTrained(callback) {
        this.#trainingFinished.then(callback);
    }

    toJSON() {
        this.#classifier.save('file://./model-1a');
        return {
            modelRebuildSettings: this.#classifier.save(),
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
        console.log(`Precision de test: ${MiscUtils.trunc(this.#testAccuracy * 100, 2)} % de acierto`);
        this.#classifier.summary();
    }

    /**
     * Exporta el dataset del modelo a un CSV.
     */
    exportDataSet() {
        let localSettings = Object.assign({}, DataLoadingSettings, {
            dataAugmentation: false,
        });

        const path = !MiscUtils.isNullOrUndef(localSettings.dataSetExportPath) ?
            localSettings.dataSetExportPath : DataSetExportPath;

        MiscUtils.writeDataSetCSV(path, this.#dataSet);
    }

    exportPreProcessDataSet() {
        let localSettings = Object.assign({}, DataLoadingSettings, {
            dataAugmentation: false,
        });

        const path = !MiscUtils.isNullOrUndef(localSettings.preProcessedDataSetExportPath) ?
            localSettings.preProcessedDataSetExportPath : PreProcessedDataSetExportPath;

        MiscUtils.writeDataSetCSV(path, this.#preProcessedDataSet);
    }

    exportSettings() {
        let localSettings = Object.assign({}, DataLoadingSettings, {});
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

        /* Set default option settings */
        this.#options = Object.assign({}, {
            learningRate: 0.1,
            iterations: 250,
            batchSize: 1000,
            decisionBoundary: 0.5,
            subject: "TestSubject_2",
            useReLu: false,
            shuffle: false,
            normalize: false
        });

        this.#costHistory = [];
        this.#learningRateHistory = [];

        let trainingPromise = {};

        /* Define model compilation settings */
        this.#compileSettings = {
            optimizer: tf.train.sgd(0.001),
            loss: "meanSquaredError",
            weights: this.weights,
            metrics: [
                tf.metrics.MSE,
                tf.metrics.binaryAccuracy,
                tf.metrics.meanAbsoluteError
            ]            
        };

        // Defines a simple logistic regression model with 32 dimensional input
        // and 3 dimensional output.
        let inputShape = this.#trainingData.samples[0].length; //this.samples.shape[1];
        let outputShape = 5;


        // Define a model for linear regression.
        this.#classifier = tf.sequential();
        this.#classifier.add(tf.layers.dense({ 
            units: outputShape, 
            inputShape: inputShape 
        }));

        this.#trainingFinished = new Promise(function (resolve, reject) {
            trainingPromise.resolve = resolve;
            trainingPromise.reject = reject;
        });
        this.#trainingPromise = trainingPromise;
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

        this.#classifier = Bayes.fromJson(modelRebuildJson);
    }

    async #train() {
        // Prepare the model for training: Specify the loss and the optimizer.
        // this.#classifier.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
        this.#classifier.compile(this.#compileSettings);

        // Generate some synthetic data for training.
        let samplesArray = this.#trainingData.samples;
        let labelsArray = this.#trainingData.labels.map(this.#remapLabel);

        const xs = tf.tensor2d(samplesArray);
        const ys = tf.tensor2d(labelsArray);

        let promises = [
            this.#classifier.fit(xs, ys, {
                batchSize: this.#options.batchSize,
                epochs: this.#options.iterations,
                verbose: false
            })
        ];

        let trainingSolver = this.#trainingPromise.resolve;
        Promise.all(promises).then(() => {
            trainingSolver();
        });
    }

    #remapLabel(label) {
        let result = new Array(5);
        result.fill(0);
        result[label] = 1;
        return result;
    }

    #remapLabelArray(labelTensor) {
        // console.log("labelTensor", labelTensor.arraySync());
        // let aa = labelTensor.argMax(1);
        // console.log("labelTensor", aa);
        // console.log("labelTensor", labelTensor.argMax(1).arraySync()[0]);
        return labelTensor.argMax(1).arraySync()[0];
    }
}

module.exports = {
    LinearRegressionModel,
    confusionMatrix
}