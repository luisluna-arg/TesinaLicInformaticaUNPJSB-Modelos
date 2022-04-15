const DecisionTree = require('decision-tree');

const decisionTreeSettings = {
    filter: false,
    shuffle: true,
    split: false,
    truncate: true,
    decimals: 4,
    normalization: true,
    fourier: true,
    deviationMatrix: true,
    dataAugmentation: false,
    selectFeatures: false,
    dataAugmentationTotal: 40000, /* Muestras totales cada vez que un un archivo o lista de archivos es aumentado */
    minTolerance: 0.0 /* entre 0 y 1, 0 para que traiga todo */
};

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
    #decisionTree = null;
    #featureNames = null;
    #trainingSamples = null;
    #trainingLabels = null;
    #options = null;

    constructor(...args) {
        let argIndex = 0;
        if (args.length == 2) {
            json = args[argIndex++];
            this.#featureNames = args[argIndex++];

            this.#decisionTree = new DecisionTree(CLASS_NAME, this.#featureNames);
            this.#decisionTree.import(json);
            this.#JSONTrained = true;
        }
        else if (args.length == 4) {
            this.#trainingSamples = args[argIndex++];
            this.#trainingLabels = args[argIndex++];
            this.#featureNames = args[argIndex++];
            this.#options = args[argIndex++];

            if (typeof this.#trainingSamples == 'undefined' || this.#trainingSamples == null || this.#trainingSamples.length == 0) {
                throw 'Coleccion features no valida';
            }

            let firstSample = this.#trainingSamples[0];
            if (!Array.isArray(firstSample) || firstSample.length != this.#featureNames.length) {
                throw 'La forma de las muestras no coincide con la esperada. Muestras: [,' +
                this.#featureNames.length +
                '], Forma recibida: [' +
                this.#trainingSamples.length +
                ',' +
                firstSample.length +
                ']';
            }

            if (typeof this.#trainingLabels == 'undefined' || this.#trainingLabels == null || this.#trainingLabels.length == 0) {
                throw 'Coleccion labels no valida';
            }

            this.#decisionTree = new DecisionTree(CLASS_NAME, this.#featureNames);
            this.#trainingLabels = this.#trainingLabels.map(o => o.toString());
            this.#trainingSamples = this.#formatSamples(this.#trainingSamples, this.#trainingLabels);
            this.#train();
        }

    }

    /* Metodos publicos */
    /* **************** */

    predict(predictionSample) {
        if (this.#JSONTrained) {
        }
        else {
            const sampleToPredict = new Sample(predictionSample, this.#featureNames);
            return this.#decisionTree.predict(sampleToPredict);
        }
    }

    toJSON() {
        return {
            settings: this.#decisionTree.toJSON(),
            trainAccuracy: this.#trainAccuracy
        };
    }

    /* Metodos privados */
    /* **************** */

    #train() {
        this.#decisionTree.train(this.#trainingSamples);
        this.#trainAccuracy = this.#decisionTree.evaluate(this.#trainingSamples);
    }

    #formatSamples(samplesToFormat, sampleLabels) {
        let localSamples = samplesToFormat;
        if (this.#options.normalize) {
            localSamples = this.#normalizeFeatures(localSamples);
        }
        return this.#createSamples(localSamples, sampleLabels);
    }

    #normalizeFeatures(data) {
        let result;

        tf.tidy(() => {
            let is2dTensor = data.length > 1 && Array.isArray(data[0]);
            result = (is2dTensor ? tf.tensor2d : tf.tensor)(data).transpose().log().transpose().arraySync();
        });

        return result;
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
    DecisionTreeModel
}