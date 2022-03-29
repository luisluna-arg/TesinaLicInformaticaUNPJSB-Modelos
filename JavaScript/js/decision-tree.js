const DecisionTree = require('decision-tree');

const tf = require('@tensorflow/tfjs-node');
const _ = require('lodash');


/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

const CLASS_NAME = "moveType";

function floatToFixed(number) {
    return parseFloat(number.toFixed(8));
}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

class Sample {

    constructor(sampleValues, featureNames, label) {
        for (let i = 0; i < sampleValues.length; i++) {
            const featureName = featureNames[i];
            this[featureName] = sampleValues[i];
        }

        if (label) {
            this[CLASS_NAME] = label;
        }
    }

}

class DirectionDecisionTree {

    // constructor(trainingSamples, trainingLabels, options) {
    constructor(...args) {

        let trainingSamples = null;
        let trainingLabels = null;
        let options = null
        let json = null;

        /* Set default option settings */
        this.options = Object.assign({
            MoveTypeEnum: {
                NONE: 0,
                DOWN: 1,
                UP: 2,
                LEFT: 3,
                RIGHT: 4
            },
            MoveTypeTokens: [
                'NONE',
                'DOWN',
                'UP',
                'LEFT',
                'RIGHT'
            ],
            verbose: false
        }, options);


        this.trainAccuracy = 0;
        this.testAccuracy = 0;
        this.JSONTrained = false;

        let argIndex = 0;
        if (args.length == 2) {
            json = args[argIndex++];
            this.FEATURE_NAMES = args[argIndex++];
            
            this.decisionTree = new DecisionTree(CLASS_NAME, this.FEATURE_NAMES);
            this.decisionTree.import(json);
            this.JSONTrained = true;
        }
        else if (args.length == 4) {
            trainingSamples = args[argIndex++];
            trainingLabels = args[argIndex++];
            this.FEATURE_NAMES = args[argIndex++];
            options = args[argIndex++];

            if (typeof trainingSamples == 'undefined' || trainingSamples == null || trainingSamples.length == 0) {
                throw 'Coleccion features no valida';
            }

            let firstSample = trainingSamples[0];
            if (!Array.isArray(firstSample) || firstSample.length != this.FEATURE_NAMES.length) {
                throw 'La forma de las muestras no coincide con la esperada. Muestras: [,' +
                this.FEATURE_NAMES.length +
                '], Forma recibida: [' +
                trainingSamples.length +
                ',' +
                firstSample.length +
                ']';
            }

            if (typeof trainingLabels == 'undefined' || trainingLabels == null || trainingLabels.length == 0) {
                throw 'Coleccion labels no valida';
            }

            this.decisionTree = new DecisionTree(CLASS_NAME, this.FEATURE_NAMES);
            this.trainingLabels = trainingLabels;
            this.trainingData = this.formatSamples(trainingSamples, this.trainingLabels);
        }
    }

    train() {
        if (this.JSONTrained) {
            console.log("No se puede entrenar, arbol creado mediante JSON");
            return;
        }

        let localTrainingData = this.trainingData;
        this.decisionTree.train(localTrainingData);
        this.trainAccuracy = this.decisionTree.evaluate(localTrainingData);
    }

    test(testSamples, testLabels) {
        if (testSamples.length == 0) return 0;

        this.testAccuracy = this.decisionTree.evaluate(this.createSamples(testSamples, testLabels));

        let incorrect = 0;
        let comparison = [];

        for (let i = 0; i < testSamples.length; i++) {
            const sample = testSamples[i];
            const label = testLabels[i][0];

            let predictedClass = this.predict(sample)[0];
            incorrect += predictedClass === label ? 0 : 1;
            comparison.push([label, predictedClass]);
        }

        return { precision: (testSamples.length - incorrect) / testSamples.length * 100 };
    }

    predict(predictionSample) {
        return this.decisionTree.predict(new Sample(predictionSample, this.FEATURE_NAMES));
    }

    summary() {
        console.log("");
        console.log("Summary");
        console.log("=======");
        console.log("Samples", this.trainingData.length);
        console.log("Feature count", this.FEATURE_NAMES.length);
        console.log("Test Accuracy", floatToFixed(this.testAccuracy * 100));
        console.log("");
    }

    toJSON() {
        return this.decisionTree.toJSON();
    }

    /* ********************************************************* */

    normalizeFeatures(data) {
        let result;

        tf.tidy(() => {
            let is2dTensor = data.length > 1 && Array.isArray(data[0]);
            result = (is2dTensor ? tf.tensor2d : tf.tensor)(data).transpose().log().transpose().arraySync();
        });

        return result;
    }

    createSamples(sampleValues, sampleLabels) {
        let results = [];
        for (let i = 0; i < sampleValues.length; i++) {
            let values = sampleValues[i];
            let label = sampleLabels[i];
            let sampleInstance = new Sample(values, this.FEATURE_NAMES, label);
            results.push(sampleInstance);
        }
        return results;


        // return sampleValues;
    }

    formatSamples(samplesToFormat, sampleLabels) {
        let localSamples = samplesToFormat;
        if (this.options.normalize) {
            localSamples = this.normalizeFeatures(localSamples);
        }
        return this.createSamples(localSamples, sampleLabels);
    }

}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

module.exports = {
    model: DirectionDecisionTree
}
