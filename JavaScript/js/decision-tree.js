const DecisionTree = require('decision-tree');

const tf = require('@tensorflow/tfjs-node');
const _ = require('lodash');


/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

const FEATURE_NAMES = [
    "delta",
    "theta",
    "lowAlpha",
    "highAlpha",
    "lowBeta",
    "highBeta",
    "lowGamma",
    "highGamma"
];

const CLASS_NAME = "moveType";

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

class Sample {

    constructor(sampleValues, label) {
        for (let i = 0; i < sampleValues.length; i++) {
            const featureName = FEATURE_NAMES[i];
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


        this.decisionTree = new DecisionTree(CLASS_NAME, FEATURE_NAMES);
        this.trainAccuracy = 0;
        this.testAccuracy = 0;
        this.JSONTrained = false;

        if (args.length == 1) {
            json = args[0];
            this.decisionTree.import(json);
            this.JSONTrained = true;
        }
        else if (args.length == 3) {
            trainingSamples = args[0];
            trainingLabels = args[1];
            options = args[2];

            if (typeof trainingSamples == 'undefined' || trainingSamples == null || trainingSamples.length == 0) {
                throw 'Coleccion features no valida';
            }

            let firstSample = trainingSamples[0];
            if (!Array.isArray(firstSample) || firstSample.length != FEATURE_NAMES.length) {
                throw 'La forma de las muestras no coincide con la esperada. Muestras: [,' +
                FEATURE_NAMES.length +
                '], Forma recibida: [' +
                trainingSamples.length +
                ',' +
                firstSample.length +
                ']';
            }

            if (typeof trainingLabels == 'undefined' || trainingLabels == null || trainingLabels.length == 0) {
                throw 'Coleccion labels no valida';
            }

            this.trainingLabels = trainingLabels;
            this.trainingData = this.formatSamples(trainingSamples, this.trainingLabels);
        }

    }

    train() {
        if (this.JSONTrained) {
            console.log("Can't train, tree already trained via JSON");
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
            const label = testLabels[i];

            let predictedClass = this.predict(sample);

            // console.log("sample", sample);
            // console.log("label", label, "predictedClass", predictedClass);

            incorrect += predictedClass === label ? 0 : 1;
            comparison.push([label, predictedClass]);
        }

        return (testSamples.length - incorrect) / testSamples.length * 100;
    }

    predict(predictionSample) {
        return this.decisionTree.predict(new Sample(predictionSample));
    }

    summary() {
        console.log("");
        console.log("Summary");
        console.log("=======");
        console.log("Samples", this.trainingData.length);
        console.log("Feature count", FEATURE_NAMES.length);
        console.log("Train Accuracy", this.trainAccuracy);
        console.log("Test Accuracy", this.testAccuracy);
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
            let sampleInstance = new Sample(values, label);
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
    DirectionDecisionTree: DirectionDecisionTree
};
