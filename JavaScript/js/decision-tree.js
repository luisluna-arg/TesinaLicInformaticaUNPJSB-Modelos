const DecisionTree = require('decision-tree');

const tf = require('@tensorflow/tfjs-node');
const _ = require('lodash');
const { isNull } = require('lodash');


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
            this[featureName] = sampleValues[i].toString();
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
            this.trainingLabels = trainingLabels.map(o => o.toString());
            this.trainingSamples = this.formatSamples(trainingSamples, this.trainingLabels);
        }
    }

    train() {
        if (this.JSONTrained) {
            console.log("No se puede entrenar, arbol creado mediante JSON");
            return;
        }

        this.decisionTree.train(this.trainingSamples);
        this.trainAccuracy = this.decisionTree.evaluate(this.trainingSamples);
    }

    test(testSamples, testLabels) {
        if (testSamples.length == 0) return 0;
        
        let currentSamples = this.createSamples(testSamples, testLabels);
        this.testAccuracy = this.decisionTree.evaluate(currentSamples);

        let correct = 0;
        let comparison = [];

        console.log("this.trainingSamples.length", this.trainingSamples.length);
        console.log("this.trainingLabels.length", this.trainingLabels.length);
        console.log("this.trainingLabels.filter(o => o < 1)", this.trainingLabels.filter(o => o < 1));

        this.predictedLabelCount = { '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 'undefined': 0 };
        this.actualLabelCount = { '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 'undefined': 0 };
        for (let i = 0; i < testSamples.length; i++) {
            const sample = testSamples[i];
            const label = testLabels[i];
            this.actualLabelCount[label]++;

            let predictedLabel = this.predict(sample);
            // console.log("predictedLabel", predictedLabel);
            // console.log("sample", sample);
            // console.log("label", label);
            this.predictedLabelCount[predictedLabel]++;

            const equals = predictedLabel == label;
            correct += equals ? 1 : 0;
            comparison.push([label, predictedLabel, equals]);
        }

        const result = { 
            testLength: testSamples.length,
            correct,
            precision: correct / testSamples.length * 100
        };

        return result;
    }

    predict(predictionSample) {
        return this.decisionTree.predict(new Sample(predictionSample, this.FEATURE_NAMES));
    }

    summary() {
        console.log("");
        console.log("Summary");
        console.log("=======");
        console.log("Samples", this.trainingSamples.length);
        console.log("Feature count", this.FEATURE_NAMES.length);
        console.log("Training Accuracy", floatToFixed(this.trainAccuracy * 100).toString() + '%');
        console.log("Test Accuracy", floatToFixed(this.testAccuracy * 100).toString() + '%');
        console.log("Predicted label count", this.predictedLabelCount);
        console.log("Actual label count", this.actualLabelCount);
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
