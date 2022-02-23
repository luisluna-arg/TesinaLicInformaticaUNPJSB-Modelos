const tf = require('@tensorflow/tfjs-node');
// @tensorflow/tfjs-node
const _ = require('lodash');

/**
 * Naive-Bayes Classifier
 * Takes an (optional) options object containing:
 */
class NaiveBayes {

    constructor(trainingFeatures, trainingLabels, options) {
        if (typeof trainingFeatures == 'undefined' || trainingFeatures == null || trainingFeatures.length == 0) {
            throw 'Coleccion features no valida';
        }

        if (typeof trainingLabels == 'undefined' || trainingLabels == null || trainingLabels.length == 0) {
            throw 'Coleccion labels no valida';
        }

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
            objectConverter: this.defaultObjectConverter,
            verbose: false,
            normalize: true
        }, options);


        this.trainingLabels = trainingLabels;

        if (this.options.normalize) {
            this.trainingFeatures = this.normalizeFeatures(trainingFeatures);
        }
        else {
            this.trainingFeatures = trainingFeatures;
        }

        this.objectConverter = this.options.objectConverter;

        this.vocabulary = {}
        this.vocabularySize = 0
        this.totalSamples = 0
        this.sampleCount = {}
        this.measurementCount = {}
        this.measurementFrequencyCount = {}
        this.labels = {}
    }

    defaultObjectConverter(sample) {
        return Object.keys(sample).map(property => sample[property]);
    }

    normalizeFeatures(data) {
        let result;

        tf.tidy(() => {
            let is2dTensor = data.length > 1 && Array.isArray(data[0]);
            result = (is2dTensor ? tf.tensor2d : tf.tensor)(data).transpose().log().transpose().arraySync();
        });

        return result;
    }

    getLabelToken(labelArray) {
        let labelToken = null;
        for (let i = 0; i < labelArray.length && labelToken == null; i++) {
            if (labelArray[i] > 0)
                labelToken = this.options.MoveTypeTokens[i + 1];
        }

        if (typeof labelToken == 'undefined' || labelToken == null) {
            labelToken = this.options.MoveTypeTokens[0];
        }

        return labelToken;
    }

    getLabelArray(labelToken) {
        let labelArray = new Array(this.options.MoveTypeTokens.length - 1).fill(0);

        for (let i = 0; i < labelArray.length; i++) {
            labelArray[i] = (this.options.MoveTypeTokens[i + 1] == labelToken) ? 1 : 0;
        }

        return labelArray;
    }

    initializeLabel(label) {
        // let labelToken = this.getLabelToken(labelToken);

        if (!this.labels[label]) {
            this.sampleCount[label] = 0
            this.measurementCount[label] = 0
            this.measurementFrequencyCount[label] = {}
            this.labels[label] = true
        }
        return this
    }

    learn(sample, labelArray) {
        let self = this
        let label = self.getLabelToken(labelArray);
        self.initializeLabel(label)
        self.sampleCount[label]++
        self.totalSamples++
        let sampleMeasurements = self.objectConverter(sample);
        let frequencyTable = self.frequencyTable(sampleMeasurements);

        Object
            .keys(frequencyTable)
            .forEach(function (feature) {
                if (!self.vocabulary[feature]) {
                    self.vocabulary[feature] = true
                    self.vocabularySize++
                }

                let frequencyInText = frequencyTable[feature]

                if (!self.measurementFrequencyCount[label][feature])
                    self.measurementFrequencyCount[label][feature] = frequencyInText
                else
                    self.measurementFrequencyCount[label][feature] += frequencyInText

                self.measurementCount[label] += frequencyInText
            })

        return self
    }

    predict(featuresToPredict) {
        let self = this
            , maxProbability = -Infinity
            , chosenLabel = null;

        if (this.options.normalize) {
            featuresToPredict = this.normalizeFeatures(featuresToPredict);
        }

        let convertedFeatures = self.objectConverter(featuresToPredict);
        let frequencyTable = self.frequencyTable(convertedFeatures);

        Object
            .keys(self.labels)
            .forEach(function (label) {
                let labelProbability = self.sampleCount[label] / self.totalSamples;
                //take the log to avoid underflow
                let logProbability = Math.log(labelProbability);

                // now determine P( w | c ) for each measurement
                Object
                    .keys(frequencyTable)
                    .forEach(function (measurement) {

                        // let flooredFeature = measurement;
                        let flooredFeature = Math.floor(measurement);
                        // let flooredFeature = Math.round(measurement, 2);
                        let frequencyInSamples = frequencyTable[flooredFeature];

                        let featureProbability = self.featureProbability(flooredFeature, label);

                        //determine the log of the P( w | c ) for this word
                        logProbability += frequencyInSamples * Math.log(featureProbability);
                    })

                if (logProbability > maxProbability) {
                    maxProbability = logProbability;
                    chosenLabel = label;
                }
            })

        let labelArray = this.getLabelArray(chosenLabel);
        return labelArray;
    }

    featureProbability(feature, label) {
        let wordFrequencyCount;
        if (typeof _.find(Object.keys(this.measurementFrequencyCount), k => k == label) == 'undefined' ||
            typeof _.find(Object.keys(this.measurementFrequencyCount[label]), k => k == feature) == 'undefined'
        ) {
            wordFrequencyCount = 0;
        }
        else {
            wordFrequencyCount = this.measurementFrequencyCount[label][feature] || 0;
        }

        let wordCount;
        if (typeof _.find(Object.keys(this.measurementCount), k => k == label) == 'undefined') {
            wordCount = 0;
        }
        else {
            wordCount = this.measurementCount[label];
        }

        //use laplace Add - 1 Smoothing equation
        return (wordFrequencyCount + 1) / (wordCount + this.vocabularySize)
    }

    frequencyTable = function (features) {
        let frequencyTable = Object.create(null);

        features.forEach(function (featureValue) {
            let frequencyKey = Math.floor(featureValue);
            if (typeof frequencyTable[frequencyKey] == 'undefined')
                frequencyTable[frequencyKey] = 0;

            frequencyTable[frequencyKey] = frequencyTable[frequencyKey] + 1;
        });

        return frequencyTable;
    }

    train() {
        for (let i = 0; i < this.trainingFeatures.length; i++) {
            const currentSample = this.trainingFeatures[i];
            const currentLabel = this.trainingLabels[i]
            this.learn(currentSample, currentLabel)
        }
    }

    test(testFeatures, testLabels) {
        if (testFeatures.length == 0) return 0;

        const self = this;

        let incorrect = 0;
        let comparison = [];

        for (let i = 0; i < testFeatures.length; i++) {
            const feature = testFeatures[i];
            const labelArray = testLabels[i];
            const predictedLabelArray = self.predict(feature);
            incorrect += _.isEqual(predictedLabelArray, labelArray) ? 0 : 1;
            comparison.push([labelArray, predictedLabelArray]);
        }

        return (testFeatures.length - incorrect) / testFeatures.length * 100;
    }

    summary() {
        console.info("");
        console.info("NaiveBayes");
        console.info("==========================================");
        console.info("Total Samples: ", this.totalSamples);
        console.info("Sample Count: ", this.sampleCount);
        console.info("Measurement Count: ", this.measurementCount);
        console.info("Measurement Frequency Count: ", this.measurementFrequencyCount);
        console.info("Labels: ", this.labels);
        console.info("");
    }

    printTrainingData() {
        console.info("");
        console.info("NaiveBayes - Training Data");
        console.info("==========================================");
        console.info("Features: ", this.trainingFeatures);
        console.info("Labels: ", this.trainingLabels);
        console.info("");
    }
}

module.exports = {
    Naivebayes: NaiveBayes
};