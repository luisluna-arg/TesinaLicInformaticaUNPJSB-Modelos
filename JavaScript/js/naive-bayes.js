const tf = require('@tensorflow/tfjs-node');
// @tensorflow/tfjs-node
const _ = require('lodash');

const labelMapper = (labelValue) => {
    let result = new Array(4);
    result.fill(0);
    result[labelValue - 1] = 1;
    return result;
};

const formatFloat = (value, decimals = 8) => parseFloat(value.toFixed(decimals));

const formatFloatArray = (value, decimals = 8) => parseFloat(value.dataSync()[0].toFixed(decimals));


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
            normalize: false
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

    getLabelToken(labelValue) {
        let labelToken = this.options.MoveTypeTokens[labelValue];

        if (typeof labelToken == 'undefined' || labelToken == null) {
            labelToken = this.options.MoveTypeTokens[0];
        }

        return labelToken;
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
                        // let flooredFeature = Math.floor(measurement);
                        let flooredFeature = Math.round(measurement, 2);
                        let frequencyInSamples = frequencyTable[flooredFeature];

                        let featureProbability = self.featureProbability(flooredFeature, label);

                        //determine the log of the P( w | c ) for this word
                        logProbability += frequencyInSamples * Math.log(featureProbability);
                    })

                if (logProbability > maxProbability) {
                    maxProbability = logProbability;
                    chosenLabel = label;
                }
            });

        let prediction = this.options.MoveTypeEnum[chosenLabel];

        return prediction;
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

        let finalResult = null;
        let predictionsValues = [];
        let labelValues = [];
        const predictions = [];

        for (let i = 0; i < testFeatures.length; i++) {
            const sample = testFeatures[i];
            const expectedLabelArray = testLabels[i][0];
            const predictionLabelArray = self.predict(sample);

            const predictionResult = [predictionLabelArray, expectedLabelArray, _.isEqual(predictionLabelArray, expectedLabelArray)];

            predictions.push(predictionResult);

            predictionsValues.push(predictionLabelArray);
            labelValues.push(expectedLabelArray);
        }

        const total = predictions.length;
        const correct = predictions.filter(o => o[2]).length;
        let precision = formatFloat(correct / total * 100);
               
        const { mean, variance } = tf.moments(predictionsValues, 0);
        let varianceVal = formatFloatArray(variance);
        let devStd = formatFloatArray(tf.sqrt(varianceVal));
        let meanVal = formatFloatArray(mean);

        finalResult = {
            mean: meanVal, 
            variance: varianceVal, 
            precision,
            devStd
        };

        return finalResult;
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
    model: NaiveBayes
};