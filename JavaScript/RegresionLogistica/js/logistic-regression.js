const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
    
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.costHistory = [];
        this.learningRateHistory = [];

        this.options = Object.assign({
            learningRate: 0.1,
            iterations: 1000,
            batchSize: 50000,
            decisionBoundary: 0.5
        }, options);

        let rowCount = this.features.shape[1];
        let columnCount = 4; /* How many classes are there */
        this.weights = tf.zeros([rowCount, columnCount]);

        this.learningRateHistory.push(this.options.learningRate);
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).sigmoid();
        const differences = currentGuesses.sub(labels);
        const slopes = features.transpose().matMul(differences).div(features.shape[0]);
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    train() {
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchQuantity; j++) {
                const { batchSize } = this.options;
                const startIndex = j * batchSize;
                const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
                const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
                this.gradientDescent(featureSlice, labelSlice);
            }
            this.recordCost();
            this.updateLearningRate();
        }
    }

    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures);
        let testLabelsTensor = tf.tensor(testLabels);
        const incorrect = predictions.sub(testLabelsTensor).abs().sum().dataSync()[0];
        const itemCount = predictions.shape[0];
        return (itemCount - incorrect) / predictions.shape[0] * 100;
    }

    processFeatures(features) {
        features = tf.tensor(features);
        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        }
        else {
            features = this.standarize(features);
        }
        features = tf.ones([features.shape[0], 1]).concat(features, 1);
        return features;
    }

    standarize(features) {
        const { mean, variance } = tf.moments(features, 0);
        this.mean = mean;
        this.variance = variance;
        return features.sub(this.mean).div(this.variance.pow(0.5));
    }

    recordCost() {
        const guesses = this.features.matMul(this.weights).sigmoid();
        const termOne = this.labels.transpose().matMul(guesses.log());
        const termTwo = this.labels.mul(-1).add(1).transpose().matMul(guesses.mul(-1).add(1).log());
        let termThree = termOne.add(termTwo).div(this.features.shape[0]).mul(-1);
        let termThree_array = termThree.arraySync();
        const cost = termThree_array[0][0]//;.get(0, 0);
        console.log(termThree_array);
        this.costHistory.unshift(cost);
    }

    updateLearningRate() {
        if (this.costHistory.length < 2) return;

        let learnRate = this.options.learningRate;
        this.options.learningRate = (this.costHistory[0] > this.costHistory[1]) ? learnRate / 2 : learnRate * 1.05;
        this.learningRateHistory.push(this.options.learningRate);
    }

    predict(observations) {
        return this.processFeatures(observations).
            matMul(this.weights).
            //sigmoid().
            softmax().
            greater(this.options.decisionBoundary).
            cast('float32');
    }
}

module.exports = LogisticRegression;