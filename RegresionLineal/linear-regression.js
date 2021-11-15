const tf = require('@tensorflow/tfjs');
const { result } = require('lodash');
const _ = require('lodash');

class LogisticRegression {
    constructor(features, labels, options) {
        this.costHistory = []; // Cross-Entropy values
        this.learningRateHistory = [];
        this.means = [];
        this.variances = [];

        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);

        this.options = Object.assign({
            learningRate: 0.1,
            iterations: 1000,
            batchSize: 50000,
            decisionBoundary: 0.5
        }, options);

        this.weights = tf.zeros([this.features.shape[1], 1]);

        this.learningRateHistory.push(this.options.learningRate);
    }

    processFeatures(features) {
        let localFeatures = [];
        for (let i = 0; i < features.length; i++) {
            let featuresTensor = tf.tensor(features[i]);
            if (!!this.means && this.means.length > 0 && this.means[i] && !!this.variances && this.variances.length > 0 && this.variances[i]) {
                featuresTensor = featuresTensor.sub(this.means[i]).div(this.variances[i].pow(0.5));
            }
            else {
                featuresTensor = this.standarize(i, featuresTensor);
            }

            let shape0 = featuresTensor.shape[0];

            let ones = tf.ones([shape0, 1]);
            
            console.log("ones.shape", ones.shape);
            console.log("featuresTensor.shape", featuresTensor.shape);

            let currentFeature = ones.concat(featuresTensor, 1);


            localFeatures.push(currentFeature);
        }
        return localFeatures;
    }

    standarize(index, features) {
        const { mean, variance } = tf.moments(features, 0);
        this.means[index] = mean;
        this.variances[index] = variance;
        return features.sub(mean).div(variance.pow(0.5));
    }

    gradientDescent(features, labels) {
        for (let i = 0; i < features.length; i++) {
            const localFeatures = features[i];
            const currentGuesses = localFeatures.matMul(this.weights[i]).sigmoid();
            const differences = currentGuesses.sub(labels);
            const slopes = localFeatures.transpose().matMul(differences).div(localFeatures.shape[0]);
            this.weights[i] = this.weights[i].sub(slopes.mul(this.options.learningRate));
        }
    }

    train() {
        const { batchSize } = this.options;

        for (let i = 0; i < this.features.length; i++) {
            const localFeatures = this.features[i];
            const batchQuantity = Math.floor(localFeatures.shape[0] / batchSize);    

            for (let j = 0; j < this.options.iterations; j++) {
                for (let k = 0; k < batchQuantity; k++) {
                    const startIndex = k * batchSize;

                    const featureSlice = localFeatures.slice([startIndex, 0], [batchSize, -1]);
                    const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

                    this.gradientDescent(featureSlice, labelSlice);
                }
                this.recordCost();
                this.updateLearningRate();
            }
        }
    }

    test(testFeatures, testLabels) {
        /* TODO Calculate costs */
        testLabels = tf.tensor(testLabels);
        //const predictions = this.predict(testFeatures);
        //const incorrect = predictions.sub(testLabels).abs().sum().get();
        return 0; 
            // (predictions.shape[0] - incorrect) / predictions.shape[0];
    }

    recordCost() {
        for (let i = 0; i < this.features.length; i++) {
            // const localFeature = this.features[i];
            // const localWeights = this.weights[i];

            // const guesses = localFeature.matMul(localWeights).sigmoid();
            // const termOne = this.labels.transpose().matMul(guesses.log());
            // const termTwo = this.labels.mul(-1).add(1).transpose().matMul(guesses.mul(-1).add(1).log());
            // const cost = termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0);

            /* TODO Calculate cost */
            const cost = 0;
            this.costHistory.unshift(cost);    
        }
    }

    updateLearningRate() {
        if (this.costHistory.length < 2) return;

        if (this.costHistory[0] > this.costHistory[1]) {
            this.options.learningRate /= 2;
        }
        else {
            this.options.learningRate *= 1.05;
        }

        this.learningRateHistory.push(this.options.learningRate);
    }

    predict(observations) {
        const result = [];

        for (let i = 0; i < observations.length; i++) {
            const observation = observations[i];
             result[i] = this.processFeatures(observation).
                matMul(this.weights[i]).
                sigmoid().
                greater(this.options.decisionBoundary).
                cast('float32');
        }

        return result;
    }
}

module.exports = LogisticRegression;