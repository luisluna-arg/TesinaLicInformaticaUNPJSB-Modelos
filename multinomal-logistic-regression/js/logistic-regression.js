const tf = require('@tensorflow/tfjs-node');
// @tensorflow/tfjs-node
const _ = require('lodash');

class LogisticRegression {
    constructor(baseFeatures, baseLabels, options) {
        if (typeof baseFeatures == 'undefined' || baseFeatures == null || baseFeatures.length == 0) {
            throw 'Coleccion features no valida';
        }

        if (typeof baseLabels == 'undefined' || baseLabels == null || baseLabels.length == 0) {
            throw 'Coleccion labels no valida';
        }

        this.options = Object.assign({
            learningRate: 0.5,
            iterations: 1000,
            batchSize: 50000,
            decisionBoundary: 0.5,
            verbose: false
        }, options);

        baseFeatures = this.normalize(baseFeatures);

        this.features = this.processFeatures(baseFeatures);
        this.labels = tf.tensor(baseLabels);
        this.costHistory = []; // Cross-Entropy values

        this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
    }

    processFeatures(featuresToProcess) {
        let featuresTensor = tf.tensor(featuresToProcess);

        if (!(this.mean && this.variance)) {
            this.standarize(featuresTensor);
        }

        featuresTensor = featuresTensor.sub(this.mean).div(this.variance.pow(0.5));
        featuresTensor = tf.ones([featuresTensor.shape[0], 1]).concat(featuresTensor, 1);

        return featuresTensor;
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

    gradientDescent(featuresToTreat, labelsToTreat) {
        let currentGuesses = featuresToTreat.matMul(this.weights);
        if (!this.options.useReLu) {
            currentGuesses = currentGuesses.softmax();
        }
        else {
            currentGuesses = currentGuesses.relu();
        }

        const differences = currentGuesses.sub(labelsToTreat);
        const slopes = featuresToTreat.transpose().matMul(differences).div(featuresToTreat.shape[0]);
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    recordCost() {
        let currentGuesses = this.features.matMul(this.weights);
        if (!this.options.useReLu) {
            currentGuesses = currentGuesses.softmax();
        }
        else {
            currentGuesses = currentGuesses.relu();
        }
        const termOne = this.labels.transpose().matMul(currentGuesses.log());
        const termTwo = this.labels.mul(-1).add(1).transpose().matMul(currentGuesses.mul(-1).add(1).log());
        const cost = termOne.add(termTwo).div(this.features.shape[0]).mul(-1).dataSync()[0];
        this.costHistory.unshift(cost);
    }

    updateLearningRate() {
        if (this.costHistory.length < 2) return;

        if (this.costHistory[0] > this.costHistory[1]) {
            this.options.learningRate /= 2;
        }
        else {
            this.options.learningRate *= 1.05;
        }
    }

    test(testFeatures, testLabels) {
        let testLabelsTensor = tf.tensor(testLabels);
        let labelIndexTensor = testLabelsTensor.argMax(1);
        const predictionTensor = this.predict(testFeatures);
        const predictionIndexTensor = predictionTensor.argMax(1);

        if (this.options.verbose) {
            console.log("Comparacion resultados");
            console.log("Esperado (ABAJO, ARRIBA, DERECHA, IZQUIERDA)");
            testLabelsTensor.print();
            console.log("Predicho (ABAJO, ARRIBA, DERECHA, IZQUIERDA)");
        }

        if (this.options.verbose) {
            predictionTensor.print();
        }

        const incorrect = predictionIndexTensor.notEqual(labelIndexTensor).sum().dataSync()[0];
        return (predictionIndexTensor.shape[0] - incorrect) / predictionIndexTensor.shape[0];
    }

    normalize(featuresToNormalize) {
        let tansposedFeatures = tf.tensor(featuresToNormalize).transpose().arraySync();
        for (let i = 0; i < tansposedFeatures.length; i++) {
            tansposedFeatures[i] = tf.tensor(tansposedFeatures[i]).log().dataSync();
        }
        tansposedFeatures = tf.tensor(tansposedFeatures).transpose().arraySync();
        return tansposedFeatures;
    }

    standarize(featuresToTreat) {
        const { mean, variance } = tf.moments(featuresToTreat, 0);
        this.mean = mean;
        this.variance = variance;
    }

    predict(featuresToPredict) {
        /* Multiplica las feature por pesos, aplica Softmax para escalar, detecta el valor activado 
        *  y transforma a arreglo de resultado
        */
        let partialResults = this.processFeatures(featuresToPredict).matMul(this.weights);

        let majorValueIndexTensor = (
                this.options.useReLu ? partialResults.relu() : partialResults.softmax()
            ).argMax(1);

        /* Create the label array */
        let resultData = majorValueIndexTensor.dataSync();
        let dataForTensor = [];
        for (let t = 0; t < resultData.length; t++) {
            let result = _.fill(Array(4), 0);
            result[resultData[t]] = 1;
            dataForTensor.push(result);
        }

        return tf.tensor(dataForTensor);
    }

}

module.exports = {
    LogisticRegression
};