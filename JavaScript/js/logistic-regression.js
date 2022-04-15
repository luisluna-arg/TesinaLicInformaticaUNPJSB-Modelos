const tf = require('@tensorflow/tfjs-node');
const _ = require('lodash');

/* ******************************************************** */
/* ******************************************************** */
/* ******************************************************** */

const labelCount = 4;

function buildLabelArray(item) {
    let result = new Array(4);
    result.fill(0);
    result[item - 1] = 1;
    return result;
}

function labelMapper(item) {
    if (!Array.isArray(item)) return buildLabelArray(item);

    if (item.length === 1) return buildLabelArray(item[0])

    return item;
}

/* ******************************************************** */
/* ******************************************************** */
/* ******************************************************** */

class LogisticRegression {
    constructor(baseSamples, baseLabels, options) {
        if (typeof baseSamples == 'undefined' || baseSamples == null || baseSamples.length == 0) {
            throw 'Coleccion de muestras no valida';
        }

        if (typeof baseLabels == 'undefined' || baseLabels == null || baseLabels.length == 0) {
            throw 'Coleccion labels no valida';
        }

        this.options = Object.assign({
            learningRate: 0.5,
            iterations: 100,
            batchSize: 5000,
            verbose: false
        }, options);

        this.samples = this.processSamples(baseSamples);
        this.labels = tf.tensor(baseLabels.map(labelMapper));

        this.costHistory = []; // Cross-Entropy values
        this.weights = tf.ones([this.samples.shape[1], this.labels.shape[1]]);
    }

    processSamples(samplesToProcess) {
        let samplesTensor = tf.tensor(samplesToProcess);

        if (!(this.mean && this.variance)) {
            this.setMeanAndVariance(samplesTensor);
        }
        samplesTensor = this.standarize(samplesTensor);
        return samplesTensor;
    }

    standarize(samplesTensor) {
        let currentSamples = samplesTensor.sub(this.mean).div(this.variance.pow(0.5));
        return tf.ones([currentSamples.shape[0], 1]).concat(currentSamples, 1);
    }

    train() {
        let batchCount = Math.floor(this.samples.shape[0] / this.options.batchSize);
        if (batchCount == 0) batchCount = 1;

        let localBatchSize = this.options.batchSize > this.samples.shape[0] ? this.samples.shape[0] : this.options.batchSize;

        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchCount; j++) {
                const startIndex = j * localBatchSize;

                let featureSlice = this.samples.slice([startIndex, 0], [localBatchSize, -1]);
                let labelSlice = this.labels.slice([startIndex, 0], [localBatchSize, -1]);

                this.gradientDescent(featureSlice, labelSlice);

                featureSlice.dispose();
                labelSlice.dispose();
            }
            this.recordCost();
            this.updateLearningRate();
        }
    }

    gradientDescent(samplesToTreat, labelsToTreat) {
        let currentGuesses = samplesToTreat.matMul(this.weights);
        if (!this.options.useReLu) {
            currentGuesses = currentGuesses.softmax();
        }
        else {
            currentGuesses = currentGuesses.relu();
        }

        const differences = currentGuesses.sub(labelsToTreat);
        const slopes = samplesToTreat.transpose().matMul(differences).div(samplesToTreat.shape[0]);
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));

        currentGuesses.dispose();
        differences.dispose();
        slopes.dispose();
    }

    recordCost() {
        let currentGuesses = this.samples.matMul(this.weights);
        if (!this.options.useReLu) {
            currentGuesses = currentGuesses.softmax();
        }
        else {
            currentGuesses = currentGuesses.relu();
        }
        const termOne = this.labels.transpose().matMul(currentGuesses.log());
        const termTwo = this.labels.mul(-1).add(1).transpose().matMul(currentGuesses.mul(-1).add(1).log());
        const termThree = termOne.add(termTwo).div(this.samples.shape[0]).mul(-1);
        this.costHistory.unshift(termThree.dataSync()[0]);

        currentGuesses.dispose();
        termOne.dispose();
        termTwo.dispose();
        termThree.dispose();
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
        let testLabelsTensor = tf.tensor(testLabels.map(item => item[0]));
        const labelTensor = this.predict(testFeatures);
        const predictions = labelTensor.arraySync();
        
        // const incorrect = testLabelsTensor.notEqual(labelTensor).sum().arraySync();
        const predictionCount = predictions.length;
        let correctCount = 0;
        if (this.options.verbose) {
            console.log("Comparacion resultados");
            let comparison = testLabels.map((currentValue, index) => {
                let singleValue = Array.isArray(currentValue) && currentValue.length == 1 ? currentValue[0] : currentValue;
                let predictionValue = predictions[index];
                const isCorrect = currentValue == predictionValue;
                correctCount += isCorrect ? 1 : 0;
                return [singleValue, predictionValue, isCorrect];
            });
            console.log(comparison);
        }

        console.log("correct", correctCount, "/", predictionCount);
        return { precision: correctCount / predictionCount * 100 };
    }

    normalize(samplesToNormalize) {
        let tansposedFeatures = tf.tensor(samplesToNormalize).transpose().arraySync();
        for (let i = 0; i < tansposedFeatures.length; i++) {
            let currentSample = tansposedFeatures[i];

            let temporalTensor = tf.tensor(currentSample);

            const inputMax = temporalTensor.max();
            const inputMin = temporalTensor.min();
            if (inputMax.sub(inputMin).dataSync() != 0) {
                tansposedFeatures[i] = temporalTensor.sub(inputMin).div(inputMax.sub(inputMin)).dataSync();
            }
            else {
                tansposedFeatures[i] = temporalTensor.dataSync();
            }

        }
        tansposedFeatures = tf.tensor(tansposedFeatures).transpose().arraySync();

        return tansposedFeatures;
    }

    normalize2(samplesToNormalize) {
        let tansposedFeatures = tf.tensor(samplesToNormalize).transpose().arraySync();
        for (let i = 0; i < tansposedFeatures.length; i++) {
            tansposedFeatures[i] = tf.tensor(tansposedFeatures[i]).log().dataSync();
        }
        tansposedFeatures = tf.tensor(tansposedFeatures).transpose().arraySync();
        return tansposedFeatures;
    }

    setMeanAndVariance(samplesToTreat) {
        const { mean, variance } = tf.moments(samplesToTreat, 0);
        this.mean = mean;
        this.variance = variance;
    }

    predict(samplesToPredict) {
        /* 
         * Multiplica las feature por pesos, aplica Softmax para 
         * escalar, detecta el valor activado y transforma a arreglo 
         * de resultado
         */
        let processedSamples = this.processSamples(samplesToPredict);
        let weightedSamples = processedSamples.matMul(this.weights);
        let partialResults = (this.options.useReLu ? weightedSamples.relu() : weightedSamples.softmax());
        let majorValueIndexTensor = partialResults.argMax(1);

        return tf.tensor(majorValueIndexTensor.arraySync().map(labelIndex => labelIndex + 1));
    }

    summary() {
        // console.log("costHistory", this.costHistory);
        // console.log("weights", this.weights.arraySync());
        console.log("mean", this.mean.dataSync());
        console.log("variance", this.variance.dataSync());
    }

}

module.exports = {
    model: LogisticRegression
};