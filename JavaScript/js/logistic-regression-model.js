// const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');
const _ = require('lodash');

/* ******************************************************** */

function buildLabelArray(item) {
    let result = new Array(4);
    result.fill(0);
    result[item - 1] = 1;
    return result;
}

function labelMapper(item) {
    return !Array.isArray(item) ? buildLabelArray(item) : item;
}

const addLayers = (model, layers) => {
    for (let i = 0; i < layers.length; i++) {
        model.add(layers[i]);
    }
};

const formatFloat = (value, decimals = 8) => parseFloat(value.toFixed(decimals));

const formatFloatArray = (value, decimals = 8) => parseFloat(value.dataSync()[0].toFixed(decimals));

/* ******************************************************** */

class LogisticRegression {

    constructor(baseSamples, baseLabels, options) {
        if (typeof baseSamples == 'undefined' || baseSamples == null || baseSamples.length == 0) {
            throw 'Coleccion de muestras no valida';
        }

        if (typeof baseLabels == 'undefined' || baseLabels == null || baseLabels.length == 0) {
            throw 'Coleccion labels no valida';
        }

        /* Set default option settings */
        this.options = Object.assign({
            learningRate: 0.1,
            epochs: 100,
            batchSize: 1000,
            decisionBoundary: 0.5,
            subject: "TestSubject_2",
            useReLu: false,
            shuffle: false
        }, options);

        this.costHistory = [];
        this.learningRateHistory = [];

        this.samples = tf.tensor(baseSamples);
        this.labels = tf.tensor(baseLabels.map(labelMapper));

        const { mean, variance } = tf.moments(this.samples, 0);
        this.mean = mean;
        this.variance = variance;

        /* Define model compilation settings */
        this.compileSettings = {
            optimizer: tf.train.sgd(this.options.learningRate),
            loss: "meanSquaredError",
            metrics: ["accuracy"]
        };

        this.model = tf.sequential();

        const units = this.labels.shape != null && this.labels.shape.length > 1 ? this.labels.shape[1] : 1;
        const inputShape = [this.samples.shape[1]]

        addLayers(this.model, [
            tf.layers.dense({ inputShape, units, activation: this.options.useReLu ? 'relu' : 'softmax' })
        ]);

    }

    async train(trainEndCallback) {
        this.model.compile(this.compileSettings);

        return await this.model.fit(this.samples, this.labels, {
            batchSize: this.options.batchSize, // 250
            epochs: this.options.epochs, // 500
            verbose: this.options.verbose,
            callbacks: { onTrainEnd: trainEndCallback }
        });
    }

    test(testData, testLabels) {
        const predictions = [];
        let finalResult = null;

        tf.tidy(() => {
            let testDataTensor = tf.tensor(testData);
            const mappedTestLabels = testLabels.map(labelMapper);
            let testLabelTensor = tf.tensor(mappedTestLabels);

            let result = this.model.evaluate(testDataTensor, testLabelTensor, {
                batchSize: this.options.batchSize
            });

            if (this.options.verbose) {
                console.log("");
                console.log("===============================");
                console.log("Samples for test : ", testDataTensor.shape);
                console.log("Labels for test : ", testLabelTensor.shape);
                console.log("===============================");

                console.log("result[" + 0 + "] | Loss: ", formatFloatArray(result[0]));
                for (let x = 1; x < result.length; x++) {
                    let metric = this.compileSettings.metrics[x - 1];
                    console.log("result[" + x + "] | " + metric + ": ", formatFloatArray(result[x]));
                }

                console.log("");
                console.log("Capas y pesos");
                for (let i = 0; i < this.model.layers.length; i++) {
                    let layer = this.model.layers[i];
                    let weights = layer.getWeights();
                    for (let j = 0; j < weights.length; j++) {
                        let weight = weights[j];
                        console.log("Dimension: [" + weight.shape[0] + ", " + weight.shape[1] + "]");
                    }
                }
            }

            let predictionsValues = [];
            let labelValues = [];

            for (let i = 0; i < testData.length; i++) {
                const dataItem = testData[i];
                const expectedLabel = mappedTestLabels[i];
                const prediction = buildLabelArray(this.predict(dataItem));

                const predictionResult = [prediction, expectedLabel, _.isEqual(prediction, expectedLabel)];

                predictions.push(predictionResult);

                predictionsValues.push(prediction);
                labelValues.push(expectedLabel);
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
        });

        return finalResult;
    }

    predict(dataItem) {
        const predictionTensor = this.model.predict(tf.tensor([dataItem]), { verbose: true });
        const predictedIndex = predictionTensor.argMax(1).dataSync()[0];
        let prediction = predictedIndex + 1;
        return prediction;
    }

    summary() {
        this.model.summary();
    }

    async export() {
        let modelPath = module.parent.path.replace("C:\\", "");
        modelPath += "\\" + "trained-models";
        modelPath += "\\" + this.options.subject;
        modelPath = "file:///" + modelPath;

        await this.model.save(modelPath);
    }

}


module.exports = {
    model: LogisticRegression
};