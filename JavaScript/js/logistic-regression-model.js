// const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');
const _ = require('lodash');

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
            iterations: 100,
            batchSize: 1000,
            decisionBoundary: 0.5,
            subject: "TestSubject_2",
            useReLu: false,
            shuffle: false,
            normalize: false
        }, options);

        this.costHistory = [];
        this.learningRateHistory = [];

        if (this.options.normalize) {
            baseSamples = this.normalize(baseSamples);
        }

        this.samples = tf.tensor2d(baseSamples);
        this.labels = tf.tensor1d(baseLabels);

        /* Define model compilation settings */
        this.compileSettings = {
            optimizer: 'sgd',
            loss: tf.losses.softmaxCrossEntropy,
            weights: this.weights,
            metrics: [
                tf.metrics.MSE,
                tf.metrics.binaryAccuracy
            ],
        };

        // Defines a simple logistic regression model with 32 dimensional input
        // and 3 dimensional output.
        let inputShape = this.samples.shape[1];
        let outputShape = 1;

        // console.log("inputShape", inputShape);
        // console.log("outputShape", outputShape);

        const x = tf.input({ shape: inputShape });
        const y = tf.layers.dense({ units: outputShape, activation: 'softmax' }).apply(x);
        this.model = tf.model({ 
            inputs: x, 
            outputs: y 
        });
    }

    normalize(samplesToNormalize) {
        let tansposedSamples = [];

        tf.tidy(() => {
            tansposedSamples = tf.tensor(samplesToNormalize).transpose().arraySync();
            for (let i = 0; i < tansposedSamples.length; i++) {
                let currentSample = tansposedSamples[i];
                let temporalTensor = tf.tensor(currentSample);
                const inputMax = temporalTensor.max();
                const inputMin = temporalTensor.min();
                if (inputMax.sub(inputMin).dataSync() != 0) {
                    tansposedSamples[i] = temporalTensor.sub(inputMin).div(inputMax.sub(inputMin)).dataSync();
                }
                else {
                    tansposedSamples[i] = temporalTensor.dataSync();
                }
            }
            tansposedSamples = tf.tensor(tansposedSamples).transpose().arraySync();
        })

        return tansposedSamples;
    }

    async train(trainEndCallback) {
        this.model.compile(this.compileSettings);
        return await this.model.fit(this.samples, this.labels, {
            batchSize: this.options.batchSize,
            epochs: this.options.iterations,
            verbose: false,
            shuffle: this.options.shuffle,
            callbacks: { onTrainEnd: trainEndCallback }
        });
    }

    test(testData, testLabels) {
        console.log("Testing");
        const predictions = [];
        let precision = 0;

        tf.tidy(() => {
            let testDataTensor = tf.tensor(testData);
            let testLabelTensor = tf.tensor(testLabels);
            let result = this.model.evaluate(testDataTensor, testLabelTensor, {
                batchSize: this.options.batchSize
            });

            if (this.options.verbose) {
                let ix = 0;
                console.log("===============================");
                console.log("Samples for test : ", testDataTensor.shape);
                console.log("Labels for test : ", testLabelTensor.shape);
                console.log("===============================");
                console.log("result[" + ix + "] | Loss: ", result[ix++].dataSync());
                console.log("result[" + ix + "] | MSE: ", result[ix++].dataSync());
                console.log("result[" + ix + "] | binaryAccuracy: ", result[ix++].dataSync());

                console.log("model");
                console.log(
                    this.model.layers.
                        map(layer => layer.getWeights().
                            map(weight => "Dimension: [" + weight.shape[0] + ", " + weight.shape[1] + "]")
                        )
                );
            }

            let predictionsValues = [];
            let labelValues = [];

            for (let i = 0; i < testData.length; i++) {
                const dataItem = testData[i];
                // console.log("dataItem", dataItem);
                const dataItemTensor = tf.tensor([dataItem]);
                // const dataItemTensor2 = tf.tensor(dataItem);

                // console.log("dataItemTensor.shape", dataItemTensor.shape);
                // console.log("dataItemTensor2.shape", dataItemTensor2.shape);

                const predictionTensor = this.model.predict(dataItemTensor, { verbose: true });

                if (this.options.verbose) {
                    // console.log("predictionTensor", predictionTensor.dataSync());
                }

                const predictedIndex = predictionTensor.argMax(1).dataSync();
                let prediction = [0, 0, 0, 0];
                prediction[predictedIndex] = 1;

                const expectedLabel = testLabels[i];
                const predictionResult = [prediction, expectedLabel, _.isEqual(prediction, expectedLabel)];

                if (this.options.verbose) {
                    // console.log("expectedLabel", expectedLabel);
                    console.log("prediction", prediction);
                }

                predictions.push(predictionResult);

                predictionsValues.push(prediction);
                labelValues.push(expectedLabel);
            }

            const total = predictions.length;
            const correct = predictions.filter(o => o[2]).length;

            precision = correct / total * 100;
        });

        return precision;
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
    LogisticRegression
};