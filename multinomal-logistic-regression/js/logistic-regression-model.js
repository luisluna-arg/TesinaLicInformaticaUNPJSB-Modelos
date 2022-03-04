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
            shuffle: false
        }, options);

        this.costHistory = [];
        this.learningRateHistory = [];

        baseSamples = this.normalize(baseSamples);

        this.samples = tf.tensor(baseSamples);
        this.labels = tf.tensor(baseLabels);

        this.weights = tf.zeros([this.samples.shape[1], this.labels.shape[1]]);

        // console.log(tf.losses);

        /* Define model compilation settings */
        this.compileSettings = {
            // optimizer: tf.train.adam(this.options.learningRate),
            // optimizer: 'adam',
            optimizer: 'sgd',
            // loss: tf.losses.meanSquaredError,
            loss: tf.losses.softmaxCrossEntropy,
            weights: this.weights,
            metrics: [
                tf.metrics.MSE,
                tf.metrics.binaryAccuracy
            ],
        };

        // optimizer='sgd',loss='categorical_crossentropy', metrics=['acc', 'mse']

        /* Define model layers */
        // Define a model for linear regression.
        // this.model = tf.sequential();
        // this.model.add(tf.layers.dense({ 
        //     inputShape: this.samples.shape[1], 
        //     units: Math.floor(512 / 2), activation: 'tanh'
        // }));
        // this.model.add(tf.layers.batchNormalization());
        // this.model.add(tf.layers.dense({ 
        //     units: Math.floor(512 / 4), activation: 'tanh' 
        // }));
        // this.model.add(tf.layers.batchNormalization());
        // this.model.add(tf.layers.dense({ 
        //     units: Math.floor(512 / 6), activation: 'tanh'
        // }));
        // this.model.add(tf.layers.batchNormalization());
        // this.model.add(tf.layers.dense({ 
        //     units: 32, activation: 'relu' 
        // }));
        // this.model.add(tf.layers.batchNormalization());
        // this.model.add(tf.layers.dense({ 
        //     units: this.labels.shape[1]
        // }));

        let inputShape = this.samples.shape[1];
        this.model = tf.sequential();
        
        this.model.add(tf.layers.dense({ units: 512, activation: 'tanh', inputShape: inputShape }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dense({ units: Math.floor(512/2), activation: 'tanh' }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dense({ units: Math.floor(512/4), activation: 'tanh' }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dense({ units: Math.floor(512/8), activation: 'tanh' }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dense({ units: this.labels.shape[1], activation: 'softmax' }));



        // this.model.add(tf.layers.dense({ units: this.labels.shape[1] }));
        // this.model.add(tf.layers.activation({ activation: 'softmax' }));
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
        let tansposedSamples = tf.tensor(samplesToNormalize).transpose().arraySync();
        for (let i = 0; i < tansposedSamples.length; i++) {
            tansposedSamples[i] = tf.tensor(tansposedSamples[i]).log().dataSync();
        }
        tansposedSamples = tf.tensor(tansposedSamples).transpose().arraySync();

        return tansposedSamples;
    }

    async train(trainEndCallback) {
        this.model.compile(this.compileSettings);

        console.log("this.samples.shape", this.samples.shape);
        console.log("this.labels.shape", this.labels.shape);
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