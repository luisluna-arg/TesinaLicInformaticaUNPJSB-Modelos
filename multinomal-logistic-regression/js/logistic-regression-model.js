const tf = require('@tensorflow/tfjs');
const { Console } = require('console');
require('@tensorflow/tfjs-node');
const _ = require('lodash');

class LogisticRegression {
    constructor(baseFeatures, baseLabels, options) {
        if (typeof baseFeatures == 'undefined' || baseFeatures == null || baseFeatures.length == 0) {
            throw 'Coleccion features no valida';
        }

        if (typeof baseLabels == 'undefined' || baseLabels == null || baseLabels.length == 0) {
            throw 'Coleccion labels no valida';
        }

        this.features = tf.tensor(baseFeatures);
        this.labels = tf.tensor(baseLabels);

        this.costHistory = [];
        this.learningRateHistory = [];

        /* Set default option settings */
        this.options = Object.assign({
            learningRate: 0.1,
            iterations: 1000,
            batchSize: 50000,
            decisionBoundary: 0.5,
            subject: "TestSubject_2",
            useReLu: false,
            shuffle: false
        }, options);

        /* Define model compilation settings */
        this.compileSettings = {
            optimizer: tf.train.sgd(this.options.learningRate),
            // optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            weights: this.weights,
            metrics: [
                tf.metrics.MSE,
                tf.metrics.binaryAccuracy
            ],
        };

        // const featuresLenght = baseFeatures.length;
        // const labelsLenght = baseLabels.length;

        // console.log("featuresLenght", featuresLenght);
        // console.log("labelsLenght", labelsLenght);

        // console.log("this.features.shape[1]", this.features.shape);
        // console.log("this.labels.shape[1]", this.labels.shape);

        /* Define model layers */
        const classCount = this.labels.shape[1];
        let layers = [
            tf.layers.dense({
                inputShape: this.features.shape[1], units: 256, 
                activation: 'relu'
                // , kernelInitializer: 'zeros'
            }),
            tf.layers.dropout({ rate: 0.25 }),
            tf.layers.dense({ units: 128, activation: 'relu' }),
            tf.layers.dropout({ rate: 0.25 }),
            tf.layers.dense({ units: this.labels.shape[1], activation: 'softmax' }),
        ];

        this.model = tf.sequential({ layers: layers });

        if (this.options.verbose) {
            this.model.summary();
        }
    }

    async train(trainEndCallback) {
        this.model.compile(this.compileSettings);

        return await this.model.fit(this.features, this.labels, {
            batchSize: this.options.batchSize,
            epochs: this.options.iterations,
            verbose: false,
            shuffle: this.options.shuffle,
            callbacks: { onTrainEnd: trainEndCallback }
        });
    }

    test(testData, testLabels) {
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
                console.log("Features for test : ", testDataTensor.shape);
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
                const dataItemTensor = tf.tensor([dataItem]);
                const predictionTensor = this.model.predict(dataItemTensor, { verbose: true });

                const predictedIndex = predictionTensor.argMax(1).dataSync();
                let prediction = [0, 0, 0, 0];
                prediction[predictedIndex] = 1;

                const label = testLabels[i];
                predictions.push([prediction, label, _.isEqual(prediction, label)]);

                predictionsValues.push(prediction);
                labelValues.push(label);
            }

            const total = predictions.length;
            const correct = predictions.filter(o => o[2]).length;

            precision = correct / total * 100;
        });

        return precision;
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