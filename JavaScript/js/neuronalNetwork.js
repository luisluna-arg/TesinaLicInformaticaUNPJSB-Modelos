// const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');
const _ = require('lodash');

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

function isNullOrUndef(value) {
    return typeof value === 'undefined' || null;
}

function arrayToLabel(labelArray) {
    if (!Array.isArray(labelArray))
        throw "Se esperaba un arreglo en representación de una etiqueta"

    return labelArray.indexOf(1) + 1;
}

function labelsToArrays(labels) {
    if (!Array.isArray(labels))
        throw "Se esperaba un arreglo de etiquetas"

    return labels.map(o => {
        let result = new Array(4);
        result.fill(0);
        result[o - 1] = 1;
        return result;
    });
}

class NeuronalNetwork {

    constructor(trainingSamples, trainingLabels, options) {
        if (typeof trainingSamples == 'undefined' || trainingSamples == null || trainingSamples.length == 0) {
            throw 'Coleccion features no valida';
        }

        if (typeof trainingLabels == 'undefined' || trainingLabels == null || trainingLabels.length == 0) {
            throw 'Coleccion labels no valida';
        }

        /* Set default option settings */
        this.options = Object.assign({
            epochs: 10,
            stepsPerEpoch: 500,
            validationSteps: 2,
            learningRate: null,
            verbose: false
        }, options);

        this.costHistory = [];
        this.learningRateHistory = [];

        this.samples = tf.tensor2d(trainingSamples);
        this.labels = tf.tensor(labelsToArrays(trainingLabels));

        /* Define model compilation settings */
        this.compileSettings = {
            optimizer: isNullOrUndef(this.options.learningRate) ? 'adam' : tf.train.adam(this.options.learningRate),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        };

        // Defines a simple logistic regression model with 32 dimensional input
        // and 3 dimensional output.
        let inputShape = this.samples.shape[1];
        let outputShape = this.labels.shape.length == 1 ? 1 : this.labels.shape[1];

        const addLayers = (model, layers) => {
            for (let i = 0; i < layers.length; i++) {
                model.add(layers[i]);
            }
        }

        this.model = tf.sequential();
        addLayers(this.model, [
            tf.layers.dense({ inputShape: inputShape, units: 256, activation: 'relu' }),
            tf.layers.dense({ units: 192, activation: 'relu' }),
            tf.layers.dense({ units: 128, activation: 'relu' }),
            tf.layers.dense({ units: outputShape, activation: 'softmax' })
        ]);
    }

    async train(trainEndCallback) {
        this.model.compile(this.compileSettings);

        return await this.model.fit(this.samples, this.labels, {
            epochs: this.options.epochs,
            stepsPerEpoch: this.options.stepsPerEpoch,
            validationSteps: this.options.validationSteps,
            verbose: this.options.verbose,
            // validation_data=val_dataset.repeat(),
            callbacks: { onTrainEnd: trainEndCallback }
        });
    }

    test(testData, testLabels) {
        const predictions = [];
        let precision = 0;

        tf.tidy(() => {
            let testDataTensor = tf.tensor(testData);
            let testLabelTensor = tf.tensor(labelsToArrays(testLabels));
            let result = this.model.evaluate(testDataTensor, testLabelTensor, {
                epochs: this.options.epochs,
                stepsPerEpoch: this.options.stepsPerEpoch,
                validationSteps: this.options.validationSteps
            });

            if (this.options.verbose) {
                let ix = 0;
                console.log("");
                console.log("===============================");
                console.log("Samples for test : ", testDataTensor.shape);
                console.log("Labels for test : ", testLabelTensor.shape);
                console.log("===============================");

                console.log("result[" + 0 + "] | Loss: ", result[0].dataSync());
                for (let x = 1; x < result.length; x++) {
                    let metric = this.compileSettings.metrics[x - 1];
                    console.log("result[" + x + "] | " + metric + ": ", result[x].dataSync());
                }

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

                let prediction = this.predict(dataItem);

                const expectedLabel = testLabels[i][0];
                const predictionResult = [prediction, expectedLabel, _.isEqual(prediction, expectedLabel)];

                predictions.push(predictionResult);

                predictionsValues.push(prediction);
                labelValues.push(expectedLabel);
            }

            if (this.options.verbose) {
                // for(let i = 0; i < predictions.length; i++) {
                //     console.log("predictions[" + i + "]", predictions[i]);
                // }
            }

            const total = predictions.length;
            const correct = predictions.filter(o => o[2]).length;
            precision = correct / total * 100;
        });

        return { precision };
    }

    predict(dataItem) {
        const dataItemTensor = tf.tensor([dataItem]);

        const predictionTensor = this.model.predict(dataItemTensor, { verbose: true });
        const predictedIndex = tf.argMax(predictionTensor.dataSync()).dataSync()[0];

        if (this.options.verbose) {
            // console.log("predictionTensor", predictionTensor.dataSync());
        }


        let prediction = new Array(4);
        prediction.fill(0);
        prediction[predictedIndex] = 1;

        return arrayToLabel(prediction);
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




    labelsToArrays


}


module.exports = {
    model: NeuronalNetwork
};