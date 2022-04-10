// const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs');
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

function padSequences(sequence, stepLength = 300, value = 0.0) {
    return sequence.map(function (e) {
        const max_length = stepLength;
        const row_length = e.length
        if (row_length > max_length) { // truncate
            return e.slice(row_length - max_length, row_length)
        }
        else if (row_length < max_length) { // pad
            return Array(max_length - row_length).fill(value).concat(e);
        }
        return e;
    });
}


function expandSamples(samples) {
    let sampleCount = samples.length;
    let featureCount = samples[0].length;
    let result = tf.reshape(samples, [sampleCount, 1, featureCount]);

    // console.log("sampleCount", sampleCount);
    // console.log("featureCount", featureCount);
    // console.log("featureCount.shape", result.shape);

    return result.arraySync();
}

class ConvNetLSTM {

    constructor(trainingSamples, trainingLabels, options) {
        if (typeof trainingSamples == 'undefined' || trainingSamples == null || trainingSamples.length == 0) {
            throw 'Coleccion features no valida';
        }

        if (typeof trainingLabels == 'undefined' || trainingLabels == null || trainingLabels.length == 0) {
            throw 'Coleccion labels no valida';
        }

        /* Set default option settings */
        this.options = Object.assign({
            epochs: 20,
            stepsPerEpoch: 68,
            validationSteps: 2,
            learningRate: 0.005,
            verbose: false
        }, options);

        this.costHistory = [];
        this.learningRateHistory = [];

        let expandedSamples = expandSamples(trainingSamples)
        this.samples = tf.tensor(expandedSamples);
        // this.labels = tf.tensor(labelsToArrays(trainingLabels));
        this.labels = tf.tensor(trainingLabels);

        /* Define model compilation settings */
        this.compileSettings = {
            optimizer: isNullOrUndef(this.options.learningRate) ? 'adam' : tf.train.adam(this.options.learningRate),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        };

        // Defines a simple logistic regression model with 32 dimensional input
        // and 3 dimensional output.
        
        const addLayers = (model, layers) => {
            for (let i = 0; i < layers.length; i++) {
                model.add(layers[i]);
            }
        }

        const dropout = 0.2;
        const recurrentDropout = 0.2;
        const lstmUnits = 128;

        this.model = tf.sequential();

        let lstmLayer = tf.layers.lstm({
            units: lstmUnits, dropout: dropout, 
            recurrentDropout: recurrentDropout, returnSequences: true
        });
        let lstmLayerNoSeq = tf.layers.lstm({
            units: lstmUnits, dropout: dropout, recurrentDropout: recurrentDropout, returnSequences: false
        });

        let conv1dACfg = { filters: 32, kernelSize: 8, strides: 1, activation: "relu", padding: "same" };
        let conv1dBCfg = { filters: 16, kernelSize: 8, strides: 1, activation: "relu", padding: "same" };
        console.log("this.samples.shape", this.samples.shape);

        const inputShape = [ this.samples.shape[1], this.samples.shape[this.samples.shape.length - 1] ];
        const outputShape = [ this.labels.shape[0], this.labels.shape[this.labels.shape.length - 1] ];
        console.log("inputShape", inputShape, 'outputShape', outputShape);

        addLayers(this.model, [
            tf.layers.inputLayer({ inputShape: inputShape }),
            tf.layers.conv1d(conv1dACfg),
            tf.layers.maxPooling1d({ poolSize: 2 }),
            tf.layers.conv1d(conv1dBCfg),
            tf.layers.maxPooling1d({ pool_size: 2 }),
            tf.layers.masking({ maskValue: 0.0 }),
            lstmLayer,
            tf.layers.bidirectional({ layer: lstmLayer }),
            tf.layers.bidirectional({ layer: lstmLayer }),
            tf.layers.bidirectional({ layer: lstmLayerNoSeq }),
            tf.layers.dense({ units: 30, activation: 'relu' }),
            tf.layers.dense({ units: 10, activation: 'relu' }),
            tf.layers.dense({ inputShape: inputShape, units: 1, activation: 'softmax' })
        ]);

    }


    

    async train(trainEndCallback) {
        this.model.compile(this.compileSettings);

        return await this.model.fit(this.samples, this.labels, {
            epochs: this.options.iterations,
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
            let testDataTensor = tf.tensor(expandSamples(testData));
            // let testLabelTensor = tf.tensor(labelsToArrays(testLabels));
            let testLabelTensor = tf.tensor(testLabels);

            console.log("testDataTensor.shape", testDataTensor.shape);
            console.log("testLabelTensor.shape", testLabelTensor.shape);

            let result = this.model.evaluate(testDataTensor, testLabelTensor, {
                epochs: this.options.epochs,
                stepsPerEpoch: this.options.stepsPerEpoch,
                // validationSteps: this.options.validationSteps
            });

            console.log("result", result);

            if (this.options.verbose) {
                let ix = 0;
                // console.log("");
                // console.log("===============================");
                // console.log("Samples for test : ", testDataTensor.shape);
                // console.log("Labels for test : ", testLabelTensor.shape);
                // console.log("===============================");

                // console.log("result[" + 0 + "] | Loss: ", result[0].dataSync());
                for (let x = 1; x < result.length; x++) {
                    let metric = this.compileSettings.metrics[x - 1];
                    // console.log("result[" + x + "] | " + metric + ": ", result[x].dataSync());
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

            let testDataExpanded = testDataTensor.arraySync();
            for (let i = 0; i < testDataExpanded.length; i++) {
                const dataItem = testDataExpanded[i];
                console.log("dataItem", dataItem);

                let prediction = this.predict(dataItem);

                const expectedLabel = testLabels[i];
                const predictionResult = [prediction, expectedLabel, _.isEqual(prediction, expectedLabel)];

                predictions.push(predictionResult);

                predictionsValues.push(prediction);
                labelValues.push(expectedLabel);
            }

            if (this.options.verbose) {
                // console.log(predictions);
            }

            const total = predictions.length;
            const correct = predictions.filter(o => o[2]).length;
            precision = correct / total * 100;
        });

        return precision;
    }

    predict(dataItem) {
        const dataItemTensor = tf.tensor(dataItem);
        const predictionTensor = this.model.predict(dataItemTensor, { verbose: true });
        const predictedIndex = tf.argMax(predictionTensor.dataSync()).dataSync()[0];

        if (this.options.verbose) {
            // // console.log("predictionTensor", predictionTensor.dataSync());
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






    testModel(testData, testLabels) {
        // import tensorflow as tf
        // from tensorflow.keras.datasets import imdb
        // from tensorflow.keras.layers import Embedding, Dense, LSTM
        // from tensorflow.keras.losses import BinaryCrossentropy
        // from tensorflow.keras.models import Sequential
        // from tensorflow.keras.optimizers import Adam
        // from tensorflow.keras.preprocessing.sequence import padSequences

        const addLayers = (model, layers) => {
            for (let i = 0; i < layers.length; i++) {
                model.add(layers[i]);
            }
        };

        // Model configuration
        let additional_metrics = ['accuracy'];
        let batch_size = 128;
        let embedding_output_dims = 15;
        let loss_function = 'binaryCrossentropy';
        let max_sequence_length = 300;
        let num_distinct_words = 5000;
        let number_of_epochs = 5;
        let optimizer = tf.train.adam();
        let validation_split = 0.20;
        let verbosity_mode = 1;

        // console.log("variables");

        // Disable eager execution
        // tf.compat.v1.disable_eager_execution()

        let x_train = this.samples.arraySync();
        let y_train = this.labels.arraySync();

        let x_test = testData;
        let y_test = testLabels;



        // Pad all sequences
        let padded_inputs = tf.tensor(x_train.slice(0, 20)); // padSequences(x_train, max_sequence_length, 0.0) // 0.0 because it corresponds with <PAD>
        let padded_inputs_test = tf.tensor(x_test.slice(0, 20)); // padSequences(x_test, max_sequence_length, 0.0) // 0.0 because it corresponds with <PAD>

        // // console.log("padded_inputs", tf.tensor(padded_inputs).shape);
        // console.log("padded_inputs", padded_inputs[padded_inputs.length - 1]);

        // Define the Keras model
        let model = tf.sequential()

        addLayers(model, [
            tf.layers.embedding({
                inputDim: num_distinct_words,
                outputDim: embedding_output_dims,
                input_length: max_sequence_length
            }),
            tf.layers.lstm({ units: 10 }),
            tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]);

        // Compile the model
        model.compile({
            optimizer: optimizer,
            loss: loss_function,
            metrics: additional_metrics,
            verbose: 0, //verbosity_mode,
        })

        // // Give a summary
        // // model.summary()

        // Train the model
        let history = model.fit(
            padded_inputs,
            y_train,
            {
                batch_size: batch_size,
                epochs: number_of_epochs,
                verbose: false, //verbosity_mode,
                // validation_split: validation_split
            })

        // // Test the model after training
        // let test_results = model.evaluate(padded_inputs_test, y_test, { verbose: false })
        // let loss = test_results[0];
        // let accuracy = 100 * test_results[1];

        // // console.log('Test results - Loss: ' + loss + ' - Accuracy: ' + accuracy + '%')


    }





}


module.exports = {
    model: ConvNetLSTM
};