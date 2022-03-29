// const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

const isNullOrUndef = (value) => typeof value === 'undefined' || null;

const expandSamples = (samples) => {
    let result = null;

    tf.tidy(() => {
        let samplesTF = tf.tensor(samples);
        result = tf.reshape(samplesTF, [samplesTF.shape[0], samplesTF.shape[1], 1]).arraySync();
    });

    return result;
};

const addLayers = (model, layers) => {
    for (let i = 0; i < layers.length; i++) {
        model.add(layers[i]);
    }
}

// define model for simple BI-LSTM + DNN based binary classifier
function defineModel() {
    // input1 = tf.layers.Input(shape=(2,1)) #take the reshape last two values, see "data = np.reshape(data,(10,2,1))" which is "data/batch-size, row, column"
    // take the reshape last two values, see "data = np.reshape(data,(10,2,1))" which is "data/batch-size, row, column"

    // input1 = tf.layers.inputLayer({ inputShape: [8, 1] });
    // lstm1 = tf.layers.bidirectional({ layer: tf.layers.lstm({ units: 32 }) })(input1)
    // dnn_hidden_layer1 = tf.layers.dense({ units: 3, activation: 'relu' })(lstm1)
    // dnn_output = tf.layers.dense({ units: 1, activation: 'sigmoid' })(dnn_hidden_layer1)
    
    // model = tf.model({ inputs: input1, outputs: dnn_output });
    
    // // compile the model
    // model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    // model.summary()


    model = tf.sequential();
    addLayers(model, [
        tf.layers.inputLayer({ inputShape: [8, 1] }),
        tf.layers.bidirectional({ layer: tf.layers.lstm({ units: 32 }) }),
        tf.layers.dense({ units: 3, activation: 'relu' }),
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    ]);

    // compile the model
    model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['accuracy'] })
    model.summary()
    return model
}

/* ****************************************************** */
/* ****************************************************** */
/* ****************************************************** */

class NetLSTM {

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

        // Reshape the data into 3-D numpy array
        // data = np.reshape(data,(10,2,1)) #Here we have a total of 10 rows or records
        this.samples = tf.tensor(expandSamples(trainingSamples)); // Here we have a total of 10 rows or records
        this.labels = tf.tensor(trainingLabels);

        /* Define model compilation settings */
        this.compileSettings = {
            optimizer: isNullOrUndef(this.options.learningRate) ? 'adam' : tf.train.adam(this.options.learningRate),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        };

        // Call the model
        this.model = defineModel()
    }

    async train(trainEndCallback) {
        this.model.compile(this.compileSettings);

        return this.model.fit(data, Y, {
            epochs: this.options.epochs,
            batchSize: this.options.batchSize,
            verbose: this.options.verbose,
            callbacks: { onTrainEnd: trainEndCallback }
        });
    }

    test(testData, testLabels) {
        const predictions = [];
        let precision = 0;

        tf.tidy(() => {
            let testDataTensor = tf.tensor(expandSamples(testData));
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

}


module.exports = {
    model: NetLSTM
};