const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {

    constructor(data, options) {
        // console.log("Creation");
        this.options = Object.assign({
            learningRate: 0.1,
            iterations: 1000,
            batchSize: 50000,
            decisionBoundary: 0.5,
            shuffle: true,
            tolerance: 0.5
        }, options);

        this.process(data);

        const classCount = this.labels.shape[1];

        this.model = tf.sequential({
            layers: [
                // Add a single input layer
                // tf.layers.dense({
                //     inputShape: this.features.shape[1],
                //     units: classCount,
                //     useBias: true,
                //     // activation: 'relu'
                // }),
                tf.layers.dense({
                    inputShape: this.features.shape[1],
                    units: classCount,
                    useBias: true,
                    activation: "softmax"
                }),
                // Add an output layer
                tf.layers.dense({
                    units: classCount,
                    useBias: true
                })
            ]
        });

        this.model.summary();
    }

    async train() {
        this.model.compile({
            optimizer: tf.train.sgd(this.options.learningRate),
            loss: tf.losses.meanSquaredError,
            weights: this.weights,
            metrics: [
                tf.metrics.MSE,
                tf.metrics.precision,
                tf.metrics.binaryAccuracy
            ],
        });

        return await this.model.fit(this.features, this.labels, {
            batchSize: this.options.batchSize,
            epochs: this.options.iterations,
            validationSplit: 0.2, /* training it will validate the quality of the training */
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    /* accion a ejecutar al procesar una iteracion */
                }
            }
        });
    }

    test(inputData) {
        const tensorData = this.convertToTensor(inputData);

        const testFeatures = tensorData.features;
        const testLabels = tensorData.labels;

        let result = this.model.evaluate(testFeatures, testLabels, { batchSize: this.options.batchSize });

        let ix = 0;
        // console.log("result[0] | Loss: ", result[ix++].dataSync());
        // console.log("result[1] | MSE: ", result[ix++].dataSync());
        // console.log("result[1] | Precision: ", result[ix++].dataSync());
        // console.log("result[1] | binaryAccuracy: ", result[ix++].dataSync());

        const testFormattedData = this.splitData(inputData);

        const predictions = [];

        tf.tidy(() => {
            for (let i = 0; i < testFormattedData.features.length; i++) {
                const dataItem = testFormattedData.features[i];
                const predicted = this.model.predict(tf.tensor([dataItem])).argMax(0).dataSync();
                const label = testFormattedData.labels[i];

                predictions.push([predicted, label]);
            }
        });

        for (let i = 0; i < predictions.length; i++) {
            // console.log(predictions[i]);
        }

        this.model.summary();

        this.model.history
        const weights = this.model.getWeights().map(o => o.dataSync());
        // console.log("weights", weights);
        // console.log("features", this.features.dataSync().length);

        let properties = '';
        for (let property in this.model) {
            properties += property + ', '
        }

        // console.log("properties", properties);
        // console.log("layers", this.model.layers);
        // console.log("history", this.model.history);
    }

    /**
     * Aleatoriza los datos y los separa en arreglos de muestras y etiquetas
     * Tambien ajusta las etiquetas de acuerdo al nivel de tolerancia fijado en options
     * @param {*} dataToSplit 
     * @param {*} shuffle 
     * @returns 
     */
    splitData(dataToSplit, shuffle = true) {
        // Step 1. Shuffle the data
        if (shuffle) { tf.util.shuffle(dataToSplit); }

        // Step 2. Split into features and labels
        const features = [];
        const labels = [];

        tf.tidy(() => {
            const limits = {
                minattention: 0, maxattention: 0,
                minmeditation: 0, maxmeditation: 0,
                mindelta: 0, maxdelta: 0,
                mintheta: 0, maxtheta: 0,
                minlowAlpha: 0, maxlowAlpha: 0,
                minhighAlpha: 0, maxhighAlpha: 0,
                minlowBeta: 0, maxlowBeta: 0,
                minhighBeta: 0, maxhighBeta: 0,
                minlowGamma: 0, maxlowGamma: 0,
                minhighGamma: 0, maxhighGamma: 0,
            }

            for (let i = 0; i < dataToSplit.length; i++) {
                const element = dataToSplit[i];
                for (let property in element) {
                    if (limits.hasOwnProperty("min" + property) && limits["min" + property] > element[property])
                        limits["min" + property] = element[property];
                    if (limits.hasOwnProperty("max" + property) && limits["max" + property] < element[property])
                        limits["max" + property] = element[property];
                }
            }

            _.each(dataToSplit, (dataItem) => {
                let regularData = [
                    dataItem.attention,
                    dataItem.meditation,
                    dataItem.delta,
                    dataItem.theta,
                    dataItem.lowAlpha,
                    dataItem.highAlpha,
                    dataItem.lowBeta,
                    dataItem.highBeta,
                    dataItem.lowGamma,
                    dataItem.highGamma
                ];

                /* Se normaliza cada variable de medicion en un rango de 0 a 1, 
                * respecto de su maximo valor observado 
                */
                let normalizedData = [
                    dataItem.attention / limits.maxattention,
                    dataItem.meditation / limits.maxmeditation,
                    dataItem.delta / limits.maxdelta,
                    dataItem.theta / limits.maxtheta,
                    dataItem.lowAlpha / limits.maxlowAlpha,
                    dataItem.highAlpha / limits.maxhighAlpha,
                    dataItem.lowBeta / limits.maxlowBeta,
                    dataItem.highBeta / limits.maxhighBeta,
                    dataItem.lowGamma / limits.maxlowGamma,
                    dataItem.highGamma / limits.maxhighGamma,
                ];

                /* 
                * Si el valor observado supero la tolerancia, se lo considera una activacion, y se aceptan sus labels
                * Si no se supera esa tolerancia minima, se desactivan las labels (array de 0)
                */
                let found = normalizedData.filter((o) => o > this.options.tolerance).length;

                if (found) {
                    /* features.push(regularData); */
                    features.push(normalizedData);
                    labels.push([dataItem.down, dataItem.up, dataItem.right, dataItem.left]);
                }
            });
        });

        return {
            features,
            labels
        };
    }

    process(data) {
        const tensorData = this.convertToTensor(data);
        this.features = tensorData.features;
        this.labels = tensorData.labels;
        this.weights = tf.zeros([this.features.shape[0], 1])

        // console.log("process.features", this.features.shape);
        // console.log("process.weights", this.weights.shape);

        // Return the min/max bounds so we can use them later.
        this.inputMax = tensorData.inputMax;
        this.inputMin = tensorData.inputMax;
        this.labelMax = tensorData.inputMax;
        this.labelMin = tensorData.inputMax;
    }

    /**
     * Convertir las muestras en tensores
     * Previamente, se extraen las etiquetas de las muestras y ponen ena arreglos separados
     * La tarea aleatoriza los datos tambien
     */
    convertToTensor(data) {
        const splittedData = this.splitData(data);

        /* Tidy limpia cualquier tensor intermediario de memoria */
        return tf.tidy(() => {
            const featuresTensor = tf.tensor(splittedData.features);
            const labelsTensor = tf.tensor(splittedData.labels);

            return {
                features: featuresTensor,
                labels: labelsTensor,
                // Return the min/max bounds so we can use them later.
                inputMax: featuresTensor.max(),
                inputMin: featuresTensor.min(),
                labelMax: labelsTensor.max(),
                labelMin: labelsTensor.min()
            }

        });
    }
}

module.exports = {
    LogisticRegression
};
