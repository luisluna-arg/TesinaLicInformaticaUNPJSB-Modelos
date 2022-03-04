const tf = require('@tensorflow/tfjs-node');
const _ = require('lodash');

class ConvNet_LSTM_1DConv {
    constructor(baseSamples, baseLabels, options) {
        if (typeof baseSamples == 'undefined' || baseSamples == null || baseSamples.length == 0) {
            throw 'Coleccion de muestras no valida';
        }

        if (typeof baseLabels == 'undefined' || baseLabels == null || baseLabels.length == 0) {
            throw 'Coleccion labels no valida';
        }

        let self = this;

        self.options = Object.assign({
            verbose: false
        }, options);

        self.compileSettings = {
            optimizer: tf.train.adam(self.options.learningRate),
            loss: tf.losses.sigmoidCrossEntropy,
            weights: self.weights,
            metrics: [tf.metrics.binaryAccuracy]
        };

        baseSamples = this.normalize(baseSamples);

        self.samples = tf.tensor(baseSamples);
        self.labels = tf.tensor(baseLabels);

        const filterCount = 32;
        const kernelSize = [3, 3];
        const inputShape = [32, 32, 3];
        self.n_classes = self.labels.shape[1];

        self.model = tf.sequential([
            tf.layers.conv2d({ filters: filterCount, kernelSize: kernelSize, activation: 'relu', padding: 'same', }),
            tf.layers.conv2d({ filters: filterCount, kernelSize: kernelSize, activation: 'relu', padding: 'same', inputShape: inputShape }),
            tf.layers.conv2d({ filters: filterCount, kernelSize: kernelSize, activation: 'relu', padding: 'same' }),
            tf.layers.maxPooling2d({ pool_size: [2, 2], strides: 2 }),
            tf.layers.flatten(),
            tf.layers.dropout(0.5),
            tf.layers.dense({ units: 512, activation: 'relu' }),
            tf.layers.dropout(0.5),
            tf.layers.dense({ units: self.n_classes, activation: 'softmax' })
        ]);
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

    reformat_input(data, labels, indices) {
        tf.util.shuffle(indices[0]);
        tf.util.shuffle(indices[0]);
        train_indices = indices[0].slice(0, len(indices[1]));
        valid_indices = indices[0].slice(len(indices[1]));
        test_indices = indices[1]
        return [
            (data[train_indices], np.squeeze(labels[train_indices]).astype(np.int32)),
            (data[valid_indices], np.squeeze(labels[valid_indices]).astype(np.int32)),
            (data[test_indices], np.squeeze(labels[test_indices]).astype(np.int32))
        ];
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
}