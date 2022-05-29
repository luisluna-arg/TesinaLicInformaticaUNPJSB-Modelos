// REFERENCES FOR THIS EXAMPLE
// Models and layers DOC: https://www.tensorflow.org/js/guide/models_and_layers
// Train models DOC: https://www.tensorflow.org/js/guide/train_models


// node --inspect-brk example-model.js // Breaks on the first line of code
// node --inspect example-model.js

const tf = require('@tensorflow/tfjs');

// Generate dummy data.
function* data() {
    for (let i = 0; i < 100; i++) {
        // Generate one sample at a time.
        yield tf.randomNormal([784]);
    }
}

function* labels() {
    for (let i = 0; i < 100; i++) {
        // Generate one sample at a time.
        yield tf.randomUniform([10]);
    }
}

// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function _model(x) {
    return x.matMul(w1).add(b1).relu().matMul(w2).add(b2);
}

const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);

console.log('xs', xs);
console.log('ys', ys);

// Zip the data and labels together, shuffle and batch 32 samples at a time.
const bufferSize = 100;
const ds = tf.data.zip({ xs, ys }).shuffle(bufferSize).batch(32);

const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate); /* learningRate */
// Train for 5 epochs.
const epochCount = 5;
for (let epoch = 0; epoch < epochCount; epoch++) {
    /* await */
    ds.forEachAsync(({ xs, ys }) => {
        optimizer.minimize(() => {
            const predYs = _model(xs);
            const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
            loss.data().then(l => console.log('Loss', l));
            return loss;
        });
    });
    console.log('Epoch', epoch);
}