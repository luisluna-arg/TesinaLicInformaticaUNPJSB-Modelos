// REFERENCES FOR THIS EXAMPLE
// Models and layers DOC: https://www.tensorflow.org/js/guide/models_and_layers
// Train models DOC: https://www.tensorflow.org/js/guide/train_models


// node --inspect-brk example-model.js // Breaks on the first line of code
// node --inspect example-model.js

const tf = require('@tensorflow/tfjs');

// 2 layer model
const model = tf.sequential({
    layers: [
        tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }),
        tf.layers.dense({ units: 10, activation: 'softmax' }),
    ]
});

// NOTE: By default dense layers include a bias, but you can exclude it by specifying {useBias: false} in the options when creating a dense layer.

model.weights.forEach(w => {
    console.log(w.name, w.shape);
});

//model.summary()

// Each weight in the model is backend by a Variable object. In TensorFlow.js, a Variable is a floating-point Tensor with one additional method 
// assign() used for updating its values. The Layers API automatically initializes the weights using best practices. 
// For the sake of demonstration, we could overwrite the weights by calling assign() on the underlying variables:

model.weights.forEach(w => {
    const newVals = tf.randomNormal(w.shape);
    // w.val is an instance of tf.Variable
    w.val.assign(newVals);
});

//model.summary()


// Before you do any training, you need to decide on three things:
// 1. An optimizer. The job of the optimizer is to decide how much to change each parameter in the model, given the current model prediction. 
//    When using the Layers API, you can provide either a string identifier of an existing optimizer (such as 'sgd' or 'adam'), or an instance of the Optimizer class.
// 2. A loss function. An objective that the model will try to minimize. Its goal is to give a single number for "how wrong" the model's prediction was. 
//    The loss is computed on every batch of data so that the model can update its weights. When using the Layers API, you can provide either a 
//    string identifier of an existing loss function (such as 'categoricalCrossentropy'), or any function that takes a predicted and a true value and returns a loss.
//    See a list of available losses in our API docs.
// 3. List of metrics. Similar to losses, metrics compute a single number, summarizing how well our model is doing. 
//    The metrics are usually computed on the whole data at the end of each epoch. At the very least, we want to monitor that our loss is going down over time. 
//    However, we often want a more human-friendly metric such as accuracy. When using the Layers API, you can provide either a string identifier of an existing metric 
//    (such as 'accuracy'), or any function that takes a predicted and a true value and returns a score. See a list of available metrics in our API docs.

model.compile({
    optimizer: 'sgd',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

model.summary()



// Training
// There are two ways to train a LayersModel:
// * Using model.fit() and providing the data as one large tensor.
// * Using model.fitDataset() and providing the data via a Dataset object.


// model.fit()
// ***********
// If your dataset fits in main memory, and is available as a single tensor, you can train a model by calling the fit() method:


// Generate dummy data.
const data = tf.randomNormal([100, 784]);
const labels = tf.randomUniform([100, 10]);

function onBatchEnd(batch, logs) {
    console.log('Accuracy', logs.acc);
}

// Train for 5 epochs with batch size of 32.
model.fit(data, labels, {
    epochs: 5,
    batchSize: 32,
    callbacks: { onBatchEnd }
}).then(info => {
    console.log('Final accuracy', info.history.acc);
});


// Under the hood, model.fit() can do a lot for us:
// Splits the data into a train and validation set, and uses the validation set to measure progress during training.
// Shuffles the data but only after the split. To be safe, you should pre-shuffle the data before passing it to fit().
// Splits the large data tensor into smaller tensors of size batchSize.
// Calls optimizer.minimize() while computing the loss of the model with respect to the batch of data.
// It can notify you on the start and end of each epoch or batch. In our case, we are notified at the end of every batch using the callbacks.onBatchEndoption. 
// Other options include: onTrainBegin, onTrainEnd, onEpochBegin, onEpochEnd and onBatchBegin.
// It yields to the main thread to ensure that tasks queued in the JS event loop can be handled in a timely manner.
// For more info, see the documentation of fit(). Note that if you choose to use the Core API, you'll have to implement this logic yourself.

// model.fitDataset()
// ******************
// If your data doesn't fit entirely in memory, or is being streamed, you can train a model by calling fitDataset(), which takes a Dataset object. 
// Here is the same training code but with a dataset that wraps a generator function:


// function* data() {
//  for (let i = 0; i < 100; i++) {
//    // Generate one sample at a time.
//    yield tf.randomNormal([784]);
//  }
// }

// function* labels() {
//  for (let i = 0; i < 100; i++) {
//    // Generate one sample at a time.
//    yield tf.randomUniform([10]);
//  }
// }

// const xs = tf.data.generator(data);
// const ys = tf.data.generator(labels);
// // We zip the data and labels together, shuffle and batch 32 samples at a time.
// const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);

// // Train the model for 5 epochs.
// model.fitDataset(ds, {epochs: 5}).then(info => {
//  console.log('Accuracy', info.history.acc);
// });

// For more info about datasets, see the documentation of model.fitDataset().

// Predicting new data
// *******************
// Once the model has been trained, you can call model.predict() to make predictions on unseen data:


// Predict 3 random samples.
const prediction = model.predict(tf.randomNormal([3, 784]));
prediction.print();