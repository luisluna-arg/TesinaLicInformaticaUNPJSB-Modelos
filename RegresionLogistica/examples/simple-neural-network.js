const tf = require('@tensorflow/tfjs');

const DATA = tf.tensor([
    // infections, infected countries
    [2.0, 1.0],
    [5.0, 1.0],
    [7.0, 4.0],
    [12.0, 5.0],
])
const nextDayInfections = tf.expandDims(tf.tensor([5.0, 7.0, 12.0, 19.0]), 1)

const HIDDEN_SIZE = 4
const model = tf.sequential()

model.add(
    tf.layers.dense({
        inputShape: [DATA.shape[1]],
        units: HIDDEN_SIZE,
        activation: "tanh",
    })
);

model.add(
    tf.layers.dense({
        units: HIDDEN_SIZE,
        activation: "tanh",
    })
);

model.add(
    tf.layers.dense({
        units: 1,
    })
);

model.summary()

const ALPHA = 0.001
model.compile({
    optimizer: tf.train.sgd(ALPHA),
    loss: "meanSquaredError",
});


/* await */
model.fit(DATA, nextDayInfections, {
    epochs: 200,
    callbacks: {
        onEpochEnd: async (epoch, logs) => {
            if (epoch % 10 === 0) {
                console.log(`Epoch ${epoch}: error: ${logs.loss}`)
            }
        },
    },
}).then(function (data) {
    console.log('[0]Weights', model.layers[0].getWeights()[0].shape);
    model.layers[0].getWeights()[0].print();
    
    console.log('[1]Weights', model.layers[1].getWeights()[0].shape)
    model.layers[1].getWeights()[0].print()
    
    const lastDayFeatures = tf.tensor([[12.0, 5.0]])
    model.predict(lastDayFeatures).print()
});
;


// console.log(model.layers[0].getWeights()[0].shape);
// model.layers[0].getWeights()[0].print();


// console.log(model.layers[1].getWeights()[0].shape)
// model.layers[1].getWeights()[0].print()


// const lastDayFeatures = tf.tensor([[12.0, 5.0]])
// model.predict(lastDayFeatures).print()