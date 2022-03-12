const tf = require('@tensorflow/tfjs-node');
const SMOTE = require('smote');
const fs = require('fs');

const { MOVE_TYPE, loadJSON, splitData } = require('./js/load-training-json');

const fileBasePath = './data/Full';
//const fileBasePath = './data/2022.2.22';

const filesToLoad = [
    { file: fileBasePath + '/ABAJO.json', moveType: MOVE_TYPE.DOWN },
    { file: fileBasePath + '/ARRIBA.json', moveType: MOVE_TYPE.UP },
    { file: fileBasePath + '/IZQUIERDA.json', moveType: MOVE_TYPE.LEFT },
    { file: fileBasePath + '/DERECHA.json', moveType: MOVE_TYPE.RIGHT }
];

let loadedData = loadJSON(filesToLoad, { shuffle: false, split: false });

/*
attention
meditation
delta
theta
lowAlpha
highAlpha
lowBeta
highBeta
lowGamma
highGamma
down
up
right
left
*/
const arrayedData = loadedData.samples.map(s =>
    [
        s.attention,
        s.meditation,
        s.delta,
        s.theta,
        s.lowAlpha,
        s.highAlpha,
        s.lowBeta,
        s.highBeta,
        s.lowGamma,
        s.highGamma,
        s.down,
        s.up,
        s.right,
        s.left
    ]
)

// Pass in your real data vectors.
const smote = new SMOTE(arrayedData);

// Here we generate 5 synthetic data points to bolster our training data with an balance an imbalanced data set.
const countToGenerate = 150000;
const newVectors = smote.generate(countToGenerate);

let generatedSamples = newVectors.map(a => {
    let labelStart = a.length - 4;
    let features = a.slice(0, labelStart).map(o => Math.floor(o));
    let index = tf.tensor(a.slice(labelStart)).softmax().argMax().dataSync();
    let labels = [0, 0, 0, 0];
    labels[index] = 1;
    return features.concat(labels);
});

let downData = arrayedData.concat(generatedSamples).filter(sampleArray => sampleArray[sampleArray.length - 4] == 1);
let up = arrayedData.concat(generatedSamples).filter(sampleArray => sampleArray[sampleArray.length - 3] == 1);
let right = arrayedData.concat(generatedSamples).filter(sampleArray => sampleArray[sampleArray.length - 2] == 1);
let left = arrayedData.concat(generatedSamples).filter(sampleArray => sampleArray[sampleArray.length - 1] == 1);

console.log(downData.length);
console.log(up.length);
console.log(right.length);
console.log(left.length);

function deleteLog(err) {
    if (err) console.error(err.message);
    else console.log("File deleted successfully");
};

function writeLog(err) {
    if (err) console.error(err.message);
    else console.log("File created successfully");
};


function dataItemCreator(dataArray) {
    let dataIndex = 0;
    let sample = {
        "ts": null,
        "eSense": {
            "attention": 0,
            "meditation": 0
        },
        "eegPower": {
            "delta": dataArray[dataIndex++],
            "theta": dataArray[dataIndex++],
            "lowAlpha": dataArray[dataIndex++],
            "highAlpha": dataArray[dataIndex++],
            "lowBeta": dataArray[dataIndex++],
            "highBeta": dataArray[dataIndex++],
            "lowGamma": dataArray[dataIndex++],
            "highGamma": dataArray[dataIndex++]
        },
        "poorSignalLevel": 0
    };
    return sample;
}


const options = { recursive:true };
let abajoFile = 'data/generated/ABAJO.json';
let arribaFile = 'data/generated/ARRIBA.json';
let derechaFile = 'data/generated/DERECHA.json';
let izquierdaFile = 'data/generated/IZQUIERDA.json';

function recreateFile(path) {
    fs.rm(path, options, (err) => { 
        deleteLog(err); 
        fs.writeFile(path, JSON.stringify(downData.map(dataItemCreator)), writeLog);
    });
}

recreateFile(abajoFile);
recreateFile(arribaFile);
recreateFile(derechaFile);
recreateFile(izquierdaFile);
