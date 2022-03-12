const { MOVE_TYPE, loadJSON, splitData } = require('./load-training-json');

const fileBasePath = '../data/Full';
const filesToLoad = [
    { file: fileBasePath + '/ABAJO.json', moveType: MOVE_TYPE.DOWN },
    { file: fileBasePath + '/ARRIBA.json', moveType: MOVE_TYPE.UP },
    { file: fileBasePath + '/IZQUIERDA.json', moveType: MOVE_TYPE.LEFT },
    { file: fileBasePath + '/DERECHA.json', moveType: MOVE_TYPE.RIGHT }
];

let loadedData = loadJSON(filesToLoad, { 
    shuffle: true, 
    split: true,
    dataAugmentation: true,
    dataAugmentationTotal: 50000
});

console.log("loadedData", loadedData);

console.log("loadedData", loadedData.samples[3270]);
console.log("loadedData", loadedData.labels[3270]);