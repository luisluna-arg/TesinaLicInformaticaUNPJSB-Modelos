const fs = require('fs');
const _ = require('lodash');

const MOVE_TYPE = {
  NONE: 0,
  DOWN: 1,
  UP: 2,
  LEFT: 3,
  RIGHT: 4
};

function loadJSON(
  filename, moveType
) {

  /* Read lecture as JSON */
  let data = JSON.parse(
    fs.readFileSync(filename, { encoding: 'utf-8' })
  ).filter(o => o.poorSignalLevel === 0);

  /* Filter no lecture records */
  return data.
    filter(o => o.poorSignalLevel == 0).
    map((record) => {
      return {
        attention: record.eSense.attention,
        meditation: record.eSense.meditation,
        delta: record.eegPower.delta,
        theta: record.eegPower.theta,
        lowAlpha: record.eegPower.lowAlpha,
        highAlpha: record.eegPower.highAlpha,
        lowBeta: record.eegPower.lowBeta,
        highBeta: record.eegPower.highBeta,
        lowGamma: record.eegPower.lowGamma,
        highGamma: record.eegPower.highGamma,
        // move: moveType
        down: moveType == MOVE_TYPE.DOWN ? 1 : 0,
        up: moveType == MOVE_TYPE.UP ? 1 : 0,
        right: moveType == MOVE_TYPE.RIGHT ? 1 : 0,
        left: moveType == MOVE_TYPE.LEFT ? 1 : 0
      }
    });
}

module.exports = {
  MOVE_TYPE,
  loadJSON
};