'use strict';

const fs = require('fs');
const _ = require('lodash');
const math = require('mathjs');

let IN_NODE;
let HID_NODE;
let OUT_NODE;

const ETA = 0.5;
const THRESHOLD = 250;

const sigmoid = x => 1 / (1 + Math.exp(-x));
const dsigmoid = x => x * (1 - x);
const dfmax = x => (0 < x) ? 1 : 0;

let epoch; //学習回数
let DATA_LEN; //学習データ数
let TEST_LEN;
let arrDiff = [];

let hid = [];
let out = [];
let delta_out = [];
let delta_hid = [];

let x; //学習データ+バイアス
let t; //学習教師信号
let v = []; //v[HID_NODE][IN_NODE]
let w = []; //w[OUT_NODE][HID_NODE]
let xtest; //テストデータ+バイアス
let ttest; //テスト教師信号

//乱数生成
//const frandWeight = () => math.random(0.5, 1.0); // 0.5 <= x < 1.0
const frandWeight = () => Math.random(); //  0 <= x < 1.0, Math.random()
const frandBias = () => -1;

/**
 * 隠れ層、出力層の計算
 */
const calculateNode = (n) => {
    for (let i = 0; i < HID_NODE; i++) {
        hid[i] = sigmoid(math.dot(x[n], v[i]));
    }

    hid[HID_NODE - 1] = frandBias(); //配列最後にバイアス

    for (let i = 0; i < OUT_NODE; i++) {
        out[i] = sigmoid(math.dot(w[i], hid));
    }
}
/**
 * 
 * @param {Array}
 * @returns {Array}
 */
const outputNode = (arrInput) => {
    let arrHid = [];
    let arrOut = [];

    for (let i = 0; i < HID_NODE; i++) {
        arrHid[i] = sigmoid(math.dot(arrInput, v[i]));
    }

    arrHid[HID_NODE - 1] = frandBias(); //配列最後にバイアス

    for (let i = 0; i < OUT_NODE; i++) {
        arrOut[i] = sigmoid(math.dot(w[i], arrHid));
    }

    return arrOut;
}
/**
 * lamba
 * @param {*} n 
 * @returns 
 */
function roundfix(n) {
    return n.toFixed(1);
}
/**
 * 結果表示
 */
const printResult = () => {
    console.log();
    for (let i = 0; i < TEST_LEN; i++) {
        const arrOut = outputNode(xtest[i]);
        const ret = _.map(arrOut, roundfix);
        console.log(ret, ttest[i], xtest[i]);
    }
}
/**
 * Main
 */
{
    const strJson = fs.readFileSync('./json/iris-train.json', 'utf8'); //xor | cell30
    let arrHshTrain = JSON.parse(strJson);
    //arrHshTrain = _.shuffle(arrHshTrain);

    const strJsonTest = fs.readFileSync('./json/iris-test.json', 'utf8');
    const arrHshTest = JSON.parse(strJsonTest);

    x = _.map(arrHshTrain, hsh => {
        let arrBuf = hsh.input;
        arrBuf.push(frandBias()); //add input bias
        return arrBuf;
    });
    t = _.map(arrHshTrain, hsh => hsh.output);

    xtest = _.map(arrHshTest, hsh => {
        let arrBuf = hsh.input;
        arrBuf.push(frandBias()); //add input bias
        return arrBuf;
    });
    ttest = _.map(arrHshTest, hsh => hsh.output);

    IN_NODE = arrHshTrain[0].input.length; //入力ノード数決定（バイアス含む）
    HID_NODE = IN_NODE + 1; //隠れノード数決定
    OUT_NODE = arrHshTrain[0].output.length; //出力ノード数決定

    DATA_LEN = x.length;
    TEST_LEN = xtest.length;

    for (let i = 0; i < HID_NODE; i++) {
        v.push([]);
    }

    for (let i = 0; i < OUT_NODE; i++) {
        w.push([]);
    }

    for (let i = 0; i < HID_NODE; i++) {
        for (let j = 0; j < IN_NODE; j++) {
            v[i].push(frandWeight());
        }
    }

    for (let i = 0; i < OUT_NODE; i++) {
        for (let j = 0; j < HID_NODE; j++) {
            w[i].push(frandWeight());
        }
    }

    for (epoch = 0; epoch <= THRESHOLD; epoch++) {
        arrDiff = [];

        for (let n = 0; n < DATA_LEN; n++) {
            calculateNode(n);

            for (let k = 0; k < OUT_NODE; k++) {
                let diff = 0.5 * Math.pow((t[n][k] - out[k]), 2);
                arrDiff[k] = _.round(diff, 6);
                //console.log(t[n][k], out[k]);
                // Δw
                delta_out[k] = (t[n][k] - out[k]) * out[k] * (1 - out[k]); //δ=(t-o)*f'(net); net=Σwo; δo/δnet=f'(net);
            }

            for (let k = 0; k < OUT_NODE; k++) { // Δw
                for (let j = 0; j < HID_NODE; j++) {
                    w[k][j] += ETA * delta_out[k] * hid[j]; //Δw=ηδH
                }
            }

            for (let i = 0; i < HID_NODE; i++) { // Δv
                delta_hid[i] = 0;

                for (let k = 0; k < OUT_NODE; k++) {
                    delta_hid[i] += delta_out[k] * w[k][i]; //Σδw
                }

                delta_hid[i] = dsigmoid(hid[i]) * delta_hid[i]; //H(1-H)*Σδw

            }

            for (let i = 0; i < HID_NODE; i++) { // Δv
                for (let j = 0; j < IN_NODE; j++) {
                    v[i][j] += ETA * delta_hid[i] * x[n][j]; //Δu=ηH(1-H)XΣδw
                }
            }
            epoch = epoch + 0; //debug
        }

        if (epoch % 10 === 0) {
            const s = epoch + '';
            console.log(`${s.padStart(5)}: ${arrDiff}`);
        }
    } //for
    printResult();
}