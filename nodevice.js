'use strict';

const fs = require('fs');

const DESIRED_ERROR = 0.001;
const THRESHOLD = 20000;
const OUT_NODE = 1;
const ETA = 0.5;
const ACTIVE = 0; // 0: sigmoid, 1: ReLU

const sigmoid = x => 1 / (1 + Math.exp(-x));
const dsigmoid = x => x * (1 - x);
const dfmax = x => (0 < x) ? 1 : 0;

let IN_NODE;
let HID_NODE;

let hid = [];
let out = [];

let delta_out = [];
let delta_hid = [];

let epoch = 0; //学習回数
let DATA_LEN; //学習データ数
let fError = Number.MAX_SAFE_INTEGER;

let x; //学習データ+バイアス
let t; //教師信号

let v = []; //v[HID_NODE][IN_NODE]
let w = []; //w[OUT_NODE][HID_NODE]

/**
 * 隠れ層、出力層の計算
 */
const findHiddenOutput = (n) => {

    for (let i = 0; i < HID_NODE; i++) {
        let neth = 0;
        for (let j = 0; j < IN_NODE; j++) {
            neth += x[n][j] * v[i][j];
        }

        if (ACTIVE === 0) {
            hid[i] = sigmoid(neth);
        } else {
            hid[i] = Math.max(0, neth);
        }
    }

    if (ACTIVE === 0) {
        hid[HID_NODE - 1] = Math.random();
    } else {
        hid[HID_NODE - 1] = -1;
    }

    for (let i = 0; i < OUT_NODE; i++) {
        let neto = 0;
        for (let j = 0; j < HID_NODE; j++) {
            neto += w[i][j] * hid[j];
        }
        out[i] = sigmoid(neto);
    }
}
/**
 * 結果表示
 */
const printResult = () => {
    console.log();
    for (let i = 0; i < DATA_LEN; i++) {
        findHiddenOutput(i);
        const fout = out[0].toFixed(2);
        console.log(`${t[i][0]}: ${fout}`);
    }
}
/**
 * Main
 */
{
    const strJson = fs.readFileSync('./json/xor.json', 'utf8'); //xor | cell30
    const arrHsh = JSON.parse(strJson);

    x = arrHsh.map(hsh => {
        let arrBuf = hsh.input;
        arrBuf.push(Math.random()); //add input bias
        return arrBuf;
    });
    t = arrHsh.map(hsh => hsh.output);

    IN_NODE = arrHsh[0].input.length; //入力ノード数決定（バイアス含む）
    HID_NODE = IN_NODE + 1; //隠れノード数決定
    DATA_LEN = x.length;

    for (let i = 0; i < HID_NODE; i++) {
        v.push([]);
    }

    for (let i = 0; i < OUT_NODE; i++) {
        w.push([]);
    }

    /* 中間層の結合荷重を初期化 */
    for (let i = 0; i < HID_NODE; i++) {
        for (let j = 0; j < IN_NODE; j++) {
            v[i].push(Math.random());
        }
    }

    /* 出力層の結合荷重の初期化 */
    for (let i = 0; i < OUT_NODE; i++) {
        for (let j = 0; j < HID_NODE; j++) {
            w[i].push(Math.random());
        }
    }

    while (DESIRED_ERROR < fError) {
        epoch++;
        fError = 0;

        for (let n = 0; n < DATA_LEN; n++) {
            findHiddenOutput(n);

            for (let k = 0; k < OUT_NODE; k++) {
                fError += 0.5 * Math.pow((t[n][k] - out[k]), 2); //誤差を日数分加算する
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

                if (ACTIVE === 0) {
                    delta_hid[i] = dsigmoid(hid[i]) * delta_hid[i]; //H(1-H)*Σδw
                } else {
                    delta_hid[i] = dfmax(hid[i]) * delta_hid[i];
                }
            }

            for (let i = 0; i < HID_NODE; i++) { // Δv
                for (let j = 0; j < IN_NODE; j++) {
                    v[i][j] += ETA * delta_hid[i] * x[n][j]; //Δu=ηH(1-H)XΣδw
                }
            }
            epoch = epoch + 0; //debug
        }

        if (epoch % 100 === 0) {
            epoch = epoch + '';
            console.log(`${epoch.padStart(5)}: ${fError.toFixed(6)}`);
        }
        if (THRESHOLD < epoch) {
            break;
        }
    } //while

    printResult();
}