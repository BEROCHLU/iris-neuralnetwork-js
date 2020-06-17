# simple-based-neuralnetwork
ニューラルネットワークとは本来もっと単純なものであったが今日では複雑化しすぎている。  
このリポジトリでは古典的であるが、最も単純な回帰分析のニューラルネットワークを公開している。
今日のニューラルネットワークでは損失関数に平均二乗誤差を使っているがここでは最小二乗法を使っている。

3種類の言語があるがそれぞれに依存関係はなく独立している。

cdevice.c
## ビルド
gcc -Wall -o "cdevice" "cdevice.c" -lm
# 実行
./cdevice


nodevice.js
## 実行
node nodevice.js


pydevice.py
## 実行
python3 pydevice.py
