# simple-based-neuralnetwork
ニューラルネットワークとは本来もっと単純なものであったが今日では複雑化しすぎている。  
このリポジトリでは古典的であるが、最も単純な回帰分析のニューラルネットワークを公開している。
今日のニューラルネットワークでは損失関数に平均二乗誤差を使っているがここでは最小二乗法を使っている。

3種類の言語があるがそれぞれに依存関係はなく独立している。

## C language
cdevice.c
### requirement
Nothing
### build
gcc -Wall -o "cdevice" "cdevice.c" -lm
# Excute
./cdevice

## node.js
nodevice.js
### Excute
node nodevice.js

### python
pydevice.py
### Excute
python3 pydevice.py


## requirement
