# simple-based-neuralnetwork
ニューラルネットワークとは本来もっと単純なものであったが今日では複雑化しすぎている。  
このリポジトリでは古典的であるが、最も単純な回帰分析のニューラルネットワークをC言語、node.js、pythonの3つの言語で公開している。
このニューラルネットワークでは損失関数に最小二乗法を使っている。

3つのファイルに依存関係はなく独立している。
それぞれのファイルに追加モジュールなくすぐ実行できる。

# C language  
### build:
`gcc -Wall -o "cdevice" "cdevice.c" -lm`
### excute:
`./cdevice`
  
# node.js
### excute:
`node nodevice.js`

# python
### excute:
`python3 pydevice.py`

# Validation
ニューラルネットワークの比較、検証にXORおよびセルオートマトン30と90の真理値表を活用している。csvフォルダ及びjsonフォルダに格納されている。  
C言語はcsvフォルダを参照し、node.js pyhtonはjsonフォルダを参照している。

# Activation Function
活性化関数はシグモイド関数またはReLUを実装している。それぞれのファイルの変数値(0 or 1)で切り替え可能である。

## ReLU
ReLUはシグモイド関数に比べ最大で10倍高速であるが、ローカルミニマムに陥りやすく不安定である。セルオートマトン90では学習不可であった。
ReLUとシグモイド関数は中間層の活性化関数及びバイアス、微分に違いがある。従ってこの3点をif文で切り分けている。
