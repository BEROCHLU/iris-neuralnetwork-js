# simple-based-neuralnetwork
ニューラルネットワークとは本来もっと単純なものであったが今日では複雑化しすぎている。  
このリポジトリでは古典的であるが、最も単純な回帰分析のニューラルネットワークをC言語、node.js、pythonの3つの言語で公開している。
このニューラルネットワークでは損失関数に最小二乗法を使っている。

3つのファイルに依存関係はなく独立している。
それぞれのファイルに追加モジュールなくすぐ実行できる。

# C language  
### build
`gcc -Wall -o "cdevice" "cdevice.c" -lm`
### excute
`./cdevice`
  
# node.js
### excute
`node nodevice.js`

# python
### excute
`python3 pydevice.py`
