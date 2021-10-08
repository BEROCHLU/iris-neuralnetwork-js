# iris-neuralnetwork-js
Node.jsで構築したニューラルネットワークでアヤメの分類問題に挑戦

# dependency
* lodash
* mathjs@6.6.5

# excute
`node main.js`

# Traning
学習回数: 数千  
学習係数: 0.5  
活性化関数: シグモイド関数  
訓練データ: 120  
テストデータ: 30  
訓練データシャッフル: あり  
入力ノード: 4  
中間ノード: 5  
出力ノード: 3  
重み: v(入力<->中間), w(中間<->出力)  
重みの初期値: 0 <= v,w < 1  
バイアス: -1  