import numpy as np
class AdalineGD(object):
    """ADAptive LInear NEuron分類器

    パラメータ
    ------------
    eta : float
        学習率(0.0より大きく1.0以下)
    n_iter : int
        トレーニングデータのトレーニング回数
    random_state : int
        重みを初期化するための乱数シード

    属性
    ------------
    w_ : 一次元配列
        適応後の重み
    cost_ : リスト
        各エポックでの誤差平方和のコスト関数

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """トレーニングデータに適合させる
        
        パラメータ
        ------------
        X : ｛配列のようなデータ構造｝，shape = [n_samples, n_features]
            トレーニングデータ
            n_samplesはサンプルの個数, n_featuresは特徴量の個数
        y : 配列のようなデータ構造，shape = [n_samples]
            目的変数
        
        戻り値
        -------
        self : object
        
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter): # トレーニング回数分トレーニングデータを反復
            net_input = self.net_input(X)
            # 活性化関数は恒等関数
            output = self.activation(net_input)
            # 誤差の計算
            errors = (y - output)
            # wの更新
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算
            cost = (errors ** 2).sum() / 2.0
            # コストの格納
            self.cost_.append(cost)
        return self
        
    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return X
    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


