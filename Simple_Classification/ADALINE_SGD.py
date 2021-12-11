from numpy.random import seed
import numpy as np
class AdalineSGD(object):
    """ADAptive LInear NEuron分類器

    パラメータ
    ------------
    eta : float
        学習率(0.0より大きく1.0以下)
    n_iter : int
        トレーニングデータのトレーニング回数
    shuffle : bool (デフォルト: True)
        Trueの場合は、循環を回避するためにエポックごとにトレーニングデータをシャッフル
    random_state : int
        重みを初期化するための乱数シード

    属性
    ------------
    w_ : 一次元配列
        適応後の重み
    cost_ : リスト
        各エポックでの誤差平方和のコスト関数

    """
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        # 重みの初期化フラグはFalseに設定
        self.w_initialized = False
        # 各エポックでトレーニングデータをシャッフルするかどうかのフラグを初期化
        self.shuffle = shuffle
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

        #　重みベクトルの生成
        self._initialize_weights(X.shape[1])
        # コストを格納するリストの生成
        self.cost_ = []
        # トレーニング回数分トレーニングデータを反復
        for i in range(self.n_iter):
            # 指定された場合はトレーニングデータをシャッフル
            if self.shuffle:
                X, y = self._shuffle(X, y)
            # 各サンプルのコストを収納するリストの生成
            cost = []
            # 各サンプルに対する計算
            for xi, target in zip(X, y):
                # 特徴量xiと目的変数yを用いた重みの更新とコストの計算
                cost.append(self._update_weights(xi, target))
            # サンプルの平均コストの計算
            avg_cost = sum(cost)/len(y)
            # 平均コストを格納
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """重みを再初期化することなくトレーニングデータに適合させる"""
        # 初期化されていない場合は初期化を実行
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        # 目的変数yの要素数が2以上の場合は
        # 各サンプルの特徴量xiろ目的変数targetで重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    def _shuffle(self, X, y):
        """トレーニングデータをシャッフル"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """重みを小さな乱数に初期化"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """ADALINEの学習規則を用いて重みを更新"""
        # 活性化関数の出力の計算
        output = self.activation(self.net_input(xi))
        # 誤差の計算
        error = (target - output)
        # wの更新
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        # コスト関数の計算
        cost = 0.5 * error ** 2
        return cost
    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return X
    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


