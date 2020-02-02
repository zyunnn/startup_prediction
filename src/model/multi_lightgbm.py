import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from data_loader import DataLoader
from config import Config



class multiLightGBM:

    # Helper function to reshape input feature and labels
    def reshape_features_labels(self, Xs, Ys):
        return Xs.reshape(Xs.shape[0],-1), Ys.flatten()

    # Train lightgbm model
    def train(self, Xs, ys):
        X, y = self.reshape_features_labels(Xs,ys)
        X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.25,random_state=7)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        params = {'boosting_type': 'gbdt', 
                'objective': 'binary',
                'learning_rate': 0.03, 
                'max_depth': 10, 
                'num_leaves': 512, 
                'feature_fraction': 0.8, 
                'bagging_fraction': 0.8, 
                'bagging_freq': 5}
        self.bst = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_eval,lgb_train], early_stopping_rounds=10)
        return self.bst

    # Predict using best iteration 
    def predict(self, X_test, y_test):
        X_test, y_test = self.reshape_features_labels(X_test, y_test)
        y_pred = self.bst.predict(X_test, num_iteration=self.bst.best_iteration)
        y_pred_rounded = np.round_(y_pred, 0)
        return y_pred_rounded


    # Evaluate model using metrics
    def evaluate(self, y_pred_rounded, y_test):
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_rounded)                                                                                                                                                                                                          
        print('Precision:', precision.mean())
        print('Recall:', recall.mean())
        print('f1-score:', f1.mean())

        print('Classification report:')
        print(classification_report(y_test, y_pred_rounded))

        print('Confusion matrix:')
        print(confusion_matrix(y_test, y_pred_rounded) )


    # Save best model
    def save_model(self):
        self.bst.save_model('./model/lightgbm.txt', num_iteration=self.bst.best_iteration)


if __name__ == "__main__":
    config = Config()
    d = DataLoader(config)
    model = multiLightGBM()
    X_train, y_train = d.get_batch_train()
    X_test, y_test = d.get_batch_test()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test, y_test)
    model.evaluate(y_pred, y_test)
    model.save_model()
