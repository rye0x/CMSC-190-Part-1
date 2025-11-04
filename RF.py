import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def eval(data,label_name,ax,with_target):
    feature_names = ['rain','Tair','rh','Mean_EVI']

    if with_target:
        feature_names.append('lncase_0')
        
    feature_data = data[feature_names]
    label_data = data[label_name]

    model = RandomForestRegressor(n_jobs = -1)
    
    X_train = feature_data.loc[0:314, :]
    X_test = feature_data.loc[314:, :]
    y_train = label_data.loc[0:314]
    y_test = label_data.loc[314:]

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    print(label_name)
    # print('Train RMSE:',np.sqrt(np.mean((y_train.values - y_train_pred) ** 2)))
    # print('Train MAE:',np.mean(np.abs(y_train.values - y_train_pred)))

    rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
    mae = np.mean(np.abs(y_test.values - y_pred))
    print('Test RMSE:',rmse)
    print('Test MAE:',mae)

    return rmse, mae, model.predict(feature_data)

def plot_corr(data, feature_names, name, font_scale=0.2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()
    corr = data[feature_names].corr()
    plt.figure(dpi=600)
    sns.set_theme(font_scale=font_scale)
    sns.heatmap(corr, annot=True, xticklabels=corr.columns, yticklabels=corr.columns, square=True, cmap='Blues')
    plt.savefig(name)
    plt.close()


city_list = ['DF', 'Fortaleza']

with_target = False

for city in city_list:
    print(f'---------------{city}---------------')

    df_pred = pd.DataFrame(columns = ['lncase_1','lncase_2','lncase_3','lncase_4'])
    
    data_path = f'./data/{city}.csv'
    data = pd.read_csv(data_path)
    
    metrics = np.zeros((4,2))

    fig = plt.figure(figsize=(10,10))

    for n in range(4):
        ax = plt.subplot(4,1,n+1)
        metrics[n,0], metrics[n,1], pred = eval(data, 'lncase_'+str(n+1), ax, with_target)
        # ax.plot([314]*10,np.exp(np.linspace(0,10,10)),'--',color='k',alpha=.4)
        ax.plot(np.arange(314,data.shape[0]),np.exp(pred[314:]),label='Prediction')
        ax.plot(np.arange(314,data.shape[0]),np.exp(data['lncase_'+str(n+1)][314:]), color='grey', label='Ground Truth')
        ax.set_title(str(n+1)+'-week(s) ahead')
        plt.legend(loc='upper left')
        
        df_pred['lncase_'+str(n+1)] = pred

    ax = fig.add_subplot(111,frameon=False)
    plt.ylabel('ln(case)')
    plt.xlabel('Week')
    plt.tick_params(labelcolor='none', which = 'both', top=False,bottom=False, left = False, right=False)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs('./output_rf', exist_ok=True)

    if with_target:
        plt.savefig(f'./output_rf/{city}_with_target.png',dpi=600)
    else:
        plt.savefig(f'./output_rf/{city}.png',dpi=600)
    plt.show()
    plt.close()

    df = pd.DataFrame(data = metrics, columns = ['RMSE', 'MAE'], index = ['lncase_1','lncase_2','lncase_3','lncase_4'])
    if with_target:
        df.to_csv(f'./output_rf/{city}_metrics_with_target.csv')
    else:
        df.to_csv(f'./output_rf/{city}_metrics.csv')
    length = len(data['date'])
    for n in range(4):
        date = data['date'][326:length-n-1]
        df_tmp = pd.concat([date, df_pred['lncase_'+str(n+1)][326:length-n-1], data['lncase_'+str(n+1)][326:length-n-1]],axis = 1, ignore_index=True)
        df_tmp.columns = ['date', 'pred_y', 'test_y']
        if with_target:
            df_tmp.to_csv(f'./output_rf/{city}.csv_RF_{n+1}-week-ahead_input_3_split_314_round_1.csv',index=False)
        else:
            df_tmp.to_csv(f'./output_rf/{city}.csv_RF_{n+1}-week-ahead_input_2_split_314_round_1.csv',index=False)
