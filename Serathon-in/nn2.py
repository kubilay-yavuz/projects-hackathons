import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import os
from graphviz import Digraph
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.layers import *
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import time

bist30 = ['AKBNK.E', 'ARCLK.E', 'ASELS.E', 'BIMAS.E', 'DOHOL.E', 'EREGL.E', 'FROTO.E', 'GARAN.E', 'HALKB.E', 'ISCTR.E', 'KCHOL.E', 'KOZAA.E', 'KOZAL.E', 'KRDMD.E', 'PETKM.E', 'PGSUS.E', 'SAHOL.E', 'SISE.E', 'SODA.E', 'TAVHL.E', 'TCELL.E', 'THYAO.E', 'TKFEN.E', 'TOASO.E', 'TSKB.E', 'TTKOM.E', 'TUPRS.E', 'VAKBN.E', 'YKBNK.E', 'EKGYO.E']
bist100 = ['ADESE.E', 'AEFES.E', 'AFYON.E', 'AGHOL.E', 'AKBNK.E', 'AKSA.E', 'AKSEN.E', 'ALARK.E', 'ALBRK.E', 'ANACM.E', 'ARCLK.E', 'ASELS.E', 'BERA.E', 'BIMAS.E', 'BJKAS.E', 'CCOLA.E', 'CEMAS.E', 'CEMTS.E', 'CLEBI.E', 'DGKLB.E', 'DOHOL.E', 'ECILC.E', 'EGEEN.E', 'ENJSA.E', 'ENKAI.E', 'EREGL.E', 'FENER.E', 'FROTO.E', 'GARAN.E', 'GENTS.E', 'GEREL.E', 'GOLTS.E', 'GSDHO.E', 'GSRAY.E', 'GUBRF.E', 'HALKB.E', 'HEKTS.E', 'HURGZ.E', 'ICBCT.E', 'IEYHO.E', 'IHLAS.E', 'IHLGM.E', 'INDES.E', 'IPEKE.E', 'ISCTR.E', 'ISDMR.E', 'ISFIN.E', 'ITTFH.E', 'KARSN.E', 'KCHOL.E', 'KERVT.E', 'KONYA.E', 'KORDS.E', 'KOZAA.E', 'KOZAL.E', 'KRDMD.E', 'MAVI.E', 'METRO.E', 'MGROS.E', 'MPARK.E', 'NETAS.E', 'NTHOL.E', 'ODAS.E', 'OTKAR.E', 'PARSN.E', 'PETKM.E', 'PGSUS.E', 'POLHO.E', 'PRKME.E', 'SAHOL.E', 'SASA.E', 'SISE.E', 'SKBNK.E', 'SODA.E', 'SOKM.E', 'TAVHL.E', 'TCELL.E', 'THYAO.E', 'TKFEN.E', 'TMSN.E', 'TOASO.E', 'TRKCM.E', 'TSKB.E', 'TTKOM.E', 'TTRAK.E', 'TUPRS.E', 'ULKER.E', 'VAKBN.E', 'VERUS.E', 'VESTL.E', 'YATAS.E', 'YKBNK.E', 'ZOREN.E', 'ALGYO.E', 'EKGYO.E', 'GOZDE.E', 'ISGYO.E', 'OZGYO.E', 'AVOD.E', 'TUKAS.E']
bestofme = ['ADEL.E', 'ADESE.E', 'AEFES.E', 'AFYON.E', 'AGHOL.E', 'AKBNK.E', 'AKENR.E', 'AKSA.E', 'AKSEN.E', 'ALARK.E', 'ALBRK.E', 'ALCTL.E', 'ALGYO.E', 'ALKIM.E', 'ANACM.E', 'ANELE.E', 'ARCLK.E', 'ASELS.E', 'AVISA.E', 'AVOD.E', 'AYEN.E', 'AYGAZ.E', 'BAGFS.E', 'BANVT.E', 'BERA.E', 'BIMAS.E', 'BIZIM.E', 'BJKAS.E', 'BRISA.E', 'BRSAN.E', 'CCOLA.E', 'CEMAS.E', 'CEMTS.E', 'CIMSA.E', 'CLEBI.E', 'CRFSA.E', 'DEVA.E', 'DGATE.E', 'DGKLB.E', 'DOAS.E', 'DOCO.E', 'DOHOL.E', 'ECILC.E', 'ECZYT.E', 'EGEEN.E', 'EKGYO.E', 'ENJSA.E', 'ENKAI.E', 'ERBOS.E', 'EREGL.E', 'FENER.E', 'FLAP.E', 'FROTO.E', 'GARAN.E', 'GENTS.E', 'GEREL.E', 'GLYHO.E', 'GOLTS.E', 'GOODY.E', 'GOZDE.E', 'GSDHO.E', 'GSRAY.E', 'GUBRF.E', 'HALKB.E', 'HEKTS.E', 'HLGYO.E', 'HURGZ.E', 'ICBCT.E', 'IEYHO.E', 'IHLAS.E', 'IHLGM.E', 'INDES.E', 'IPEKE.E', 'ISCTR.E', 'ISDMR.E', 'ISFIN.E', 'ISGYO.E', 'ITTFH.E', 'IZMDC.E', 'KARSN.E', 'KARTN.E', 'KCHOL.E', 'KERVT.E', 'KIPA.E', 'KLGYO.E', 'KONYA.E', 'KORDS.E', 'KOZAA.E', 'KOZAL.E', 'KRDMD.E', 'KRONT.E', 'LOGO.E', 'MAVI.E', 'METRO.E', 'MGROS.E', 'MPARK.E', 'NETAS.E', 'NTHOL.E', 'NTTUR.E', 'NUGYO.E', 'ODAS.E', 'OTKAR.E', 'OZGYO.E', 'OZKGY.E', 'PARSN.E', 'PETKM.E', 'PGSUS.E', 'POLHO.E', 'PRKME.E', 'SAFGY.E', 'SAHOL.E', 'SASA.E', 'SELEC.E', 'SISE.E', 'SKBNK.E', 'SNGYO.E', 'SODA.E', 'SOKM.E', 'TATGD.E', 'TAVHL.E', 'TCELL.E', 'THYAO.E', 'TKFEN.E', 'TKNSA.E', 'TMSN.E', 'TOASO.E', 'TRCAS.E', 'TRGYO.E', 'TRKCM.E', 'TSKB.E', 'TSPOR.E', 'TTKOM.E', 'TTRAK.E', 'TUKAS.E', 'TUPRS.E', 'ULKER.E', 'VAKBN.E', 'VERUS.E', 'VESBE.E', 'VESTL.E', 'VKGYO.E', 'YATAS.E', 'YAZIC.E', 'YKBNK.E', 'ZOREN.E']

if os.path.exists("data.bin"):
    with open("data.bin", "rb") as file:
        data = pickle.load(file).dropna(axis=1)
else:
    data = pd.read_csv("bist100.csv")
    data = data.dropna(axis=1, thresh=int(data.shape[0] / 2)).interpolate(method="nearest")
    data['TARIH'] = pd.to_datetime(data["TARIH"],format="%Y-%m-%d")
    data.set_index(data['TARIH'], inplace=True)


    data2 = pd.read_csv("../eurofxref-hist.csv")
    data2["RON"] = data2["ROL"].fillna(0) / 10000.0 + data2["RON"].fillna(0)
    data2["TRY"] = data2["TRL"].fillna(0) / 1000000.0 + data2["TRY"].fillna(0)
    data2 = data2.drop(columns=["ROL", "TRL", "LTL", "LVL", "MTL", "SIT", "SKK", "CYP", "EEK", "ISK"])
    data2 = data2.dropna(how='all', axis='columns')
    data2['Date'] = pd.to_datetime(data2["Date"], format="%Y-%m-%d")
    data2.set_index(data2['Date'], inplace=True)
    columns = data2.columns[1:]
    truth = data2["TRY"]
    for column in columns:
        data2[column] = data2["TRY"] / data2[column]
    data2 = data2.drop(columns=["TRY"])
    print(data['TARIH'].tail(1))

    data = pd.merge(data, data2, how="inner", left_index=True, right_index=True).drop(columns=["Date"])
    data['yil'] = data['TARIH'].apply(lambda x:x.year)
    data['ay'] = data['TARIH'].apply(lambda x:x.month)
    data['gun'] = data['TARIH'].apply(lambda x:x.day)
    data['haftagunu'] = data['TARIH'].apply(lambda x:x.dayofweek)
    data["TARIH"]=data["TARIH"].apply(lambda x:x.value)
    with open("data.bin", "wb") as file:
        pickle.dump(data, file)
#data = data.replace([np.inf, -np.inf], 0)
data["TARIH"] = data["TARIH"] / 100000000000
columns = [column for column in data.columns.tolist() if "KAPANIS FIYATI" in column]
print(columns)
for column in columns:
    delta = data[column].diff(1)
    up, down = delta.copy(), delta.copy()
    emaslow = delta.copy().ewm(span=26, min_periods=1).mean()
    emafast = delta.copy().ewm(span=12, min_periods=1).mean()
    emafastest = delta.copy().ewm(span=5, min_periods=1).mean()
    emaa = emafast - emaslow
    emab = emafastest - emafast
    macd = emaa.ewm(span=9, min_periods=1).mean()
    #data["MACD_1_{0}".format(column)] = emaa.fillna(0)
    #data["MACD_2_{0}".format(column)] = macd.fillna(0)
    #data["last_{0}".format(column)] = (data[column].diff(1) * -1 / data[column]).fillna(0).apply(lambda x: x if x != np.inf else 0)
    #data["MA_20_{0}".format(column)] = delta.copy().rolling(window=20).mean()
    #data["MA_5_{0}".format(column)] = delta.copy().rolling(window=5).mean()

    up[up < 0] = 0
    down[down > 0] = 0

    roll_up2 = up.rolling(window=100).mean()
    roll_down2 = down.abs().rolling(window=100).mean()
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))

    #data["RSI_{0}".format(column)] = RSI2.fillna(0)

#backup_data = data.copy()
back = 40
window = 20
#data = backup_data.copy()
guess_scaler = MinMaxScaler()
min_max_scaler = MinMaxScaler()

#data["guess"] = data[base_column_str].diff(-1) * -1 / data[base_column_str]
#data["guess"] = data["guess"].apply(lambda x: x if x != np.inf else 0)
#data = data.dropna()

var_sets = ["yil", "ay", "gun", "haftagunu"]
dfDummies = [pd.get_dummies(data[col], prefix='one_hot_') for col in var_sets]
data = pd.concat([data, *dfDummies], axis=1).drop(columns=var_sets)

guess_columns = []
for column_ in columns:
    if any(x in str(column_) for x in bestofme):
        column_name = "guess_{0}".format(column_)
        guess_columns.append(column_name)
        data[column_name] = data[column_].diff(-1) * -1 / data[column_]
        data[column_name] = data[column_name].apply(lambda x: x if x != np.inf else 0).fillna(0)
truth = np.array(data[guess_columns][-back:])
new_guess = data[guess_columns].copy()
new_guess[new_guess < 0] = 0
new_guess = new_guess.divide(new_guess.sum(axis=1), axis=0)
data[guess_columns] = new_guess.fillna(0)

print(data[guess_columns].tail())

data = data.replace([np.inf, -np.inf], 0)
print(data.head())
to_predict = data[-window-1:].fillna(0)

data = data[:-1]
#data = data.dropna()
#data[guess_columns] = guess_scaler.fit_transform(data[guess_columns])
#data = data.drop(columns=columns)

minmax_columns = [str(column) for column in data.columns if str(column) not in var_sets and str(column) not in guess_columns]
data[minmax_columns] = min_max_scaler.fit_transform(data[minmax_columns])

features = np.array(data.drop(columns=guess_columns), dtype=np.float)
features = features
labels = data[guess_columns]
labels = np.array(labels)

to_predict_features =  np.array(to_predict.drop(columns=guess_columns), dtype=np.float)
to_predict_labels = to_predict[guess_columns]
to_predict_labels = np.array(to_predict_labels)


train_data_gen = TimeseriesGenerator(features[:-back], labels[:-back],
                                     length=window, sampling_rate=1, stride=1,
                                 batch_size=32  )

test_data_gen = TimeseriesGenerator(features[-(window + back):], labels[-(window + back):],
                                    length=window, sampling_rate=1, stride=1,
                                    batch_size=1)

predict_data_gen = TimeseriesGenerator(to_predict_features, to_predict_labels,
                                    length=window, sampling_rate=1, stride=1,
                                    batch_size=1)

model = keras.Sequential()
model.add(CuDNNLSTM(64, input_shape=(window, features.shape[-1])))
for i in [256, 256, 256]:
    model.add(Dense(i))
    model.add(BatchNormalization())
    model.add(ReLU(6.0))
    model.add(Dropout(0.1))
model.add(Dense(labels.shape[-1], activation="softmax"))

model.summary()

model.compile(keras.optimizers.adam(lr=1e-2), loss="binary_crossentropy", metrics=["binary_crossentropy", "mae"])
# 1.26867

logdir = "logs/scalars/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint = ModelCheckpoint("model.h5", monitor='loss', verbose=2, save_best_only=True, mode='min',
                             save_weights_only=True)
reduce = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='min', min_delta=1e-4,
                           cooldown=6, min_lr=1e-8)
es = EarlyStopping(patience=30, min_delta=1e-4, monitor="loss", restore_best_weights=False)



model.fit_generator(train_data_gen, epochs=1000, verbose=2, validation_data=test_data_gen, callbacks=[checkpoint, reduce, es, tensorboard_callback])


predictions = model.predict_generator(test_data_gen)
#predictions = guess_scaler.inverse_transform(predictions)

#losses = np.mean(np.power(predictions - truth, 2), axis=1)
#losses = np.array([np.zeros((predictions.shape[-1])) + loss for loss in losses.flatten()])
#print(losses.shape, predictions.shape)
#predictions-=losses

#predictions = np.transpose(predictions)
#truth = np.transpose(truth)

capita = 100.0
for i in range(predictions.shape[0]):
    new_predictions = np.array(predictions[i])
    new_predictions[new_predictions < 0] = 0
    #= guess_scaler.fit_transform(np.expand_dims(new_predictions, 2)).flatten()
    sum_ = new_predictions.sum()
    if sum_ == 0:
        continue
    capita_share = new_predictions / sum_ * capita
    capita_share += capita_share * truth[i]
    capita = capita_share.sum()
    print(capita)

nextday = dict(zip([column.split("_")[-1][:-2] for column in columns], model.predict_generator(predict_data_gen)[0] * 500))
print(nextday)