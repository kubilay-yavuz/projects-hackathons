import pandas as pd
import os

folder = "files"
datas = []
i = 0
for file in os.listdir(folder):
    extension = file.split(".")[-1]
    if extension == "csv":
        data = pd.read_csv(os.path.join(folder, file), sep=";", usecols=["TARIH", "KAPANIS FIYATI", "TOPLAM ISLEM HACMI", "TOPLAM ISLEM ADEDI", "ISLEM  KODU", "TOPLAM SOZLESME SAYISI", "BIST 100 ENDEKS"])[1:]
        data = data[data["BIST 100 ENDEKS"] == "1"].drop(columns=["BIST 100 ENDEKS"])
        data = pd.pivot(data, columns="ISLEM  KODU", index="TARIH", values=["KAPANIS FIYATI", "TOPLAM ISLEM HACMI", "TOPLAM ISLEM ADEDI", "TOPLAM SOZLESME SAYISI"])
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
        datas.append(data)
        i+=1
        if i % 100 == 0:
            print(file)

new_data = pd.concat(datas, axis=0)
new_data.to_csv("bist100.csv")