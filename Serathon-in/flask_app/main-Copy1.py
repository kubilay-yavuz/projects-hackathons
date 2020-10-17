""" Shows how to use flask and matplotlib together.
Shows SVG, and png.
The SVG is easier to style with CSS, and hook JS events to in browser.
python3 -m venv venv
. ./venv/bin/activate
pip install flask matplotlib
python flask_matplotlib.py
"""
import io
import random
from flask import Flask, Response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
import os
import glob
import json
from matplotlib.figure import Figure
import pandas as pd
import pandas as pd
import json
import requests 
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
app = Flask(__name__)


@app.route("/",methods=["POST","GET"])
def index():
    """ Returns html with the img tag for your plot.
    """
    num_x_points = request.args.get("stock_value")
    if not num_x_points:
        num_x_points="TSKB"
#     if "bist/analyze_{}.E.txt".format(num_x_points) in glob.glob("bist/*"):
    with open("bist/analyze_{}.E.txt".format(num_x_points),"r") as file:
        analy=file.read()
        analy=analy.replace("\n","<br>")
#     else:
#         analy="hello"
    # in a real app you probably want to use a flask template.
    return f"""<head><title>Dashboard</title></head><body style = "font-family:Helvetica Neue,verdana,garamond,serif;font-size:16px;">
    <h1>Stock predictions for {num_x_points}</h1>
    <form method=get action="/">
    <select name="stock_value" style = "font-family:Helvetica Neue,verdana,garamond,serif;">
          <option value="AKBNK">AKBNK</option>
          <option value="ARCLK">ARCLK</option>
          <option value="ASELS">ASELS</option>
          <option value="BIMAS">BIMAS</option>
          <option value="DOHOL">DOHOL</option>
          <option value="EKGYO">EKGYO</option>
          <option value="EREGL">EREGL</option>
          <option value="FROTO">FROTO</option>
          <option value="GARAN">GARAN</option>
          <option value="HALKB">HALKB</option>
          <option value="ISCTR">ISCTR</option>
          <option value="KCHOL">KCHOL</option>
          <option value="KOZAA">KOZAA</option>
          <option value="KOZAL">KOZAL</option>
          <option value="KRDMD">KRDMD</option>
          <option value="PETKM">PETKM</option>
          <option value="PGSUS">PGSUS</option>
          <option value="SAHOL">SAHOL</option>
          <option value="SISE">SISE</option>
          <option value="SODA">SODA</option>
          <option value="TAVHL">TAVHL</option>
          <option value="TCELL">TCELL</option>
          <option value="THYAO">THYAO</option>
          <option value="TKFEN">TKFEN</option>
          <option value="TOASO">TOASO</option>
          <option value="TSKB">TSKB</option>
          <option value="TTKOM">TTKOM</option>
          <option value="TUPRS">TUPRS</option>
          <option value="VAKBN">VAKBN</option>
          <option value="YKBNK">YKBNK</option>      
      <select/>
<input type=submit style = "font-family:Helvetica Neue,verdana,garamond,serif;" value="Update Graph">

    </form>

    <h3>Graphics of the stock respect to time</h3>
    <img style="display: inline; float: left;" src="/matplot-as-image-{num_x_points}.png"
         alt="random points as png"
         height="600">
    <h4 style="width: 600px; display: inline; float: left;"><p>{analy}</p></h4>
    <img style="display: inline; float: left;" src="/burki-as-image-{num_x_points}.png"
         alt="random points as png"
         height="600">        
    <form><input type="checkbox" checked>KAP<br>
    <input type="checkbox" checked>Bloomberg <br>
        <input type="checkbox" checked>NTV <br>
    <input type="checkbox">TRT<br>

    <input type="checkbox" checked>Twitter (@borsalab)<br>
    <input type="checkbox" >Twitter (#Borsa)<br>
    <input type="checkbox" >Twitter (#{num_x_points})<br>
    <input type="button" style = "font-family:Helvetica Neue,verdana,garamond,serif;" value="+ Add New" checked> <br>
    </form></body>
    """
    # from flask import render_template
    # return render_template("yourtemplate.html", num_x_points=num_x_points)


@app.route("/matplot-as-image-<num_x_points>.png")
def plot_png(num_x_points):
    """ renders the plot on the fly.
    """
    fig = Figure(figsize=(13,8))
    axis = fig.add_subplot(1, 1, 1)
    with open("plots/{}.E_plot.json".format(num_x_points),"r") as file:
        text=file.read()
        preds_dict=json.loads(text)
    x_points=[]
    for i in range(11,51):
        x_points.append(pd.to_datetime("2018.07.31 10:{}".format(i)))
    pred=preds_dict["prediction"]
    truth=preds_dict["truth"]
    axis.plot(x_points, truth,color="blue")
    axis.plot(x_points[-20:], pred,color="red")
    axis.set_xlabel("Time")
    axis.set_ylabel("Delta Stock Price / Stock Price")
    axis.set_title("Derivation of the Stock Price")
    axis.legend(("Ground Truth","Prediction"))
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")

@app.route("/burki-as-image-<num_x_points>.png")
def plot_png_burki(num_x_points=50):
#     url = "https://robodemo.infina.com.tr/robo/api/v0.7L/distribution"
#     token = '61aa55ad1f9acc25955c4e8c2e3ab7978925c6174b2b03a1a075c3fbfa92cd9d'
    fig, axis = plt.subplots(figsize=(12, 8), subplot_kw=dict(aspect="equal"))
# #     start_amount = input("Başlangıç Miktarı:")
# #     monthly_amount = input("Aylık Yatırım Miktarı:")
# #     age = input("Yaşınız:")
# #     risk = input("Risk Kategoriniz:")
#     start_amount="31131"
#     monthly_amount="1500"
#     age="25"
#     risk="2"
#     s = requests.Session()
#     data = {
#             "start_amount": start_amount,
#             "monthly_amount": monthly_amount,
#             "age" : int(age),
#             "risk": risk}
#     r = s.post(url, json=data, headers = {"X-Client-Token" : token})
#     with open("anan.txt","w") as file:
#         file.write(r.text)
    json_data = {"error":0,"message":"Success","distribution":[{"assetCode":"GOLD","weight":5.68200},{"assetCode":"BONDS","weight":49.11200},{"assetCode":"USD","weight":33.1600},{"assetCode":"XU030","weight":12.04500}]}
    vals = []
    key =['GOLD', 'BONDS', 'USD', 'BIST30']
    for i in range(4):
        vals.append(json_data["distribution"][i]["weight"])
    data = vals
    ingredients = key

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%".format(pct)

    wedges, texts, autotexts = axis.pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))
    axis.legend(wedges, ingredients,
              title="Suggested Portfolio",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              prop={'size': 23})
    plt.setp(autotexts, size=24, weight="bold")
    axis.set_title("Your Portfolio Distribution",fontdict={'fontsize': 24, 'fontweight': 'medium'})
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    plt.close()
    return Response(output.getvalue(), mimetype="image/png")

if __name__ == "__main__":
    import webbrowser

#     webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)