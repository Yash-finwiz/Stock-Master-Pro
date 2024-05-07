from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext

from plotly.offline import plot
import plotly.graph_objects as go
from django.conf import settings
import plotly.express as px
from django.http import JsonResponse
import snscrape.modules.twitter as sntwitter
import tweepy
from statsmodels.tsa.statespace.sarimax import SARIMAX

from praw import Reddit
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, RobertaTokenizer, RobertaForSequenceClassification
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import torch



from scipy.special import softmax
from tweepy import OAuthHandler, API

import talib
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta



import pandas as pd
import numpy as np
import json

import yfinance as yf
import datetime as dt
from prophet import Prophet

from .ML_Models.arima import predict_arima
from .ML_Models.lstm import predict_lstm

from .models import Project

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout




from textblob import TextBlob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import preprocessor as p
from transformers import XLNetTokenizer, XLNetForSequenceClassification




from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import mplfinance as mpf
import matplotlib.pyplot as plt
import io
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from plotly.subplots import make_subplots





from talib import RSI, CCI, ROC, MINMAX


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
import yfinance as yf
import numpy as np
from talib import RSI, CCI, ROC, SMA
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import GridSearchCV


import pandas_ta as ta

from backtesting import Backtest, Strategy
from bokeh.io import save
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook
from bokeh.io import export_png

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService


from decouple import config



# The Home page when Server loads up
def index(request):
    # ================================================= Left Card Plot =========================================================
    # Here we use yf.download function
    data = yf.download(
        
        # passes the ticker
        tickers=['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM'],
        
        group_by = 'ticker',
        
        threads=True, # Set thread value to true
        
        # used for access data[ticker]
        period='1mo', 
        interval='1d'
    
    )

    data.reset_index(level=0, inplace=True)



    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AAPL']['Adj Close'], name="AAPL")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AMZN']['Adj Close'], name="AMZN")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['QCOM']['Adj Close'], name="QCOM")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['META']['Adj Close'], name="META")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['NVDA']['Adj Close'], name="NVDA")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['JPM']['Adj Close'], name="JPM")
            )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')


    # ================================================ To show recent stocks ==============================================
    
    df1 = yf.download(tickers = 'AAPL', period='1d', interval='1d')
    df2 = yf.download(tickers = 'AMZN', period='1d', interval='1d')
    df3 = yf.download(tickers = 'GOOGL', period='1d', interval='1d')
    df4 = yf.download(tickers = 'UBER', period='1d', interval='1d')
    df5 = yf.download(tickers = 'TSLA', period='1d', interval='1d')
    df6 = yf.download(tickers = 'TWTR', period='1d', interval='1d')

    df1.insert(0, "Ticker", "AAPL")
    df2.insert(0, "Ticker", "AMZN")
    df3.insert(0, "Ticker", "GOOGL")
    df4.insert(0, "Ticker", "UBER")
    df5.insert(0, "Ticker", "TSLA")
    df6.insert(0, "Ticker", "TWTR")

    df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
    df.reset_index(level=0, inplace=True)
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    df = df.astype(convert_dict)
    df.drop('Date', axis=1, inplace=True)

    json_records = df.reset_index().to_json(orient ='records')
    recent_stocks = []
    recent_stocks = json.loads(json_records)

    # ========================================== Page Render section =====================================================

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })

def search(request):
    return render(request, 'search.html', {})

def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })


# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value, period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        return render(request, 'API_Down.html', {})

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        return render(request, 'Invalid_Days_Format.html', {})

    Valid_Ticker = [
    "A","AA","AAC","AACG","AACIW","AADI","AAIC","AAIN","AAL","AAMC","AAME","AAN","AAOI","AAON","AAP","AAPL","AAQC","AAT","AATC","AAU","AAWW","AB","ABB","ABBV","ABC","ABCB","ABCL","ABCM","ABEO","ABEV","ABG","ABIO","ABM","ABMD","ABNB","ABOS","ABR","ABSI","ABST","ABT","ABTX","ABUS","ABVC","AC","ACA","ACAB","ACAD","ACAQ","ACAXR","ACB","ACBA","ACC","ACCD","ACCO","ACEL","ACER","ACET","ACEV","ACEVW","ACGL","ACGLN","ACGLO","ACH","ACHC","ACHL","ACHR","ACHV","ACI","ACII","ACIU","ACIW","ACKIT","ACLS","ACLX","ACM","ACMR","ACN","ACNB","ACON","ACOR","ACP","ACQR","ACQRU","ACR","ACRE","ACRS","ACRX","ACST","ACT","ACTD","ACTDW","ACTG","ACU","ACV","ACVA","ACXP","ADAG","ADALW","ADAP","ADBE","ADC","ADCT","ADER","ADES","ADEX","ADGI","ADI","ADIL","ADM","ADMA","ADMP","ADN","ADNT","ADNWW","ADP","ADPT","ADRA","ADRT","ADSE","ADSEW","ADSK","ADT","ADTH","ADTN","ADTX","ADUS","ADV","ADVM","ADX","ADXN","AE","AEAC","AEACW","AEAE","AEAEW","AEE","AEF","AEFC","AEG","AEHAW","AEHL","AEHR","AEI","AEIS","AEL","AEM","AEMD","AENZ","AEO","AEP","AEPPZ","AER","AERC","AERI","AES","AESC","AESE","AEVA","AEY","AEYE","AEZS","AFAQ","AFAR","AFB","AFBI","AFCG","AFG","AFGB","AFGC","AFGD","AFGE","AFIB","AFL","AFMD","AFRI","AFRIW","AFRM","AFT","AFTR","AFYA","AG","AGAC","AGBAR","AGCB","AGCO","AGD","AGE","AGEN","AGFS","AGFY","AGGR","AGI","AGIL","AGILW","AGIO","AGL","AGLE","AGM","AGMH","AGNC","AGNCM","AGNCN","AGNCO","AGNCP","AGO","AGR","AGRI","AGRO","AGRX","AGS","AGTC","AGTI","AGX","AGYS","AHCO","AHG","AHH","AHI","AHPA","AHPI","AHRNW","AHT","AI","AIB","AIC","AIF","AIG","AIH","AIHS","AIKI","AIM","AIMAW","AIMC","AIN","AINC","AINV","AIO","AIP","AIR","AIRC","AIRG","AIRI","AIRS","AIRT","AIRTP","AIT","AIU","AIV","AIZ","AIZN","AJG","AJRD","AJX","AJXA","AKA","AKAM","AKAN","AKBA","AKICU","AKR","AKRO","AKTS","AKTX","AKU","AKUS","AKYA","AL","ALB","ALBO","ALC","ALCC","ALCO","ALDX","ALE","ALEC","ALEX","ALF","ALFIW","ALG","ALGM","ALGN","ALGS","ALGT","ALHC","ALIM","ALIT","ALJJ","ALK","ALKS","ALKT","ALL","ALLE","ALLG","ALLK","ALLO","ALLR","ALLT","ALLY","ALNA","ALNY","ALORW","ALOT","ALPA","ALPN","ALPP","ALR","ALRM","ALRN","ALRS","ALSA","ALSAR","ALSAU","ALSAW","ALSN","ALT","ALTG","ALTO","ALTR","ALTU","ALTUU","ALTUW","ALV","ALVO","ALVR","ALX","ALXO","ALYA","ALZN","AM","AMAL","AMAM","AMAO","AMAOW","AMAT","AMBA","AMBC","AMBO","AMBP","AMC","AMCI","AMCR","AMCX","AMD","AME","AMED","AMEH","AMG","AMGN","AMH","AMK","AMKR","AMLX","AMN","AMNB","AMOT","AMOV","AMP","AMPE","AMPG","AMPH","AMPI","AMPL","AMPS","AMPY","AMR","AMRC","AMRK","AMRN","AMRS","AMRX","AMS","AMSC","AMSF","AMST","AMSWA","AMT","AMTB","AMTD","AMTI","AMTX","AMWD","AMWL","AMX","AMYT","AMZN","AN","ANAB","ANAC","ANDE","ANEB","ANET","ANF","ANGH","ANGHW","ANGI","ANGN","ANGO","ANIK","ANIP","ANIX","ANNX","ANPC","ANSS","ANTE","ANTX","ANVS","ANY","ANZU","ANZUW","AOD","AOGO","AOMR","AON","AORT","AOS","AOSL","AOUT","AP","APA","APAC","APACW","APAM","APCX","APD","APDN","APEI","APEN","APG","APGB","APH","API","APLD","APLE","APLS","APLT","APM","APMIU","APO","APOG","APP","APPF","APPH","APPHW","APPN","APPS","APRE","APRN","APT","APTM","APTO","APTV","APTX","APVO","APWC","APXI","APYX","AQB","AQMS","AQN","AQNA","AQNB","AQNU","AQST","AQUA","AR","ARAV","ARAY","ARBE","ARBEW","ARBK","ARBKL","ARC","ARCB","ARCC","ARCE","ARCH","ARCK","ARCKW","ARCO","ARCT","ARDC","ARDS","ARDX","ARE","AREB","AREC","AREN","ARES","ARGD","ARGO","ARGU","ARGUU","ARGUW","ARGX","ARHS","ARI","ARIS","ARIZW","ARKO","ARKOW","ARKR","ARL","ARLO","ARLP","ARMK","ARMP","ARNC","AROC","AROW","ARQQ","ARQQW","ARQT","ARR","ARRWU","ARRWW","ARRY","ARTE","ARTEW","ARTL","ARTNA","ARTW","ARVL","ARVN","ARW","ARWR","ASA","ASAI","ASAN","ASAQ","ASAX","ASAXU","ASB","ASC","ASCAU","ASCB","ASCBR","ASG","ASGI","ASGN","ASH","ASIX","ASLE","ASLN","ASM","ASMB","ASML","ASND","ASNS","ASO","ASPA","ASPC","ASPCU","ASPCW","ASPN","ASPS","ASPU","ASR","ASRT","ASRV","ASTC","ASTE","ASTL","ASTLW","ASTR","ASTS","ASTSW","ASUR","ASX","ASXC","ASYS","ASZ","ATA","ATAI","ATAQ","ATAX","ATC","ATCO","ATCX","ATEC","ATEN","ATER","ATEX","ATGE","ATHA","ATHE","ATHM","ATHX","ATI","ATIF","ATIP","ATKR","ATLC","ATLCL","ATLCP","ATLO","ATNF","ATNFW","ATNI","ATNM","ATNX","ATO","ATOM","ATOS","ATR","ATRA","ATRC","ATRI","ATRO","ATSG","ATTO","ATUS","ATVC","ATVCU","ATVI","ATXI","ATXS","ATY","AU","AUB","AUBAP","AUBN","AUD","AUDC","AUGX","AUID","AUMN","AUPH","AUR","AURA","AURC","AUROW","AUS","AUST","AUTL","AUTO","AUUD","AUVI","AUY","AVA","AVAC","AVAH","AVAL","AVAN","AVAV","AVB","AVCO","AVCT","AVCTW","AVD","AVDL","AVDX","AVEO","AVGO","AVGOP","AVGR","AVID","AVIR","AVK","AVLR","AVNS","AVNT","AVNW","AVO","AVPT","AVPTW","AVRO","AVT","AVTE","AVTR","AVTX","AVXL","AVY","AVYA","AWF","AWH","AWI","AWK","AWP","AWR","AWRE","AWX","AX","AXAC","AXDX","AXGN","AXL","AXLA","AXNX","AXON","AXP","AXR","AXS","AXSM","AXTA","AXTI","AXU","AY","AYI","AYLA","AYRO","AYTU","AYX","AZ","AZEK","AZN","AZO","AZPN","AZRE","AZTA","AZUL","AZYO","AZZ","B","BA","BABA","BAC","BACA","BAFN","BAH","BAK","BALL","BALY","BAM","BAMH","BAMI","BAMR","BANC","BAND","BANF","BANFP","BANR","BANX","BAOS","BAP","BARK","BASE","BATL","BATRA","BATRK","BAX","BB","BBAI","BBAR","BBBY","BBCP","BBD","BBDC","BBDO","BBGI","BBI","BBIG","BBIO","BBLG","BBLN","BBN","BBQ","BBSI","BBU","BBUC","BBVA","BBW","BBWI","BBY","BC","BCAB","BCAC","BCACU","BCACW","BCAN","BCAT","BCBP","BCC","BCDA","BCDAW","BCE","BCEL","BCH","BCLI","BCML","BCO","BCOR","BCOV","BCOW","BCPC","BCRX","BCS","BCSA","BCSAW","BCSF","BCTX","BCTXW","BCV","BCX","BCYC","BDC","BDJ","BDL","BDN","BDSX","BDTX","BDX","BDXB","BE","BEAM","BEAT","BECN","BEDU","BEEM","BEKE","BELFA","BELFB","BEN","BENE","BENER","BENEW","BEP","BEPC","BEPH","BEPI","BERY","BEST","BFAC","BFAM","BFC","BFH","BFI","BFIIW","BFIN","BFK","BFLY","BFRI","BFRIW","BFS","BFST","BFZ","BG","BGB","BGCP","BGFV","BGH","BGI","BGNE","BGR","BGRY","BGRYW","BGS","BGSF","BGSX","BGT","BGX","BGXX","BGY","BH","BHAC","BHACU","BHAT","BHB","BHC","BHE","BHF","BHFAL","BHFAM","BHFAN","BHFAO","BHFAP","BHG","BHIL","BHK","BHLB","BHP","BHR","BHSE","BHSEW","BHV","BHVN","BIDU","BIG","BIGC","BIGZ","BIIB","BILI","BILL","BIMI","BIO","BIOC","BIOL","BIOR","BIOSW","BIOT","BIOTU","BIOTW","BIOX","BIP","BIPC","BIPH","BIPI","BIRD","BIT","BITF","BIVI","BJ","BJDX","BJRI","BK","BKCC","BKD","BKE","BKEP","BKEPP","BKH","BKI","BKKT","BKN","BKNG","BKR","BKSC","BKSY","BKT","BKTI","BKU","BKYI","BL","BLBD","BLBX","BLCM","BLCO","BLCT","BLD","BLDE","BLDEW","BLDP","BLDR","BLE","BLEU","BLEUU","BLEUW","BLFS","BLFY","BLI","BLIN","BLK","BLKB","BLMN","BLND","BLNG","BLNGW","BLNK","BLNKW","BLPH","BLRX","BLSA","BLTE","BLTS","BLTSW","BLU","BLUA","BLUE","BLW","BLX","BLZE","BMA","BMAC","BMAQ","BMAQR","BMBL","BME","BMEA","BMEZ","BMI","BMO","BMRA","BMRC","BMRN","BMTX","BMY","BNED","BNFT","BNGO","BNL","BNOX","BNR","BNRG","BNS","BNSO","BNTC","BNTX","BNY","BOAC","BOAS","BOC","BODY","BOE","BOH","BOKF","BOLT","BON","BOOM","BOOT","BORR","BOSC","BOTJ","BOWL","BOX","BOXD","BOXL","BP","BPAC","BPMC","BPOP","BPOPM","BPRN","BPT","BPTH","BPTS","BPYPM","BPYPN","BPYPO","BPYPP","BQ","BR","BRAC","BRACR","BRAG","BRBR","BRBS","BRC","BRCC","BRCN","BRDG","BRDS","BREZ","BREZR","BREZW","BRFH","BRFS","BRG","BRID","BRIV","BRIVW","BRKHU","BRKL","BRKR","BRLI","BRLT","BRMK","BRN","BRO","BROG","BROS","BRP","BRPM","BRPMU","BRPMW","BRQS","BRSP","BRT","BRTX","BRW","BRX","BRY","BRZE","BSAC","BSBK","BSBR","BSET","BSFC","BSGA","BSGAR","BSGM","BSIG","BSKY","BSKYW","BSL","BSM","BSMX","BSQR","BSRR","BST","BSTZ","BSVN","BSX","BSY","BTA","BTAI","BTB","BTBD","BTBT","BTCM","BTCS","BTCY","BTG","BTI","BTMD","BTMDW","BTN","BTO","BTOG","BTRS","BTT","BTTR","BTTX","BTU","BTWN","BTWNU","BTWNW","BTX","BTZ","BUD","BUI","BUR","BURL","BUSE","BV","BVH","BVN","BVS","BVXV","BW","BWA","BWAC","BWACW","BWAQR","BWAY","BWB","BWC","BWCAU","BWEN","BWFG","BWG","BWMN","BWMX","BWNB","BWSN","BWV","BWXT","BX","BXC","BXMT","BXMX","BXP","BXRX","BXSL","BY","BYD","BYFC","BYM","BYN","BYND","BYRN","BYSI","BYTS","BYTSW","BZ","BZFD","BZFDW","BZH","BZUN","C","CAAP","CAAS","CABA","CABO","CAC","CACC","CACI","CADE","CADL","CAE","CAF","CAG","CAH","CAJ","CAKE","CAL","CALA","CALB","CALM","CALT","CALX","CAMP","CAMT","CAN","CANF","CANG","CANO","CAPD","CAPL","CAPR","CAR","CARA","CARE","CARG","CARR","CARS","CARV","CASA","CASH","CASI","CASS","CASY","CAT","CATC","CATO","CATY","CB","CBAN","CBAT","CBAY","CBD","CBFV","CBH","CBIO","CBL","CBNK","CBOE","CBRE","CBRG","CBRL","CBSH","CBT","CBTX","CBU","CBZ","CC","CCAP","CCB","CCBG","CCCC","CCCS","CCD","CCEL","CCEP","CCF","CCI","CCJ","CCK","CCL","CCLP","CCM","CCNC","CCNE","CCNEP","CCO","CCOI","CCRD","CCRN","CCS","CCSI","CCU","CCV","CCVI","CCXI","CCZ","CD","CDAK","CDAY","CDE","CDEV","CDLX","CDMO","CDNA","CDNS","CDR","CDRE","CDRO","CDTX","CDW","CDXC","CDXS","CDZI","CDZIP","CE","CEA","CEAD","CEADW","CECE","CEE","CEG","CEI","CEIX","CELC","CELH","CELU","CELZ","CEM","CEMI","CEN","CENN","CENQW","CENT","CENTA","CENX","CEPU","CEQP","CERE","CERS","CERT","CET","CETX","CETXP","CEV","CEVA","CF","CFB","CFBK","CFFE","CFFI","CFFN","CFG","CFIV","CFIVW","CFLT","CFMS","CFR","CFRX","CFSB","CFVI","CFVIW","CG","CGA","CGABL","CGAU","CGBD","CGC","CGEM","CGEN","CGNT","CGNX","CGO","CGRN","CGTX","CHAA","CHCI","CHCO","CHCT","CHD","CHDN","CHE","CHEA","CHEF","CHEK","CHGG","CHH","CHI","CHK","CHKEL","CHKEW","CHKEZ","CHKP","CHMG","CHMI","CHN","CHNG","CHNR","CHPT","CHRA","CHRB","CHRD","CHRS","CHRW","CHS","CHSCL","CHSCM","CHSCN","CHSCO","CHSCP","CHT","CHTR","CHUY","CHW","CHWA","CHWAW","CHWY","CHX","CHY","CI","CIA","CIB","CIDM","CIEN","CIF","CIFR","CIFRW","CIG","CIGI","CIH","CII","CIIGW","CIK","CIM","CINC","CINF","CING","CINT","CIO","CION","CIR","CISO","CITEW","CIVB","CIVI","CIX","CIXX","CIZN","CJJD","CKPT","CKX","CL","CLAQW","CLAR","CLAS","CLAYU","CLB","CLBK","CLBS","CLBT","CLBTW","CLDT","CLDX","CLEU","CLF","CLFD","CLGN","CLH","CLIM","CLIR","CLLS","CLM","CLMT","CLNE","CLNN","CLOV","CLPR","CLPS","CLPT","CLR","CLRB","CLRO","CLS","CLSD","CLSK","CLSN","CLST","CLVR","CLVRW","CLVS","CLVT","CLW","CLWT","CLX","CLXT","CM","CMA","CMAX","CMAXW","CMBM","CMC","CMCA","CMCL","CMCM","CMCO","CMCSA","CMCT","CME","CMG","CMI","CMLS","CMMB","CMP","CMPO","CMPOW","CMPR","CMPS","CMPX","CMRA","CMRAW","CMRE","CMRX","CMS","CMSA","CMSC","CMSD","CMT","CMTG","CMTL","CMU","CNA","CNC","CNCE","CND","CNDB","CNDT","CNET","CNEY","CNF","CNFRL","CNHI","CNI","CNK","CNM","CNMD","CNNB","CNNE","CNO","CNOB","CNOBP","CNP","CNQ","CNR","CNS","CNSL","CNSP","CNTA","CNTB","CNTG","CNTQ","CNTQW","CNTX","CNTY","CNVY","CNX","CNXA","CNXC","CNXN","CO","COCO","COCP","CODA","CODI","CODX","COE","COF","COFS","COGT","COHN","COHU","COIN","COKE","COLB","COLD","COLI","COLIU","COLIW","COLL","COLM","COMM","COMP","COMS","COMSP","COMSW","CONN","CONX","CONXW","COO","COOK","COOL","COOLU","COOP","COP","CORR","CORS","CORT","CORZ","CORZW","COSM","COST","COTY","COUP","COUR","COVA","COVAU","COVAW","COWN","COWNL","CP","CPA","CPAC","CPAR","CPARU","CPARW","CPB","CPE","CPF","CPG","CPHC","CPHI","CPIX","CPK","CPLP","CPNG","CPOP","CPRI","CPRT","CPRX","CPS","CPSH","CPSI","CPSS","CPT","CPTK","CPTN","CPTNW","CPUH","CPZ","CQP","CR","CRAI","CRBP","CRBU","CRC","CRCT","CRDF","CRDL","CRDO","CREC","CREG","CRESW","CRESY","CREX","CRF","CRGE","CRGY","CRH","CRHC","CRI","CRIS","CRK","CRKN","CRL","CRM","CRMD","CRMT","CRNC","CRNT","CRNX","CRON","CROX","CRS","CRSP","CRSR","CRT","CRTD","CRTDW","CRTO","CRTX","CRU","CRUS","CRVL","CRVS","CRWD","CRWS","CRXT","CRXTW","CS","CSAN","CSBR","CSCO","CSCW","CSGP","CSGS","CSII","CSIQ","CSL","CSPI","CSQ","CSR","CSSE","CSSEN","CSSEP","CSTE","CSTL","CSTM","CSTR","CSV","CSWC","CSWI","CSX","CTAQ","CTAS","CTBB","CTBI","CTDD","CTEK","CTG","CTGO","CTHR","CTIB","CTIC","CTKB","CTLP","CTLT","CTMX","CTO","CTOS","CTR","CTRA","CTRE","CTRM","CTRN","CTS","CTSH","CTSO","CTT","CTV","CTVA","CTXR","CTXRW","CTXS","CUBA","CUBE","CUBI","CUE","CUEN","CUK","CULL","CULP","CURI","CURO","CURV","CUTR","CUZ","CVAC","CVBF","CVCO","CVCY","CVE","CVEO","CVET","CVGI","CVGW","CVI","CVII","CVLG","CVLT","CVLY","CVM","CVNA","CVR","CVRX","CVS","CVT","CVV","CVX","CW","CWAN","CWBC","CWBR","CWCO","CWEN","CWH","CWK","CWST","CWT","CX","CXAC","CXDO","CXE","CXH","CXM","CXW","CYAN","CYBE","CYBN","CYBR","CYCC","CYCCP","CYCN","CYD","CYH","CYN","CYRN","CYRX","CYT","CYTH","CYTK","CYTO","CYXT","CZNC","CZOO","CZR","CZWI","D","DAC","DADA","DAIO","DAKT","DAL","DALN","DAN","DAO","DAOO","DAOOU","DAOOW","DAR","DARE","DASH","DATS","DAVA","DAVE","DAVEW","DAWN","DB","DBD","DBGI","DBI","DBL","DBRG","DBTX","DBVT","DBX","DC","DCBO","DCF","DCFC","DCFCW","DCGO","DCGOW","DCI","DCO","DCOM","DCOMP","DCP","DCPH","DCRD","DCRDW","DCT","DCTH","DD","DDD","DDF","DDI","DDL","DDOG","DDS","DDT","DE","DEA","DECA","DECK","DEI","DELL","DEN","DENN","DEO","DESP","DEX","DFFN","DFH","DFIN","DFP","DFS","DG","DGHI","DGICA","DGII","DGLY","DGNU","DGX","DH","DHACW","DHBC","DHBCU","DHC","DHCAU","DHCNI","DHCNL","DHF","DHHC","DHI","DHIL","DHR","DHT","DHX","DHY","DIAX","DIBS","DICE","DIN","DINO","DIOD","DIS","DISA","DISH","DIT","DK","DKL","DKNG","DKS","DLA","DLB","DLCA","DLHC","DLNG","DLO","DLPN","DLR","DLTH","DLTR","DLX","DLY","DM","DMA","DMAC","DMB","DMF","DMLP","DMO","DMRC","DMS","DMTK","DNA","DNAA","DNAB","DNAC","DNAD","DNAY","DNB","DNLI","DNMR","DNN","DNOW","DNP","DNUT","DNZ","DO","DOC","DOCN","DOCS","DOCU","DOGZ","DOLE","DOMA","DOMO","DOOO","DOOR","DORM","DOUG","DOV","DOW","DOX","DOYU","DPG","DPRO","DPSI","DPZ","DQ","DRCT","DRD","DRE","DRH","DRI","DRIO","DRMA","DRMAW","DRQ","DRRX","DRTS","DRTSW","DRTT","DRUG","DRVN","DS","DSAC","DSACU","DSACW","DSEY","DSGN","DSGR","DSGX","DSKE","DSL","DSM","DSP","DSS","DSU","DSWL","DSX","DT","DTB","DTC","DTE","DTEA","DTF","DTG","DTIL","DTM","DTOCW","DTP","DTSS","DTST","DTW","DUK","DUKB","DUNE","DUNEW","DUO","DUOL","DUOT","DV","DVA","DVAX","DVN","DWAC","DWACU","DWACW","DWIN","DWSN","DX","DXC","DXCM","DXF","DXLG","DXPE","DXR","DXYN","DY","DYAI","DYFN","DYN","DYNT","DZSI","E","EA","EAC","EACPW","EAD","EAF","EAI","EAR","EARN","EAST","EAT","EB","EBACU","EBAY","EBC","EBET","EBF","EBIX","EBMT","EBON","EBR","EBS","EBTC","EC","ECAT","ECC","ECCC","ECCW","ECCX","ECF","ECL","ECOM","ECOR","ECPG","ECVT","ED","EDAP","EDBL","EDBLW","EDD","EDF","EDI","EDIT","EDN","EDNC","EDR","EDRY","EDSA","EDTK","EDTX","EDU","EDUC","EE","EEA","EEFT","EEIQ","EEX","EFC","EFL","EFOI","EFR","EFSC","EFSCP","EFT","EFTR","EFX","EGAN","EGBN","EGF","EGHT","EGIO","EGLE","EGLX","EGO","EGP","EGRX","EGY","EH","EHAB","EHC","EHI","EHTH","EIC","EICA","EIG","EIGR","EIM","EIX","EJH","EKSO","EL","ELA","ELAN","ELAT","ELBM","ELC","ELDN","ELEV","ELF","ELMD","ELOX","ELP","ELS","ELSE","ELTK","ELV","ELVT","ELY","ELYM","ELYS","EM","EMAN","EMBC","EMBK","EMBKW","EMCF","EMD","EME","EMF","EMKR","EML","EMLD","EMN","EMO","EMP","EMR","EMWP","EMX","ENB","ENBA","ENCP","ENCPW","ENDP","ENER","ENERW","ENFN","ENG","ENIC","ENJ","ENJY","ENJYW","ENLC","ENLV","ENO","ENOB","ENOV","ENPC","ENPH","ENR","ENS","ENSC","ENSG","ENSV","ENTA","ENTFW","ENTG","ENTX","ENTXW","ENV","ENVA","ENVB","ENVX","ENX","ENZ","EOCW","EOD","EOG","EOI","EOLS","EOS","EOSE","EOSEW","EOT","EP","EPAC","EPAM","EPC","EPD","EPHY","EPHYU","EPHYW","EPIX","EPM","EPR","EPRT","EPSN","EPWR","EPZM","EQ","EQBK","EQC","EQD","EQH","EQHA","EQIX","EQNR","EQOS","EQR","EQRX","EQRXW","EQS","EQT","EQX","ERAS","ERC","ERES","ERESU","ERF","ERH","ERIC","ERIE","ERII","ERJ","ERO","ERYP","ES","ESAB","ESAC","ESCA","ESE","ESEA","ESGR","ESGRO","ESGRP","ESI","ESLT","ESMT","ESNT","ESOA","ESPR","ESQ","ESRT","ESS","ESSA","ESSC","ESSCW","ESTA","ESTC","ESTE","ET","ETAC","ETACW","ETB","ETD","ETG","ETJ","ETN","ETNB","ETO","ETON","ETR","ETRN","ETSY","ETTX","ETV","ETW","ETWO","ETX","ETY","EUCR","EURN","EVA","EVAX","EVBG","EVBN","EVC","EVCM","EVER","EVEX","EVF","EVFM","EVG","EVGN","EVGO","EVGOW","EVGR","EVH","EVI","EVK","EVLO","EVLV","EVM","EVN","EVO","EVOJ","EVOJU","EVOJW","EVOK","EVOP","EVR","EVRG","EVRI","EVT","EVTC","EVTL","EVTV","EVV","EW","EWBC","EWCZ","EWTX","EXAI","EXAS","EXC","EXD","EXEL","EXFY","EXG","EXK","EXLS","EXN","EXP","EXPD","EXPE","EXPI","EXPO","EXPR","EXR","EXTN","EXTR","EYE","EYEN","EYES","EYPT","EZFL","EZGO","EZPW","F","FA","FACA","FACT","FAF","FAM","FAMI","FANG","FANH","FARM","FARO","FAST","FAT","FATBB","FATBP","FATE","FATH","FATP","FAX","FBC","FBHS","FBIO","FBIOP","FBIZ","FBK","FBMS","FBNC","FBP","FBRT","FBRX","FC","FCAP","FCAX","FCBC","FCCO","FCEL","FCF","FCFS","FCN","FCNCA","FCNCO","FCNCP","FCO","FCPT","FCRD","FCRX","FCT","FCUV","FCX","FDBC","FDEU","FDMT","FDP","FDS","FDUS","FDX","FE","FEAM","FEDU","FEI","FELE","FEMY","FEN","FENC","FENG","FEO","FERG","FET","FEXD","FEXDR","FF","FFA","FFBC","FFC","FFHL","FFIC","FFIE","FFIEW","FFIN","FFIV","FFNW","FFWM","FGB","FGBI","FGBIP","FGEN","FGF","FGFPP","FGI","FGIWW","FGMC","FHB","FHI","FHN","FHS","FHTX","FIAC","FIACW","FIBK","FICO","FIF","FIGS","FINM","FINMW","FINS","FINV","FINW","FIS","FISI","FISV","FITB","FITBI","FITBO","FITBP","FIVE","FIVN","FIX","FIXX","FIZZ","FKWL","FL","FLAC","FLACU","FLAG","FLC","FLEX","FLGC","FLGT","FLIC","FLL","FLME","FLNC","FLNG","FLNT","FLO","FLR","FLS","FLT","FLUX","FLWS","FLXS","FLYA","FLYW","FMAO","FMBH","FMC","FMIV","FMIVW","FMN","FMNB","FMS","FMTX","FMX","FMY","FN","FNA","FNB","FNCB","FNCH","FND","FNF","FNGR","FNHC","FNKO","FNLC","FNV","FNVTW","FNWB","FNWD","FOA","FOCS","FOF","FOLD","FONR","FOR","FORA","FORD","FORG","FORM","FORR","FORTY","FOSL","FOSLL","FOUN","FOUNU","FOUNW","FOUR","FOX","FOXA","FOXF","FOXW","FPAC","FPAY","FPF","FPH","FPI","FPL","FR","FRA","FRAF","FRBA","FRBK","FRBN","FRBNW","FRC","FRD","FREE","FREQ","FREY","FRG","FRGAP","FRGE","FRGI","FRGT","FRHC","FRLAW","FRLN","FRME","FRMEP","FRO","FROG","FRON","FRONU","FRPH","FRPT","FRSG","FRSGW","FRSH","FRST","FRSX","FRT","FRWAW","FRXB","FSBC","FSBW","FSD","FSEA","FSFG","FSI","FSK","FSLR","FSLY","FSM","FSNB","FSP","FSR","FSRD","FSRDW","FSRX","FSS","FSSI","FSSIW","FST","FSTR","FSTX","FSV","FT","FTAA","FTAI","FTAIN","FTAIO","FTAIP","FTCH","FTCI","FTCV","FTCVU","FTCVW","FTDR","FTEK","FTEV","FTF","FTFT","FTHM","FTHY","FTI","FTK","FTNT","FTPA","FTPAU","FTRP","FTS","FTV","FTVI","FUBO","FUL","FULC","FULT","FULTP","FUN","FUNC","FUND","FURY","FUSB","FUSN","FUTU","FUV","FVAM","FVCB","FVIV","FVRR","FWBI","FWONA","FWONK","FWP","FWRD","FWRG","FXCO","FXCOR","FXLV","FXNC","FYBR","G","GAB","GABC","GACQ","GACQW","GAIA","GAIN","GAINN","GALT","GAM","GAMB","GAMC","GAME","GAN","GANX","GAPA","GAQ","GASS","GATEW","GATO","GATX","GAU","GB","GBAB","GBBK","GBBKR","GBBKW","GBCI","GBDC","GBIO","GBL","GBLI","GBNH","GBOX","GBR","GBRGR","GBS","GBT","GBTG","GBX","GCBC","GCI","GCMG","GCMGW","GCO","GCP","GCV","GD","GDDY","GDEN","GDL","GDNRW","GDO","GDOT","GDRX","GDS","GDV","GDYN","GE","GECC","GECCM","GECCN","GECCO","GEEX","GEEXU","GEF","GEG","GEGGL","GEHI","GEL","GENC","GENE","GENI","GEO","GEOS","GER","GERN","GES","GET","GEVO","GF","GFAI","GFAIW","GFF","GFGD","GFI","GFL","GFLU","GFS","GFX","GGAA","GGAAU","GGAAW","GGAL","GGB","GGE","GGG","GGMC","GGN","GGR","GGROW","GGT","GGZ","GH","GHAC","GHACU","GHC","GHG","GHIX","GHL","GHLD","GHM","GHRS","GHSI","GHY","GIB","GIC","GIFI","GIGM","GIII","GIIX","GIIXW","GIL","GILD","GILT","GIM","GIPR","GIPRW","GIS","GIW","GIWWW","GKOS","GL","GLAD","GLBE","GLBL","GLBS","GLDD","GLDG","GLEE","GLG","GLHA","GLLIR","GLLIW","GLMD","GLNG","GLO","GLOB","GLOP","GLP","GLPG","GLPI","GLQ","GLRE","GLS","GLSI","GLSPT","GLT","GLTO","GLU","GLUE","GLV","GLW","GLYC","GM","GMAB","GMBL","GMBLP","GMDA","GME","GMED","GMFI","GMGI","GMRE","GMS","GMTX","GMVD","GNAC","GNACU","GNE","GNFT","GNK","GNL","GNLN","GNPX","GNRC","GNS","GNSS","GNT","GNTX","GNTY","GNUS","GNW","GO","GOAC","GOBI","GOCO","GOED","GOEV","GOEVW","GOF","GOGL","GOGO","GOL","GOLD","GOLF","GOOD","GOODN","GOODO","GOOG","GOOGL","GOOS","GORO","GOSS","GOTU","GOVX","GP","GPAC","GPACU","GPACW","GPC","GPCO","GPCOW","GPI","GPJA","GPK","GPL","GPMT","GPN","GPOR","GPP","GPRE","GPRK","GPRO","GPS","GRAB","GRABW","GRAY","GRBK","GRC","GRCL","GRCYU","GREE","GREEL","GRF","GRFS","GRIL","GRIN","GRMN","GRNA","GRNAW","GRNQ","GROM","GROMW","GROV","GROW","GROY","GRPH","GRPN","GRTS","GRTX","GRVI","GRVY","GRWG","GRX","GS","GSAQ","GSAQW","GSAT","GSBC","GSBD","GSEV","GSHD","GSIT","GSK","GSL","GSLD","GSM","GSMG","GSQB","GSRM","GSRMU","GSUN","GSV","GT","GTAC","GTACU","GTBP","GTE","GTEC","GTES","GTH","GTHX","GTIM","GTLB","GTLS","GTN","GTPB","GTX","GTXAP","GTY","GUG","GURE","GUT","GVA","GVCIU","GVP","GWH","GWRE","GWRS","GWW","GXII","GXO","H","HA","HAAC","HAACU","HAACW","HAE","HAFC","HAIA","HAIAU","HAIAW","HAIN","HAL","HALL","HALO","HAPP","HARP","HAS","HASI","HAYN","HAYW","HBAN","HBANM","HBANP","HBB","HBCP","HBI","HBIO","HBM","HBNC","HBT","HCA","HCAR","HCARU","HCARW","HCAT","HCC","HCCI","HCDI","HCDIP","HCDIW","HCDIZ","HCI","HCIC","HCICU","HCII","HCKT","HCM","HCNE","HCNEU","HCNEW","HCP","HCSG","HCTI","HCVI","HCWB","HD","HDB","HDSN","HE","HEAR","HEES","HEI","HELE","HEP","HEPA","HEPS","HEQ","HERA","HERAU","HERAW","HES","HESM","HEXO","HFBL","HFFG","HFRO","HFWA","HGBL","HGEN","HGLB","HGTY","HGV","HHC","HHGCW","HHLA","HHS","HI","HIBB","HIE","HIG","HIGA","HIHO","HII","HIII","HIL","HILS","HIMS","HIMX","HIO","HIPO","HITI","HIVE","HIW","HIX","HL","HLBZ","HLBZW","HLF","HLG","HLGN","HLI","HLIO","HLIT","HLLY","HLMN","HLNE","HLT","HLTH","HLVX","HLX","HMC","HMCO","HMCOU","HMLP","HMN","HMNF","HMPT","HMST","HMTV","HMY","HNGR","HNI","HNNA","HNNAZ","HNRA","HNRG","HNST","HNVR","HNW","HOFT","HOFV","HOFVW","HOG","HOLI","HOLX","HOMB","HON","HONE","HOOD","HOOK","HOPE","HOTH","HOUR","HOUS","HOV","HOVNP","HOWL","HP","HPE","HPF","HPI","HPK","HPKEW","HPP","HPQ","HPS","HPX","HQH","HQI","HQL","HQY","HR","HRB","HRI","HRL","HRMY","HROW","HROWL","HRT","HRTG","HRTX","HRZN","HSAQ","HSBC","HSC","HSCS","HSDT","HSIC","HSII","HSKA","HSON","HST","HSTM","HSTO","HSY","HT","HTA","HTAQ","HTBI","HTBK","HTCR","HTD","HTFB","HTFC","HTGC","HTGM","HTH","HTHT","HTIA","HTIBP","HTLD","HTLF","HTLFP","HTOO","HTPA","HTY","HTZ","HTZWW","HUBB","HUBG","HUBS","HUDI","HUGE","HUGS","HUIZ","HUM","HUMA","HUMAW","HUN","HURC","HURN","HUSA","HUT","HUYA","HVBC","HVT","HWBK","HWC","HWCPZ","HWKN","HWKZ","HWM","HXL","HY","HYB","HYFM","HYI","HYLN","HYMC","HYMCW","HYMCZ","HYPR","HYRE","HYT","HYW","HYZN","HYZNW","HZN","HZNP","HZO","HZON","IAA","IAC","IACC","IAE","IAF","IAG","IART","IAS","IAUX","IBA","IBCP","IBER","IBEX","IBIO","IBKR","IBM","IBN","IBOC","IBP","IBRX","IBTX","ICAD","ICCC","ICCH","ICCM","ICD","ICE","ICFI","ICHR","ICL","ICLK","ICLR","ICMB","ICNC","ICPT","ICUI","ICVX","ID","IDA","IDAI","IDBA","IDCC","IDE","IDEX","IDN","IDR","IDRA","IDT","IDW","IDXX","IDYA","IE","IEA","IEAWW","IEP","IESC","IEX","IFBD","IFF","IFN","IFRX","IFS","IGA","IGAC","IGACW","IGC","IGD","IGI","IGIC","IGICW","IGMS","IGR","IGT","IGTAR","IH","IHD","IHG","IHIT","IHRT","IHS","IHT","IHTA","IIF","III","IIII","IIIIU","IIIIW","IIIN","IIIV","IIM","IINN","IINNW","IIPR","IIVI","IIVIP","IKNA","IKT","ILMN","ILPT","IMAB","IMAC","IMAQ","IMAQR","IMAQW","IMAX","IMBI","IMBIL","IMCC","IMCR","IMGN","IMGO","IMH","IMKTA","IMMP","IMMR","IMMX","IMNM","IMO","IMOS","IMPL","IMPP","IMPPP","IMPX","IMRA","IMRN","IMRX","IMTE","IMTX","IMUX","IMV","IMVT","IMXI","INAQ","INBK","INBKZ","INBX","INCR","INCY","INDB","INDI","INDIW","INDO","INDP","INDT","INFA","INFI","INFN","INFU","INFY","ING","INGN","INGR","INKA","INKAW","INKT","INM","INMB","INMD","INN","INNV","INO","INOD","INPX","INSE","INSG","INSI","INSM","INSP","INST","INSW","INT","INTA","INTC","INTEW","INTG","INTR","INTT","INTU","INTZ","INUV","INVA","INVE","INVH","INVO","INVZ","INVZW","INZY","IOBT","IONM","IONQ","IONR","IONS","IOSP","IOT","IOVA","IP","IPA","IPAR","IPAXW","IPDN","IPG","IPGP","IPHA","IPI","IPOD","IPOF","IPSC","IPVA","IPVF","IPVI","IPW","IPWR","IPX","IQ","IQI","IQMD","IQMDW","IQV","IR","IRBT","IRDM","IREN","IRIX","IRL","IRM","IRMD","IRNT","IRRX","IRS","IRT","IRTC","IRWD","IS","ISAA","ISD","ISDR","ISEE","ISIG","ISLE","ISLEW","ISO","ISPC","ISPO","ISPOW","ISR","ISRG","ISSC","ISTR","ISUN","IT","ITCB","ITCI","ITGR","ITHX","ITHXU","ITHXW","ITI","ITIC","ITOS","ITP","ITQ","ITRG","ITRI","ITRM","ITRN","ITT","ITUB","ITW","IVA","IVAC","IVC","IVCAU","IVCAW","IVCB","IVCBW","IVCP","IVDA","IVH","IVR","IVT","IVZ","IX","IXHL","IZEA","J","JACK","JAGX","JAKK","JAMF","JAN","JANX","JAQCW","JAZZ","JBGS","JBHT","JBI","JBL","JBLU","JBSS","JBT","JCE","JCI","JCIC","JCICW","JCSE","JCTCF","JD","JEF","JELD","JEMD","JEQ","JFIN","JFR","JFU","JG","JGGCU","JGGCW","JGH","JHAA","JHG","JHI","JHS","JHX","JILL","JJSF","JKHY","JKS","JLL","JLS","JMACW","JMIA","JMM","JMSB","JNCE","JNJ","JNPR","JOAN","JOB","JOBY","JOE","JOF","JOFF","JOFFU","JOFFW","JOUT","JPC","JPI","JPM","JPS","JPT","JQC","JRI","JRO","JRS","JRSH","JRVR","JSD","JSM","JSPR","JSPRW","JT","JUGG","JUGGW","JUN","JUPW","JUPWW","JVA","JWAC","JWEL","JWN","JWSM","JXN","JYAC","JYNT","JZXN","K","KACL","KACLR","KAHC","KAI","KAII","KAIR","KAL","KALA","KALU","KALV","KALWW","KAMN","KAR","KARO","KAVL","KB","KBAL","KBH","KBNT","KBNTW","KBR","KC","KCGI","KD","KDNY","KDP","KE","KELYA","KEN","KEP","KEQU","KERN","KERNW","KEX","KEY","KEYS","KF","KFFB","KFRC","KFS","KFY","KGC","KHC","KIDS","KIIIW","KIM","KIND","KINS","KINZ","KINZU","KINZW","KIO","KIQ","KIRK","KKR","KKRS","KLAC","KLAQ","KLAQU","KLIC","KLR","KLTR","KLXE","KMB","KMDA","KMF","KMI","KMPB","KMPH","KMPR","KMT","KMX","KN","KNBE","KNDI","KNOP","KNSA","KNSL","KNTE","KNTK","KNX","KO","KOD","KODK","KOF","KOP","KOPN","KORE","KOS","KOSS","KPLT","KPLTW","KPRX","KPTI","KR","KRBP","KRC","KREF","KRG","KRKR","KRMD","KRNL","KRNLU","KRNT","KRNY","KRO","KRON","KROS","KRP","KRT","KRTX","KRUS","KRYS","KSCP","KSM","KSPN","KSS","KT","KTB","KTCC","KTF","KTH","KTN","KTOS","KTRA","KTTA","KUKE","KULR","KURA","KVHI","KVSC","KW","KWAC","KWR","KXIN","KYCH","KYMR","KYN","KZIA","KZR","L","LAAA","LAB","LABP","LAC","LAD","LADR","LAKE","LAMR","LANC","LAND","LANDM","LANDO","LARK","LASR","LAUR","LAW","LAZ","LAZR","LAZY","LBAI","LBC","LBPH","LBRDA","LBRDK","LBRDP","LBRT","LBTYA","LBTYK","LC","LCA","LCAA","LCFY","LCFYW","LCI","LCID","LCII","LCNB","LCTX","LCUT","LCW","LDHA","LDHAW","LDI","LDOS","LDP","LE","LEA","LEAP","LECO","LEDS","LEE","LEG","LEGA","LEGH","LEGN","LEJU","LEN","LEO","LESL","LEU","LEV","LEVI","LEXX","LFAC","LFACU","LFACW","LFC","LFG","LFLY","LFLYW","LFMD","LFMDP","LFST","LFT","LFTR","LFUS","LFVN","LGAC","LGHL","LGHLW","LGI","LGIH","LGL","LGMK","LGND","LGO","LGST","LGSTW","LGTO","LGTOW","LGV","LGVN","LH","LHC","LHCG","LHDX","LHX","LI","LIAN","LIBYW","LICY","LIDR","LIDRW","LIFE","LII","LILA","LILAK","LILM","LILMW","LIN","LINC","LIND","LINK","LION","LIONW","LIQT","LITB","LITE","LITM","LITT","LIVE","LIVN","LIXT","LIZI","LJAQ","LJAQU","LJPC","LKCO","LKFN","LKQ","LL","LLAP","LLL","LLY","LMACA","LMACU","LMACW","LMAO","LMAT","LMB","LMDX","LMFA","LMND","LMNL","LMNR","LMPX","LMST","LMT","LNC","LND","LNDC","LNFA","LNG","LNN","LNSR","LNT","LNTH","LNW","LOAN","LOB","LOCL","LOCO","LODE","LOGC","LOGI","LOMA","LOOP","LOPE","LOTZ","LOTZW","LOV","LOVE","LOW","LPCN","LPG","LPI","LPL","LPLA","LPRO","LPSN","LPTH","LPTX","LPX","LQDA","LQDT","LRCX","LRFC","LRMR","LRN","LSAK","LSCC","LSEA","LSEAW","LSF","LSI","LSPD","LSTR","LSXMA","LSXMB","LSXMK","LTBR","LTC","LTCH","LTCHW","LTH","LTHM","LTRN","LTRPA","LTRX","LTRY","LTRYW","LU","LUCD","LULU","LUMN","LUMO","LUNA","LUNG","LUV","LUXA","LUXAU","LUXAW","LVAC","LVACW","LVLU","LVO","LVOX","LVRA","LVS","LVTX","LW","LWLG","LX","LXEH","LXFR","LXP","LXRX","LXU","LYB","LYEL","LYFT","LYG","LYL","LYLT","LYRA","LYT","LYTS","LYV","LZ","LZB","M","MA","MAA","MAAQ","MAAQW","MAC","MACA","MACAU","MACAW","MACC","MACK","MAG","MAIN","MAN","MANH","MANT","MANU","MAPS","MAPSW","MAQC","MAQCU","MAQCW","MAR","MARA","MARK","MARPS","MAS","MASI","MASS","MAT","MATV","MATW","MATX","MAV","MAX","MAXN","MAXR","MBAC","MBCN","MBI","MBII","MBIN","MBINN","MBINO","MBINP","MBIO","MBNKP","MBOT","MBRX","MBTCR","MBTCU","MBUU","MBWM","MC","MCAA","MCAAW","MCAC","MCB","MCBC","MCBS","MCD","MCFT","MCG","MCHP","MCHX","MCI","MCK","MCLD","MCN","MCO","MCR","MCRB","MCRI","MCS","MCW","MCY","MD","MDB","MDC","MDGL","MDGS","MDGSW","MDIA","MDJH","MDLZ","MDNA","MDRR","MDRX","MDT","MDU","MDV","MDVL","MDWD","MDWT","MDXG","MDXH","ME","MEAC","MEACW","MEC","MED","MEDP","MEDS","MEG","MEGI","MEI","MEIP","MEKA","MELI","MEOA","MEOAW","MEOH","MERC","MESA","MESO","MET","META","METC","METCL","METX","METXW","MF","MFA","MFC","MFD","MFG","MFGP","MFH","MFIN","MFM","MFV","MG","MGA","MGEE","MGF","MGI","MGIC","MGLD","MGM","MGNI","MGNX","MGPI","MGR","MGRB","MGRC","MGRD","MGTA","MGTX","MGU","MGY","MHD","MHF","MHH","MHI","MHK","MHLA","MHLD","MHN","MHNC","MHO","MHUA","MIC","MICS","MICT","MIDD","MIGI","MILE","MIMO","MIN","MIND","MINDP","MINM","MIO","MIR","MIRM","MIRO","MIST","MIT","MITC","MITK","MITO","MITQ","MITT","MIXT","MIY","MKC","MKD","MKFG","MKL","MKSI","MKTW","MKTX","ML","MLAB","MLAC","MLCO","MLI","MLKN","MLM","MLNK","MLP","MLR","MLSS","MLTX","MLVF","MMAT","MMC","MMD","MMI","MMLP","MMM","MMMB","MMP","MMS","MMSI","MMT","MMU","MMX","MMYT","MN","MNDO","MNDT","MNDY","MNKD","MNMD","MNOV","MNP","MNPR","MNRL","MNRO","MNSB","MNSBP","MNSO","MNST","MNTK","MNTS","MNTSW","MNTV","MNTX","MO","MOBQ","MOBQW","MOD","MODD","MODN","MODV","MOFG","MOGO","MOGU","MOH","MOHO","MOLN","MOMO","MON","MONCW","MOR","MORF","MORN","MOS","MOTS","MOV","MOVE","MOXC","MP","MPA","MPAA","MPACR","MPB","MPC","MPLN","MPLX","MPV","MPW","MPWR","MPX","MQ","MQT","MQY","MRAI","MRAM","MRBK","MRC","MRCC","MRCY","MREO","MRIN","MRK","MRKR","MRM","MRNA","MRNS","MRO","MRSN","MRTN","MRTX","MRUS","MRVI","MRVL","MS","MSA","MSAC","MSB","MSBI","MSC","MSCI","MSD","MSDA","MSDAW","MSEX","MSFT","MSGE","MSGM","MSGS","MSI","MSM","MSN","MSPR","MSPRW","MSPRZ","MSTR","MT","MTA","MTAC","MTACW","MTAL","MTB","MTBC","MTBCO","MTBCP","MTC","MTCH","MTCN","MTCR","MTD","MTDR","MTEK","MTEKW","MTEM","MTEX","MTG","MTH","MTLS","MTMT","MTN","MTNB","MTOR","MTP","MTR","MTRN","MTRX","MTRY","MTSI","MTTR","MTVC","MTW","MTX","MTZ","MU","MUA","MUC","MUDS","MUDSW","MUE","MUFG","MUI","MUJ","MULN","MUR","MURFW","MUSA","MUX","MVBF","MVF","MVIS","MVO","MVST","MVSTW","MVT","MWA","MX","MXC","MXCT","MXE","MXF","MXL","MYD","MYE","MYFW","MYGN","MYI","MYMD","MYN","MYNA","MYNZ","MYO","MYOV","MYPS","MYRG","MYSZ","MYTE","NAAC","NAACW","NAAS","NABL","NAC","NAD","NAII","NAK","NAN","NAOV","NAPA","NARI","NAT","NATH","NATI","NATR","NAUT","NAVB","NAVI","NAZ","NBB","NBEV","NBH","NBHC","NBIX","NBN","NBO","NBR","NBRV","NBSE","NBSTW","NBTB","NBTX","NBW","NBXG","NBY","NC","NCA","NCAC","NCACU","NCACW","NCLH","NCMI","NCNA","NCNO","NCR","NCSM","NCTY","NCV","NCZ","NDAC","NDACU","NDACW","NDAQ","NDLS","NDMO","NDP","NDRA","NDSN","NE","NEA","NECB","NEE","NEGG","NEM","NEN","NEO","NEOG","NEON","NEOV","NEP","NEPH","NEPT","NERV","NESR","NESRW","NET","NETI","NEU","NEWP","NEWR","NEWT","NEWTL","NEX","NEXA","NEXI","NEXT","NFBK","NFE","NFG","NFGC","NFJ","NFLX","NFYS","NG","NGC","NGD","NGG","NGL","NGM","NGMS","NGS","NGVC","NGVT","NH","NHC","NHI","NHIC","NHICW","NHS","NHTC","NHWK","NI","NIC","NICE","NICK","NID","NIE","NILE","NIM","NINE","NIO","NIQ","NISN","NIU","NJR","NKE","NKG","NKLA","NKSH","NKTR","NKTX","NKX","NL","NLIT","NLITU","NLITW","NLOK","NLS","NLSN","NLSP","NLSPW","NLTX","NLY","NM","NMAI","NMCO","NMFC","NMG","NMI","NMIH","NML","NMM","NMMC","NMR","NMRD","NMRK","NMS","NMT","NMTC","NMTR","NMZ","NN","NNBR","NNDM","NNI","NNN","NNOX","NNVC","NNY","NOA","NOAC","NOACW","NOAH","NOC","NODK","NOG","NOK","NOM","NOMD","NOTV","NOV","NOVA","NOVN","NOVT","NOW","NPAB","NPCE","NPCT","NPFD","NPK","NPO","NPTN","NPV","NQP","NR","NRACU","NRACW","NRBO","NRC","NRDS","NRDY","NREF","NRG","NRGV","NRGX","NRIM","NRIX","NRK","NRO","NRP","NRSN","NRSNW","NRT","NRUC","NRXP","NRXPW","NRZ","NS","NSA","NSC","NSIT","NSL","NSP","NSPR","NSR","NSS","NSSC","NSTB","NSTG","NSTS","NSYS","NTAP","NTB","NTCO","NTCT","NTES","NTG","NTGR","NTIC","NTIP","NTLA","NTNX","NTR","NTRA","NTRB","NTRBW","NTRS","NTRSO","NTST","NTUS","NTWK","NTZ","NU","NUE","NUO","NURO","NUS","NUTX","NUV","NUVA","NUVB","NUVL","NUW","NUWE","NUZE","NVACR","NVAX","NVCN","NVCR","NVCT","NVDA","NVEC","NVEE","NVEI","NVFY","NVG","NVGS","NVIV","NVMI","NVNO","NVO","NVOS","NVR","NVRO","NVS","NVSA","NVSAU","NVSAW","NVST","NVT","NVTA","NVTS","NVVE","NVVEW","NVX","NWBI","NWE","NWFL","NWG","NWL","NWLI","NWN","NWPX","NWS","NWSA","NX","NXC","NXDT","NXE","NXGL","NXGLW","NXGN","NXJ","NXN","NXP","NXPI","NXPL","NXRT","NXST","NXTC","NXTP","NYC","NYCB","NYMT","NYMTL","NYMTM","NYMTN","NYMTZ","NYMX","NYT","NYXH","NZF","O","OB","OBCI","OBE","OBLG","OBNK","OBSV","OC","OCAX","OCC","OCCI","OCCIO","OCFC","OCFT","OCG","OCGN","OCN","OCSL","OCUL","OCUP","OCX","ODC","ODFL","ODP","ODV","OEC","OEG","OEPW","OEPWW","OESX","OFC","OFG","OFIX","OFLX","OFS","OG","OGE","OGEN","OGI","OGN","OGS","OHI","OHPA","OHPAU","OHPAW","OI","OIA","OII","OIIM","OIS","OKE","OKTA","OKYO","OLB","OLED","OLIT","OLK","OLLI","OLMA","OLN","OLO","OLP","OLPX","OM","OMAB","OMC","OMCL","OMEG","OMER","OMEX","OMF","OMGA","OMI","OMIC","OMQS","ON","ONB","ONBPO","ONBPP","ONCR","ONCS","ONCT","ONCY","ONDS","ONEM","ONEW","ONL","ONON","ONTF","ONTO","ONTX","ONVO","ONYX","ONYXW","OOMA","OP","OPA","OPAD","OPBK","OPCH","OPEN","OPFI","OPGN","OPI","OPINL","OPK","OPNT","OPP","OPRA","OPRT","OPRX","OPT","OPTN","OPTT","OPY","OR","ORA","ORAN","ORC","ORCC","ORCL","ORGN","ORGNW","ORGO","ORGS","ORI","ORIC","ORLA","ORLY","ORMP","ORN","ORRF","ORTX","OSBC","OSCR","OSG","OSH","OSIS","OSK","OSPN","OSS","OST","OSTK","OSTR","OSTRU","OSTRW","OSUR","OSW","OTECW","OTEX","OTIC","OTIS","OTLK","OTLY","OTMO","OTRK","OTRKP","OTTR","OUST","OUT","OVBC","OVID","OVLY","OVV","OWL","OWLT","OXAC","OXACW","OXBR","OXBRW","OXLC","OXLCL","OXLCM","OXLCN","OXLCO","OXLCP","OXLCZ","OXM","OXSQ","OXSQG","OXSQL","OXUS","OXUSW","OXY","OYST","OZ","OZK","OZKAP","PAA","PAAS","PAC","PACB","PACI","PACK","PACW","PACWP","PACX","PACXU","PACXW","PAG","PAGP","PAGS","PAHC","PAI","PALI","PALT","PAM","PANL","PANW","PAQC","PAQCU","PAQCW","PAR","PARA","PARAA","PARAP","PARR","PASG","PATH","PATI","PATK","PAVM","PAVMZ","PAX","PAXS","PAY","PAYA","PAYC","PAYO","PAYS","PAYX","PB","PBA","PBBK","PBF","PBFS","PBFX","PBH","PBHC","PBI","PBLA","PBPB","PBR","PBT","PBTS","PBYI","PCAR","PCB","PCCT","PCF","PCG","PCGU","PCH","PCK","PCM","PCN","PCOR","PCPC","PCQ","PCRX","PCSA","PCSB","PCT","PCTI","PCTTW","PCTY","PCVX","PCX","PCYG","PCYO","PD","PDCE","PDCO","PDD","PDEX","PDFS","PDI","PDLB","PDM","PDO","PDOT","PDS","PDSB","PDT","PEAK","PEAR","PEARW","PEB","PEBK","PEBO","PECO","PED","PEG","PEGA","PEGR","PEGRU","PEGY","PEI","PEN","PENN","PEO","PEP","PEPG","PEPL","PEPLW","PERI","PESI","PETQ","PETS","PETV","PETVW","PETZ","PEV","PFBC","PFC","PFD","PFDR","PFDRW","PFE","PFG","PFGC","PFH","PFHC","PFHD","PFIE","PFIN","PFIS","PFL","PFLT","PFMT","PFN","PFO","PFS","PFSI","PFSW","PFTA","PFTAU","PFX","PFXNL","PG","PGC","PGEN","PGNY","PGP","PGR","PGRE","PGRU","PGRW","PGRWU","PGTI","PGY","PGYWW","PGZ","PH","PHAR","PHAS","PHAT","PHCF","PHD","PHG","PHGE","PHI","PHIC","PHIO","PHK","PHM","PHR","PHT","PHUN","PHUNW","PHVS","PHX","PI","PIAI","PICC","PII","PIII","PIIIW","PIK","PIM","PINC","PINE","PING","PINS","PIPP","PIPR","PIRS","PIXY","PJT","PK","PKBK","PKE","PKG","PKI","PKOH","PKX","PL","PLAB","PLAG","PLAY","PLBC","PLBY","PLCE","PLD","PLG","PLL","PLM","PLMI","PLMIU","PLMIW","PLMR","PLNT","PLOW","PLPC","PLRX","PLSE","PLTK","PLTR","PLUG","PLUS","PLX","PLXP","PLXS","PLYA","PLYM","PM","PMCB","PMD","PME","PMF","PMGM","PMGMW","PML","PMM","PMO","PMT","PMTS","PMVP","PMX","PNACU","PNBK","PNC","PNF","PNFP","PNFPP","PNI","PNM","PNNT","PNR","PNRG","PNT","PNTG","PNTM","PNW","POAI","PODD","POET","POLA","POLY","POND","PONO","PONOW","POOL","POR","PORT","POSH","POST","POW","POWI","POWL","POWRU","POWW","POWWP","PPBI","PPBT","PPC","PPG","PPIH","PPL","PPSI","PPT","PPTA","PRA","PRAA","PRAX","PRBM","PRCH","PRCT","PRDO","PRDS","PRE","PRFT","PRFX","PRG","PRGO","PRGS","PRI","PRIM","PRK","PRLB","PRLD","PRLH","PRM","PRMW","PRO","PROC","PROCW","PROF","PROV","PRPB","PRPC","PRPH","PRPL","PRPO","PRQR","PRS","PRSO","PRSR","PRSRU","PRSRW","PRT","PRTA","PRTG","PRTH","PRTK","PRTS","PRTY","PRU","PRVA","PRVB","PSA","PSAG","PSAGU","PSAGW","PSB","PSEC","PSF","PSFE","PSHG","PSMT","PSN","PSNL","PSNY","PSNYW","PSO","PSPC","PSTG","PSTH","PSTI","PSTL","PSTV","PSTX","PSX","PT","PTA","PTC","PTCT","PTE","PTEN","PTGX","PTIX","PTLO","PTMN","PTN","PTNR","PTON","PTPI","PTR","PTRA","PTRS","PTSI","PTVE","PTY","PUBM","PUCK","PUCKW","PUK","PULM","PUMP","PUYI","PV","PVBC","PVH","PVL","PW","PWFL","PWOD","PWP","PWR","PWSC","PWUPW","PX","PXD","PXLW","PXS","PXSAP","PYCR","PYN","PYPD","PYPL","PYR","PYS","PYT","PYXS","PZC","PZG","PZN","PZZA","QCOM","QCRH","QD","QDEL","QFIN","QFTA","QGEN","QH","QIPT","QK","QLGN","QLI","QLYS","QMCO","QNGY","QNRX","QNST","QQQX","QRHC","QRTEA","QRTEB","QRTEP","QRVO","QS","QSI","QSR","QTEK","QTEKW","QTNT","QTRX","QTT","QTWO","QUAD","QUBT","QUIK","QUMU","QUOT","QURE","QVCC","QVCD","R","RA","RAAS","RACE","RAD","RADA","RADI","RAIL","RAIN","RAM","RAMMU","RAMMW","RAMP","RANI","RAPT","RARE","RAVE","RBA","RBAC","RBB","RBBN","RBCAA","RBCN","RBKB","RBLX","RBOT","RC","RCA","RCAT","RCB","RCC","RCEL","RCFA","RCG","RCHG","RCHGU","RCHGW","RCI","RCII","RCKT","RCKY","RCL","RCLF","RCLFW","RCM","RCMT","RCON","RCOR","RCRT","RCRTW","RCS","RCUS","RDBX","RDBXW","RDCM","RDFN","RDHL","RDI","RDIB","RDN","RDNT","RDUS","RDVT","RDW","RDWR","RDY","RE","REAL","REAX","REE","REEAW","REED","REFI","REFR","REG","REGN","REI","REKR","RELI","RELIW","RELL","RELX","RELY","RENEU","RENEW","RENN","RENT","REPL","REPX","RERE","RES","RETA","RETO","REV","REVB","REVBW","REVEW","REVG","REVH","REVHU","REVHW","REX","REXR","REYN","REZI","RF","RFACW","RFI","RFIL","RFL","RFM","RFMZ","RFP","RGA","RGC","RGCO","RGEN","RGF","RGLD","RGLS","RGNX","RGP","RGR","RGS","RGT","RGTI","RGTIW","RH","RHE","RHI","RHP","RIBT","RICK","RIDE","RIG","RIGL","RILY","RILYG","RILYK","RILYL","RILYM","RILYN","RILYO","RILYP","RILYT","RILYZ","RIO","RIOT","RIV","RIVN","RJF","RKDA","RKLB","RKLY","RKT","RKTA","RL","RLAY","RLGT","RLI","RLJ","RLMD","RLTY","RLX","RLYB","RM","RMAX","RMBI","RMBL","RMBS","RMCF","RMD","RMED","RMGC","RMGCW","RMI","RMM","RMMZ","RMNI","RMO","RMR","RMT","RMTI","RNA","RNAZ","RNDB","RNER","RNERW","RNG","RNGR","RNLX","RNP","RNR","RNST","RNW","RNWK","RNWWW","RNXT","ROAD","ROC","ROCC","ROCGU","ROCK","ROCLU","ROCLW","ROG","ROIC","ROIV","ROIVW","ROK","ROKU","ROL","ROLL","ROLLP","RONI","ROOT","ROP","ROSE","ROSEU","ROSEW","ROSS","ROST","ROVR","RPAY","RPD","RPHM","RPID","RPM","RPRX","RPT","RPTX","RQI","RRBI","RRC","RRGB","RRR","RRX","RS","RSF","RSG","RSI","RSKD","RSLS","RSSS","RSVR","RSVRW","RTL","RTLPO","RTLPP","RTLR","RTX","RUBY","RUN","RUSHA","RUSHB","RUTH","RVAC","RVACU","RVACW","RVLP","RVLV","RVMD","RVNC","RVP","RVPH","RVPHW","RVSB","RVSN","RVT","RWAY","RWLK","RWT","RXDX","RXRX","RXST","RXT","RY","RYAAY","RYAM","RYAN","RYI","RYN","RYTM","RZA","RZB","RZLT","S","SA","SABR","SABRP","SABS","SABSW","SACC","SACH","SAFE","SAFM","SAFT","SAGA","SAGE","SAH","SAI","SAIA","SAIC","SAIL","SAITW","SAL","SALM","SAM","SAMAW","SAMG","SAN","SANA","SAND","SANM","SANW","SAP","SAR","SASI","SASR","SAT","SATL","SATLW","SATS","SAVA","SAVE","SB","SBAC","SBBA","SBCF","SBET","SBEV","SBFG","SBFM","SBFMW","SBGI","SBH","SBI","SBIG","SBII","SBLK","SBNY","SBNYP","SBOW","SBR","SBRA","SBS","SBSI","SBSW","SBT","SBTX","SBUX","SCAQU","SCCB","SCCC","SCCE","SCCF","SCCO","SCD","SCHL","SCHN","SCHW","SCI","SCKT","SCL","SCLE","SCLEU","SCLEW","SCM","SCOA","SCOAW","SCOB","SCOBU","SCOBW","SCOR","SCPH","SCPL","SCPS","SCRM","SCRMW","SCS","SCSC","SCTL","SCU","SCVL","SCWO","SCWX","SCX","SCYX","SD","SDAC","SDACU","SDACW","SDC","SDGR","SDH","SDHY","SDIG","SDPI","SE","SEAC","SEAS","SEAT","SEB","SECO","SEDG","SEE","SEED","SEEL","SEER","SEIC","SELB","SELF","SEM","SEMR","SENEA","SENS","SERA","SES","SESN","SEV","SEVN","SF","SFB","SFBS","SFE","SFET","SFIX","SFL","SFM","SFNC","SFST","SFT","SG","SGA","SGBX","SGC","SGEN","SGFY","SGH","SGHC","SGHL","SGHT","SGIIW","SGLY","SGMA","SGML","SGMO","SGRP","SGRY","SGTX","SGU","SHAC","SHAK","SHBI","SHC","SHCA","SHCAU","SHCR","SHCRW","SHEL","SHEN","SHG","SHI","SHIP","SHLS","SHLX","SHO","SHOO","SHOP","SHPW","SHQA","SHQAU","SHW","SHYF","SI","SIBN","SID","SIDU","SIEB","SIEN","SIER","SIF","SIFY","SIG","SIGA","SIGI","SIGIP","SII","SILC","SILK","SILV","SIMO","SINT","SIOX","SIRE","SIRI","SISI","SITC","SITE","SITM","SIVB","SIVBP","SIX","SJ","SJI","SJIJ","SJIV","SJM","SJR","SJT","SJW","SKE","SKIL","SKIN","SKLZ","SKM","SKT","SKX","SKY","SKYA","SKYH","SKYT","SKYW","SKYX","SLAB","SLAC","SLB","SLCA","SLCR","SLCRW","SLDB","SLDP","SLDPW","SLF","SLG","SLGC","SLGG","SLGL","SLGN","SLHG","SLHGP","SLI","SLM","SLN","SLNH","SLNHP","SLNO","SLP","SLQT","SLRC","SLRX","SLS","SLVM","SLVR","SLVRU","SM","SMAP","SMAPW","SMAR","SMBC","SMBK","SMCI","SMED","SMFG","SMFL","SMFR","SMFRW","SMG","SMHI","SMID","SMIHU","SMIT","SMLP","SMLR","SMM","SMMF","SMMT","SMP","SMPL","SMR","SMRT","SMSI","SMTC","SMTI","SMTS","SMWB","SNA","SNAP","SNAX","SNAXW","SNBR","SNCE","SNCR","SNCRL","SNCY","SND","SNDA","SNDL","SNDR","SNDX","SNES","SNEX","SNFCA","SNGX","SNMP","SNN","SNOA","SNOW","SNP","SNPO","SNPS","SNPX","SNRH","SNRHU","SNRHW","SNSE","SNT","SNTG","SNTI","SNV","SNX","SNY","SO","SOBR","SOFI","SOFO","SOHO","SOHOB","SOHON","SOHOO","SOHU","SOI","SOJC","SOJD","SOJE","SOL","SOLN","SOLO","SON","SOND","SONM","SONN","SONO","SONX","SONY","SOPA","SOPH","SOR","SOS","SOTK","SOUN","SOUNW","SOVO","SP","SPB","SPCB","SPCE","SPE","SPFI","SPG","SPGI","SPGS","SPH","SPI","SPIR","SPKB","SPKBU","SPKBW","SPLK","SPLP","SPNE","SPNS","SPNT","SPOK","SPOT","SPPI","SPR","SPRB","SPRC","SPRO","SPSC","SPT","SPTK","SPTKW","SPTN","SPWH","SPWR","SPXC","SPXX","SQ","SQFT","SQFTP","SQFTW","SQL","SQLLW","SQM","SQNS","SQSP","SQZ","SR","SRAD","SRAX","SRC","SRCE","SRCL","SRDX","SRE","SREA","SREV","SRG","SRGA","SRI","SRL","SRLP","SRNE","SRPT","SRRK","SRSA","SRT","SRTS","SRV","SRZN","SRZNW","SSAA","SSB","SSBI","SSBK","SSD","SSIC","SSKN","SSL","SSNC","SSNT","SSP","SSRM","SSSS","SST","SSTI","SSTK","SSU","SSY","SSYS","ST","STAA","STAB","STAF","STAG","STAR","STBA","STC","STCN","STE","STEM","STEP","STER","STEW","STG","STGW","STIM","STK","STKL","STKS","STLA","STLD","STM","STN","STNE","STNG","STOK","STON","STOR","STR","STRA","STRC","STRCW","STRE","STRL","STRM","STRN","STRNW","STRO","STRR","STRS","STRT","STRY","STSA","STSS","STSSW","STT","STTK","STVN","STWD","STX","STXS","STZ","SU","SUAC","SUI","SUM","SUMO","SUN","SUNL","SUNW","SUP","SUPN","SUPV","SURF","SURG","SURGW","SUZ","SVC","SVFA","SVFAU","SVFAW","SVFD","SVM","SVNAW","SVRA","SVRE","SVT","SVVC","SWAG","SWAV","SWBI","SWCH","SWET","SWETW","SWI","SWIM","SWIR","SWK","SWKH","SWKS","SWN","SWT","SWTX","SWVL","SWVLW","SWX","SWZ","SXC","SXI","SXT","SXTC","SY","SYBT","SYBX","SYF","SYK","SYM","SYN","SYNA","SYNH","SYNL","SYPR","SYRS","SYTA","SYTAW","SYY","SZC","T","TA","TAC","TACT","TAIT","TAK","TAL","TALK","TALKW","TALO","TALS","TANH","TANNI","TANNL","TANNZ","TAOP","TAP","TARA","TARO","TARS","TASK","TAST","TATT","TAYD","TBB","TBBK","TBC","TBCPU","TBI","TBK","TBKCP","TBLA","TBLD","TBLT","TBLTW","TBNK","TBPH","TC","TCBC","TCBI","TCBIO","TCBK","TCBP","TCBPW","TCBS","TCBX","TCDA","TCFC","TCI","TCMD","TCN","TCOM","TCON","TCPC","TCRR","TCRT","TCRX","TCS","TCVA","TCX","TD","TDC","TDCX","TDF","TDG","TDOC","TDS","TDUP","TDW","TDY","TEAF","TEAM","TECH","TECK","TECTP","TEDU","TEF","TEI","TEKK","TEKKU","TEL","TELA","TELL","TELZ","TEN","TENB","TENX","TEO","TER","TERN","TESS","TETC","TETCU","TETCW","TETE","TETEU","TEVA","TEX","TFC","TFFP","TFII","TFSA","TFSL","TFX","TG","TGA","TGAA","TGAN","TGB","TGH","TGI","TGLS","TGNA","TGR","TGS","TGT","TGTX","TGVC","TGVCW","TH","THACW","THC","THCA","THCP","THFF","THG","THM","THMO","THO","THQ","THR","THRM","THRN","THRX","THRY","THS","THTX","THW","THWWW","TIG","TIGO","TIGR","TIL","TILE","TIMB","TINV","TIPT","TIRX","TISI","TITN","TIVC","TIXT","TJX","TK","TKAT","TKC","TKLF","TKNO","TKR","TLGA","TLGY","TLGYW","TLIS","TLK","TLRY","TLS","TLSA","TLYS","TM","TMAC","TMBR","TMC","TMCI","TMCWW","TMDI","TMDX","TME","TMHC","TMKR","TMKRU","TMKRW","TMO","TMP","TMQ","TMST","TMUS","TMX","TNC","TNDM","TNET","TNGX","TNK","TNL","TNON","TNP","TNXP","TNYA","TOI","TOIIW","TOL","TOMZ","TOP","TOPS","TOST","TOUR","TOWN","TPB","TPC","TPG","TPGY","TPH","TPHS","TPIC","TPL","TPR","TPST","TPTA","TPTX","TPVG","TPX","TPZ","TR","TRAQ","TRC","TRCA","TRDA","TREE","TREX","TRGP","TRHC","TRI","TRIB","TRIN","TRIP","TRKA","TRMB","TRMD","TRMK","TRMR","TRN","TRNO","TRNS","TRON","TROO","TROW","TROX","TRP","TRQ","TRS","TRST","TRT","TRTL","TRTN","TRTX","TRU","TRUE","TRUP","TRV","TRVG","TRVI","TRVN","TRX","TS","TSAT","TSBK","TSCO","TSE","TSEM","TSHA","TSI","TSIB","TSLA","TSLX","TSM","TSN","TSP","TSPQ","TSQ","TSRI","TSVT","TT","TTC","TTCF","TTD","TTE","TTEC","TTEK","TTGT","TTI","TTM","TTMI","TTNP","TTOO","TTP","TTSH","TTWO","TU","TUEM","TUFN","TUP","TURN","TUSK","TUYA","TV","TVC","TVE","TVTX","TW","TWI","TWIN","TWKS","TWLO","TWLV","TWN","TWND","TWNI","TWNK","TWO","TWOA","TWOU","TWST","TWTR","TX","TXG","TXMD","TXN","TXRH","TXT","TY","TYDE","TYG","TYL","TYME","TYRA","TZOO","TZPS","TZPSW","U","UA","UAA","UAL","UAMY","UAN","UAVS","UBA","UBCP","UBER","UBFO","UBOH","UBP","UBS","UBSI","UBX","UCBI","UCBIO","UCL","UCTT","UDMY","UDR","UE","UEC","UEIC","UFAB","UFCS","UFI","UFPI","UFPT","UG","UGI","UGIC","UGP","UGRO","UHAL","UHS","UHT","UI","UIHC","UIS","UK","UKOMW","UL","ULBI","ULCC","ULH","ULTA","UMBF","UMC","UMH","UMPQ","UNAM","UNB","UNCY","UNF","UNFI","UNH","UNIT","UNM","UNMA","UNP","UNTY","UNVR","UONE","UONEK","UP","UPC","UPH","UPLD","UPS","UPST","UPTDW","UPWK","URBN","URG","URGN","URI","UROY","USA","USAC","USAK","USAP","USAS","USAU","USB","USCB","USCT","USDP","USEA","USEG","USER","USFD","USIO","USLM","USM","USNA","USPH","USWS","USWSW","USX","UTAA","UTAAW","UTF","UTG","UTHR","UTI","UTL","UTMD","UTME","UTRS","UTSI","UTZ","UUU","UUUU","UVE","UVSP","UVV","UWMC","UXIN","UZD","UZE","UZF","V","VABK","VAC","VACC","VAL","VALE","VALN","VALU","VAPO","VATE","VAXX","VBF","VBIV","VBLT","VBNK","VBTX","VC","VCEL","VCIF","VCKA","VCKAW","VCNX","VCSA","VCTR","VCV","VCXA","VCXAU","VCXAW","VCXB","VCYT","VECO","VECT","VEDU","VEEE","VEEV","VEL","VELO","VELOU","VENA","VENAR","VENAW","VEON","VERA","VERB","VERBW","VERI","VERO","VERU","VERV","VERX","VERY","VET","VEV","VFC","VFF","VFL","VG","VGFC","VGI","VGM","VGR","VGZ","VHAQ","VHC","VHI","VHNAW","VIA","VIAO","VIASP","VIAV","VICI","VICR","VIEW","VIEWW","VIGL","VINC","VINE","VINO","VINP","VIOT","VIPS","VIR","VIRC","VIRI","VIRT","VIRX","VISL","VIST","VITL","VIV","VIVE","VIVK","VIVO","VJET","VKI","VKQ","VKTX","VLAT","VLCN","VLD","VLDR","VLDRW","VLGEA","VLN","VLNS","VLO","VLON","VLRS","VLT","VLTA","VLY","VLYPO","VLYPP","VMAR","VMC","VMCAW","VMD","VMEO","VMGA","VMI","VMO","VMW","VNCE","VNDA","VNET","VNO","VNOM","VNRX","VNT","VNTR","VOC","VOD","VOR","VORB","VORBW","VOXX","VOYA","VPG","VPV","VQS","VRA","VRAR","VRAY","VRCA","VRDN","VRE","VREX","VRM","VRME","VRMEW","VRNA","VRNS","VRNT","VRPX","VRRM","VRSK","VRSN","VRT","VRTS","VRTV","VRTX","VS","VSACW","VSAT","VSCO","VSEC","VSH","VST","VSTA","VSTM","VSTO","VTAQ","VTAQW","VTEX","VTGN","VTIQ","VTIQW","VTN","VTNR","VTOL","VTR","VTRS","VTRU","VTSI","VTVT","VTYX","VUZI","VVI","VVNT","VVOS","VVPR","VVR","VVV","VWE","VWEWW","VXRT","VYGG","VYGR","VYNE","VYNT","VZ","VZIO","VZLA","W","WAB","WABC","WAFD","WAFDP","WAFU","WAL","WALD","WALDW","WARR","WASH","WAT","WATT","WAVC","WAVD","WAVE","WB","WBA","WBD","WBEV","WBS","WBT","WBX","WCC","WCN","WD","WDAY","WDC","WDFC","WDH","WDI","WDS","WE","WEA","WEAV","WEBR","WEC","WEJO","WEJOW","WEL","WELL","WEN","WERN","WES","WETF","WEX","WEYS","WF","WFC","WFCF","WFG","WFRD","WGO","WH","WHD","WHF","WHG","WHLM","WHLR","WHLRD","WHLRP","WHR","WIA","WILC","WIMI","WINA","WING","WINT","WINVR","WIRE","WISA","WISH","WIT","WIW","WIX","WK","WKEY","WKHS","WKME","WKSP","WKSPW","WLDN","WLFC","WLK","WLKP","WLMS","WLY","WM","WMB","WMC","WMG","WMK","WMPN","WMS","WMT","WNC","WNEB","WNNR","WNS","WNW","WOLF","WOOF","WOR","WORX","WOW","WPC","WPCA","WPCB","WPM","WPP","WPRT","WQGA","WRAP","WRB","WRBY","WRE","WRK","WRLD","WRN","WSBC","WSBCP","WSBF","WSC","WSFS","WSM","WSO","WSR","WST","WSTG","WTBA","WTER","WTFC","WTFCM","WTFCP","WTI","WTM","WTRG","WTRH","WTS","WTT","WTTR","WTW","WU","WULF","WVE","WVVI","WVVIP","WW","WWAC","WWACW","WWD","WWE","WWR","WWW","WY","WYNN","WYY","X","XAIR","XBIO","XBIT","XCUR","XEL","XELA","XELAP","XELB","XENE","XERS","XFIN","XFINW","XFLT","XFOR","XGN","XHR","XIN","XL","XLO","XM","XMTR","XNCR","XNET","XOM","XOMA","XOMAO","XOMAP","XOS","XOSWW","XP","XPAX","XPAXW","XPDB","XPDBU","XPDBW","XPEL","XPER","XPEV","XPL","XPO","XPOA","XPOF","XPON","XPRO","XRAY","XRTX","XRX","XSPA","XTLB","XTNT","XXII","XYF","XYL","Y","YALA","YCBD","YELL","YELP","YETI","YEXT","YGMZ","YI","YJ","YMAB","YMM","YMTX","YORW","YOTAR","YOTAW","YOU","YPF","YQ","YRD","YSG","YTEN","YTPG","YTRA","YUM","YUMC","YVR","YY","Z","ZBH","ZBRA","ZCMD","ZD","ZDGE","ZEAL","ZEN","ZENV","ZEPP","ZEST","ZETA","ZEUS","ZEV","ZG","ZGN","ZH","ZI","ZIM","ZIMV","ZING","ZINGW","ZION","ZIONL","ZIONO","ZIONP","ZIP","ZIVO","ZKIN","ZLAB","ZM","ZNH","ZNTL","ZOM","ZS","ZT","ZTEK","ZTO","ZTR","ZTS","ZUMZ","ZUO","ZVIA","ZVO","ZWRK","ZWS","ZY","ZYME","ZYNE","ZYXI"
    ]

    if ticker_value not in Valid_Ticker:
        return render(request, 'Invalid_Ticker.html', {})
    
    if number_of_days < 0:
        return render(request, 'Negative_Days.html', {})
    
    if number_of_days > 365:
        return render(request, 'Overflow_days.html', {})
    

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (USD per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')



    # ========================================== Machine Learning ==========================================

#================================prophet model============================================================


    """def predict_prophet(ticker_value, number_of_days):
        df_prophet = yf.download(tickers=ticker_value, period='1y', interval='1d')
        df_prophet.reset_index(inplace=True)
        df_prophet.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)

        # Create and fit the Prophet model
        model = Prophet()
        model.fit(df_prophet)

        # Make future dataframe for prediction
        future = model.make_future_dataframe(periods=number_of_days)
            
        # Forecasting
        forecast = model.predict(future)
        forecast_subset = forecast.tail(number_of_days)

        return forecast_subset

    # Use the function to get the forecast
    #ticker_value = 'UBER'
    #number_of_days = 20
    forecast_pro = predict_prophet(ticker_value, number_of_days)

    # Prepare data for Plotly
    pred_dict_pro = {
        "Date": forecast_pro['ds'],
        "Prediction": forecast_pro['yhat'],
        "Lower Bound": forecast_pro['yhat_lower'],
        "Upper Bound": forecast_pro['yhat_upper']
    }

    pred_df_pro = pd.DataFrame(pred_dict_pro)

    # Create Plotly figure
    pred_fig_pro = go.Figure()

    # Add prediction trace
    pred_fig_pro.add_trace(go.Scatter(
        x=pred_df_pro['Date'], 
        y=pred_df_pro['Prediction'],
        mode='lines',
        name='Prediction'
    ))

    # Add uncertainty traces
    pred_fig_pro.add_trace(go.Scatter(
        x=pred_df_pro['Date'], 
        y=pred_df_pro['Upper Bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))

    pred_fig_pro.add_trace(go.Scatter(
        x=pred_df_pro['Date'], 
        y=pred_df_pro['Lower Bound'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(0,100,80,0.2)',
        fill='tonexty',
        showlegend=False
    ))

    # Update layout
    pred_fig_pro.update_layout(
        title='Stock Market Prediction with Uncertainty Bands',
        xaxis_title='Date',
        yaxis_title='Price',
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )

    # Add rangeslider
    pred_fig_pro.update_xaxes(rangeslider_visible=True)

    # Generate the HTML for the figure
    plot_div_pred_pro = plot(pred_fig_pro, auto_open=False, output_type='div')"""

    """def predict_prophet(ticker_value, number_of_days):
        df_prophet = yf.download(tickers=ticker_value, period='1y', interval='1d')
        df_prophet.reset_index(inplace=True)
        df_prophet.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)

            # Create and fit the Prophet model
        model = Prophet()
        model.fit(df_prophet)

            # Make future dataframe for prediction
        future = model.make_future_dataframe(periods=number_of_days)
            
            # Forecasting
        forecast = model.predict(future)
        forecast_subset = forecast.tail(number_of_days)

        return forecast_subset
    forecast_pro = predict_prophet(ticker_value, number_of_days)
    
    
    #pred_dict_pro = {"Date": [], "Prediction": []}

    pred_dict_pro = {
        "Date": [],
        "Prediction": [],
        "Lower Bound": forecast_pro['yhat_lower'],
        "Upper Bound": forecast_pro['yhat_upper']
    }
    for i in range(len(forecast_pro)):
        pred_dict_pro["Date"].append(forecast_pro['ds'].iloc[i])
        pred_dict_pro["Prediction"].append(forecast_pro['yhat'].iloc[i])
    
    pred_df_pro = pd.DataFrame(pred_dict_pro)
    pred_fig_pro = go.Figure([go.Scatter(x=pred_df_pro['Date'], y=pred_df_pro['Prediction'])])
    pred_fig_pro.add_trace(go.Scatter(
        x=pred_df_pro['Date'], 
        y=pred_df_pro['Upper Bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))

    pred_fig_pro.add_trace(go.Scatter(
        x=pred_df_pro['Date'], 
        y=pred_df_pro['Lower Bound'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(0,100,80,0.2)',
        fill='tonexty',
        showlegend=False
    ))
    pred_fig_pro.update_xaxes(rangeslider_visible=True)
    pred_fig_pro.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred_pro = plot(pred_fig_pro, auto_open=False, output_type='div')"""


    
    df_prophet = yf.download(tickers=ticker_value, period='1y', interval='1d')

    # Prepare the data for Prophet
    df_prophet.reset_index(inplace=True)
    df_prophet.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)

    # Create and fit the Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Make future dataframe for prediction
    future = model.make_future_dataframe(periods=number_of_days)

    # Forecasting
    forecast = model.predict(future)
    forecast_pro = forecast.tail(number_of_days)

    # Extract predicted and actual data
    actual_data = df_prophet[['ds', 'y']]
    predicted_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Create traces for Plotly graph

    trace_actual = go.Scatter(
        x=actual_data['ds'],
        y=actual_data['y'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    )

    trace_predicted = go.Scatter(
        x=predicted_data['ds'],
        y=predicted_data['yhat'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='red')
    )

    # Traces for uncertainty intervals
    trace_uncertainty_upper = go.Scatter(
        x=predicted_data['ds'],
        y=predicted_data['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    )

    trace_uncertainty_lower = go.Scatter(
        x=predicted_data['ds'],
        y=predicted_data['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(0,100,80,0.2)',
        fill='tonexty',
        showlegend=False
    )

    trace_future_predicted = go.Scatter(
        x=predicted_data['ds'][-number_of_days:],
        y=predicted_data['yhat'][-number_of_days:],
        mode='lines',
        name='Forecasted Price',
        line=dict(color='orange')
    )

    # Combine traces
    data = [trace_actual, trace_predicted, trace_future_predicted, trace_uncertainty_upper, trace_uncertainty_lower]

    # Layout for the graph
    layout = go.Layout(
        title=f'Stock Price Prediction Using Prophet for {ticker_value}',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price (USD)'),
        showlegend=True
    )
    #paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white",

    # Create figure and add traces
    pred_fig_pro = go.Figure(data=data, layout=layout)

    # Show the figure
    pred_fig_pro.update_xaxes(rangeslider_visible=True)
    pred_fig_pro.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred_pro = plot(pred_fig_pro, auto_open=False, output_type='div')


    pred_dict_pro = {
        "Date": [],
        "Prediction": [],
        "Lower Bound": forecast_pro['yhat_lower'],
        "Upper Bound": forecast_pro['yhat_upper']
    }
    for i in range(len(forecast_pro)):
        pred_dict_pro["Date"].append(forecast_pro['ds'].iloc[i])
        pred_dict_pro["Prediction"].append(forecast_pro['yhat'].iloc[i])
    
    pred_df_pro = pd.DataFrame(pred_dict_pro)
    pred_fig_pro_f = go.Figure([go.Scatter(x=pred_df_pro['Date'], y=pred_df_pro['Prediction'],
        line=dict(color='orange'), name='Forecasted Price')])
    pred_fig_pro_f.add_trace(go.Scatter(
        x=pred_df_pro['Date'], 
        y=pred_df_pro['Upper Bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))

    pred_fig_pro_f.add_trace(go.Scatter(
        x=pred_df_pro['Date'], 
        y=pred_df_pro['Lower Bound'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(0,100,80,0.2)',
        fill='tonexty',
        showlegend=False
    ))
    pred_fig_pro_f.update_xaxes(rangeslider_visible=True)
    pred_fig_pro_f.update_layout(title=f'Forecasted Price of {ticker_value} for next {number_of_days} Using Prophet', paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred_pro_f = plot(pred_fig_pro_f, auto_open=False, output_type='div')


    # Merge actual and predicted data
    evaluation_df = pd.merge(actual_data, predicted_data, on='ds', how='inner')

    # Calculate MAE, MSE, and RMSE
    pro_mae = mean_absolute_error(evaluation_df['y'], evaluation_df['yhat'])
    pro_mse = mean_squared_error(evaluation_df['y'], evaluation_df['yhat'])
    pro_rmse = sqrt(pro_mse)

   


        #===================lstm model==================================================

    """df_ml = yf.download(tickers=ticker_value, period='3mo', interval='1h')
    df_ml = df_ml[['Adj Close']]
    forecast_out = int(number_of_days)
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)

    # Split data into training and testing sets
    X = np.array(df_ml.drop(['Prediction'], axis = 1))
    X = preprocessing.scale(X)

    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]

    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Reshape the input data for LSTM model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Reshape the input data for evaluation
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Evaluate the model on the testing data
    loss = model.evaluate(X_test, y_test)

    # Make predictions on the forecast data
    X_forecast = np.reshape(X_forecast, (X_forecast.shape[0], X_forecast.shape[1], 1))
    forecast_prediction = model.predict(X_forecast)
    forecast = forecast_prediction.flatten().tolist()

    # Generate dates for the forecast
    last_date = df_ml.index[-1]
    forecast_dates = [last_date + dt.timedelta(days=i) for i in range(1, number_of_days + 1)]

    # Create a dictionary for the predictions
    pred_dict = {"Date": forecast_dates, "Prediction": forecast}
    pred_df = pd.DataFrame(pred_dict)

    # Plot the predictions
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')"""

    """pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast_lstm)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast_lstm[i])
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')"""



    # Define start and end dates
    START = "2022-02-01"
    TODAY = datetime.now().strftime("%Y-%m-%d")
    ticker = ticker_value

    # Load data from Yahoo Finance
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    df = load_data(ticker)


    # Preprocess the data
    df_close = df[['Date', 'Close']].copy()
    df_close['Date'] = pd.to_datetime(df_close['Date'])
    df_close.set_index('Date', inplace=True)

    # Convert the dataframe to a numpy array
    dataset = df_close.values

    # Get the number of rows to train the model on (80% of the dataset)
    training_data_len = int(np.ceil(len(dataset) * .8))

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(100, len(train_data)):
        x_train.append(train_data[i-100:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=5)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 100:, :]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(100, len(test_data)):
        x_test.append(test_data[i-100:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Forecast the next 10 days
    forecast_days = number_of_days
    forecast = []
    current_batch = scaled_data[-100:].reshape((1, 100, 1))

    current_batch = scaled_data[-100:].reshape((1, 100, 1))

    for i in range(forecast_days):
        # Get the next prediction
        current_pred = model.predict(current_batch)[0]

        # Append the prediction
        forecast.append(current_pred)

        # Use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # Inverse transform to get the actual forecasted prices
    forecast = scaler.inverse_transform(forecast)

    # Create a new DataFrame for the forecasted values with the corresponding dates
    forecast_dates = [df_close.index[-1] + timedelta(days=i) for i in range(1, forecast_days+1)]
    forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecast'])

    # Plot the results using Plotly
    pred_fig = go.Figure()
    # Plot the forecasted prices
    pred_fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecasted Price', line=dict(color='orange')))
    
    # Plot the validation predictions
    pred_fig.add_trace(go.Scatter(x=df_close.index[training_data_len:], y=predictions[:,0], mode='lines', name='Predicted Price', line=dict(color='red')))
    # Plot historical close prices
    pred_fig.add_trace(go.Scatter(x=df_close.index, y=df_close['Close'], mode='lines', name='Actual Price', line=dict(color='blue')))

    

    # Update the layout
    pred_fig.update_layout(xaxis_title='Date', yaxis_title='Price (USD)', showlegend=True)
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    # Show the plot
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    pred_fig_f = go.Figure()
    pred_fig_f.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecasted Price', line=dict(color='orange')))
    pred_fig_f.update_layout(title=f'Forecasted Price of {ticker_value} for next {number_of_days} Using Lstm ', xaxis_title='Date', yaxis_title='Price (USD)', showlegend=True)
    pred_fig_f.update_xaxes(rangeslider_visible=True)
    pred_fig_f.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred_f = plot(pred_fig_f, auto_open=False, output_type='div')

    l_mae = mean_absolute_error(y_test, predictions)
    l_mse = mean_squared_error(y_test, predictions)
    l_rmse = sqrt(l_mse)








        #=================arima model=================================================

    df = yf.download(tickers=ticker_value, period='1y', interval='1d')
    df_sarima = df['Adj Close'].dropna()

    # Ensure the index is a DatetimeIndex with a frequency for SARIMA
    df_sarima.index = pd.DatetimeIndex(df_sarima.index).to_period('B')

    # Fit SARIMA model (these parameters are placeholders, you need to find the best ones for your data)
    # The seasonal_order tuple below should be filled with (P, D, Q, m) values
    seasonal_order = (1, 1, 1, 5)  # Example seasonal order, may need to be fine-tuned for your data
    model = SARIMAX(df_sarima, order=(1, 1, 1), seasonal_order=seasonal_order)
    results = model.fit()

    # In-sample prediction, skipping the first 2 months (approximately 42 days)
    skip_days = 6
    df_sarima_predicted = results.predict(start=skip_days, end=len(df_sarima) - 1)

    # Forecast future values
    forecast_results = results.get_forecast(steps=number_of_days)
    forecast_index = pd.date_range(df_sarima.index[-1].to_timestamp(), periods=number_of_days, freq='B')
    forecast = forecast_results.predicted_mean

    # Plot actual, predicted, and forecasted prices
    pred_df_arima = go.Figure()

    # Forecasted prices for the next number of days
    pred_df_arima.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecasted Price', line=dict(color='orange')))

    # Predicted prices for the historical period, starting after 2 months
    predicted_index = df_sarima.index[skip_days:].to_timestamp()
    pred_df_arima.add_trace(go.Scatter(x=predicted_index, y=df_sarima_predicted, mode='lines', name='Predicted Price', line=dict(color='red')))

    # Actual prices
    pred_df_arima.add_trace(go.Scatter(x=df_sarima.index.to_timestamp(), y=df_sarima, mode='lines', name='Actual Price', line=dict(color='blue')))


    # Update layout
    pred_df_arima.update_layout(title=f'Stock Price Prediction Using ARIMA for {ticker_value}',
                    xaxis_title='Date',
                    yaxis_title='Price',
            )

    #pred_df_arima.show()

    
    pred_df_arima.update_xaxes(rangeslider_visible=True)
    pred_df_arima.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred_arima = plot(pred_df_arima, auto_open=False, output_type='div')

    pred_df_arima_f = go.Figure()
    pred_df_arima_f.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecasted Price', line=dict(color='orange')))
    pred_df_arima_f.update_layout(title=f'Forecasted Price of {ticker_value} for next {number_of_days} Using Arima ',
                    xaxis_title='Date',
                    yaxis_title='Price',
            )
    pred_df_arima_f.update_xaxes(rangeslider_visible=True)
    pred_df_arima_f.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred_arima_f = plot(pred_df_arima_f, auto_open=False, output_type='div')

    df_sarima = df_sarima[df_sarima.index.isin(df_sarima_predicted.index)]

    # Calculate the errors
    a_mae = mean_absolute_error(df_sarima, df_sarima_predicted)
    a_mse = mean_squared_error(df_sarima, df_sarima_predicted)
    a_rmse = sqrt(a_mse)


    


    # ========================================== Display Ticker Info ==========================================

    ticker = pd.read_csv('app/Data/Tickers.csv')
    to_search = ticker_value
    ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                    'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
    for i in range(0,ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            Last_Sale = ticker.Last_Sale[i]
            Net_Change = ticker.Net_Change[i]
            Percent_Change = ticker.Percent_Change[i]
            Market_Cap = ticker.Market_Cap[i]
            Country = ticker.Country[i]
            IPO_Year = ticker.IPO_Year[i]
            Volume = ticker.Volume[i]
            Sector = ticker.Sector[i]
            Industry = ticker.Industry[i]
            break

    # ========================================== Page Render section ==========================================
    

    return render(request, "result.html", context={ 'plot_div': plot_div, 
                                                    'confidence' : 0.75,
                                                    #'forecast_lstm': forecast_lstm,
                                                    'ticker_value':ticker_value,
                                                    'number_of_days':number_of_days,
                                                    'plot_div_pred':plot_div_pred,
                                                    'plot_div_pred_f':plot_div_pred_f,
                                                    'plot_div_pred_arima': plot_div_pred_arima,
                                                    'plot_div_pred_arima_f': plot_div_pred_arima_f,
                                                    'plot_div_pred_pro' :plot_div_pred_pro,
                                                    'plot_div_pred_pro_f':plot_div_pred_pro_f,
                                                    'pro_mae': pro_mae,
                                                    'pro_mse': pro_mse,
                                                    'pro_rmse' : pro_rmse,
                                                    'a_rmse': a_rmse,
                                                    'a_mse': a_mse,
                                                    'a_mae': a_mae,
                                                    'l_rmse':l_rmse,
                                                    'l_mse':l_mse,
                                                    'l_mae': l_mae,
                                                    'Symbol':Symbol,
                                                    'Name':Name,
                                                    'Last_Sale':Last_Sale,
                                                    'Net_Change':Net_Change,
                                                    'Percent_Change':Percent_Change,
                                                    'Market_Cap':Market_Cap,
                                                    'Country':Country,
                                                    'IPO_Year':IPO_Year,
                                                    'Volume':Volume,
                                                    'Sector':Sector,
                                                    'Industry':Industry,
                                                    })


#============================================================sentimental analysis=============================================================



'''def clean_tweet(tweet_text):
    cleaned_text = p.clean(tweet_text)
    cleaned_text = re.sub('&amp;', '&', cleaned_text)
    cleaned_text = re.sub(':', '', cleaned_text)
    cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')
    return cleaned_text'''

"""def analyze_sentiment(tweet):

    return polarity"""



"""def recommending(symbol, today_stock, mean, global_polarity):
    if today_stock.iloc[-1]['Close'] < mean:
        idea = "RISE"
        decision = "BUY"
    else:
        idea = "FALL"
        decision = "SELL"

    return idea, decision"""





"""def sentimental(request):

    #symbol = request.POST.get('ticker')
    symbol = 'AAPL'

    #ct = settings.TWITTER_API_CREDENTIALS
    stock_data = pd.read_csv('app\Data\Yahoo-Finance-Ticker-Symbols.csv')
    #mean_value = stock_data['Close'].mean()
    stock_ticker_map = pd.read_csv('app\Data\Yahoo-Finance-Ticker-Symbols.csv')
    stock_full_form = stock_ticker_map[stock_ticker_map['Ticker'] == symbol]
    symbol = stock_full_form['Name'].to_list()[0][0:12]

    #ct = settings.TWITTER_API_CREDENTIALS
    consumer_key= 'gqCRnO1hyxPqweyTs36VVC0s1'
    consumer_secret= 'sCm02LX8NvWnM8JdPORMxh9UNhIaWfEaafhbMbHI82DevtqW46'
    access_token='1446783631146373121-zP4qERHsnHTSRZBw2kpnm1K8QOf5ey'
    access_token_secret='uG2Bd7KtZymOeIv40k762PYparDiNWbRIY21RggrK9mmk'
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        user = tweepy.API(auth)

        #tweets = tweepy.Cursor(user.search_tweets, q=symbol, tweet_mode='extended', lang='en', exclude_replies=True).items(20)
        tweets = user.user_timeline(screen_name=symbol, count=20, tweet_mode='extended', lang='en', exclude_replies=True)


        tweet_list = []  # List of tweets alongside polarity
        global_polarity = 0  # Polarity of all tweets === Sum of polarities of individual tweets
        tw_list = []  # List of tweets only => to be displayed on the web page
        pos = 0  # Num of positive tweets
        neg = 1  # Num of negative tweets
        for tweet in tweets:
            tw2 = tweet.full_text
            tw = tweet.full_text
            tw = clean_tweet(tw)
            tw = tw.encode('ascii', 'ignore').decode('ascii')

            blob = TextBlob(tw)
            polarity = sum(sentence.sentiment.polarity for sentence in blob.sentences)

            #polarity = analyze_sentiment(tw)
            global_polarity += polarity

            if polarity > 0:
                pos += 1
            elif polarity < 0:
                neg += 1

            tw_list.append(tw2)
            tweet_list.append({'text': tw, 'polarity': polarity})

        if len(tweet_list) != 0:
            global_polarity = global_polarity / len(tweet_list)

        neutral = 20 - pos - neg
        if neutral < 0:
            neg += neutral
            neutral = 20
    except NameError as e:
        print("Error: {e}")

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg, neutral]
    colors = ['rgb(0,255,0)', 'rgb(255,0,0)', 'rgb(128,128,128)']

    
    pie_chart = go.Figure([go.Pie(labels=labels, values=sizes, marker=dict(colors=colors))])
    pie_chart.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    # Convert the Plotly chart to HTML
    chart_html = pie_chart.to_html(full_html=False)

    tw_pol = "Overall Positive" if global_polarity > 0 else "Overall Negative"

    
    

    return render(request, 'results.html', context={'tw_list': tw_list, 
                                                    'tw_pol': tw_pol,
                                                    'pos': pos, 
                                                    'neg': neg, 
                                                    'neutral': neutral,
                                                    #'idea': idea, 
                                                    #'decision': decision
                                                    })"""













# views.py

"""from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse

import pandas as pd
import tweepy
import matplotlib.pyplot as plt
from textblob import TextBlob
import preprocessor as p
import re

def clean_tweet(tweet_text):
    cleaned_text = p.clean(tweet_text)
    cleaned_text = re.sub('&amp;', '&', cleaned_text)
    cleaned_text = re.sub(':', '', cleaned_text)
    cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')
    return cleaned_text

def analyze_sentiment(tweet):
    blob = TextBlob(tweet)
    polarity = sum(sentence.sentiment.polarity for sentence in blob.sentences)
    return polarity

def retrieve_tweets(symbol):
    stock_ticker_map = pd.read_csv('path/to/your/Yahoo-Finance-Ticker-Symbols.csv')
    stock_full_form = stock_ticker_map[stock_ticker_map['Ticker'] == symbol]
    symbol = stock_full_form['Name'].to_list()[0][0:12]

    auth = tweepy.OAuthHandler(settings.TWITTER_API_CREDENTIALS['consumer_key'],
                               settings.TWITTER_API_CREDENTIALS['consumer_secret'])
    auth.set_access_token(settings.TWITTER_API_CREDENTIALS['access_token'],
                          settings.TWITTER_API_CREDENTIALS['access_token_secret'])
    user = tweepy.API(auth)

    tweets = tweepy.Cursor(user.search_tweets, q=symbol, tweet_mode='extended', lang='en', exclude_replies=True).items(20)

    tweet_list = []  
    global_polarity = 0  
    tw_list = []  
    pos = 0  
    neg = 0

    for tweet in tweets:
        tw2 = tweet.full_text
        tw = tweet.full_text
        tw = clean_tweet(tw)
        tw = tw.encode('ascii', 'ignore').decode('ascii')

        polarity = analyze_sentiment(tw)
        global_polarity += polarity

        if polarity > 0:
            pos += 1
        elif polarity < 0:
            neg += 1

        tw_list.append(tw2)
        tweet_list.append({'text': tw, 'polarity': polarity})

    if tweet_list:
        global_polarity = global_polarity / len(tweet_list)

    neutral = 20 - pos - neg
    if neutral < 0:
        neg += neutral
        neutral = 20

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg, neutral]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=65)
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.tight_layout()
    plt.savefig('path/to/your/static/SA.png')
    plt.close(fig)

    tw_pol = "Overall Positive" if global_polarity > 0 else "Overall Negative"

    return global_polarity, tw_list, tw_pol, pos, neg, neutral

def recommending(symbol, today_stock, mean, global_polarity):
    if today_stock.iloc[-1]['Close'] < mean:
        idea = "RISE"
        decision = "BUY"
    else:
        idea = "FALL"
        decision = "SELL"

    return idea, decision

def analyze_stock(request, symbol):
    try:
        stock_data = pd.read_csv('path/to/your/Yahoo-Finance-Ticker-Symbols.csv')

        global_polarity, tw_list, tw_pol, pos, neg, neutral = retrieve_tweets(symbol)

        # Calculate mean_value based on stock_data (replace this with your actual logic)
        mean_value = stock_data['Close'].mean()

        idea, decision = recommending(symbol, stock_data, mean_value, global_polarity)

        return render(request, 'results.html', {'tw_list': tw_list, 'tw_pol': tw_pol,
                                                'pos': pos, 'neg': neg, 'neutral': neutral,
                                                'idea': idea, 'decision': decision})
    except Exception as e:
        return JsonResponse({'error': str(e)})"""



"""def sentimental(request):
    #symbol = 'AAPL'

    consumer_key= 'gqCRnO1hyxPqweyTs36VVC0s1'
    consumer_secret= 'sCm02LX8NvWnM8JdPORMxh9UNhIaWfEaafhbMbHI82DevtqW46'
    access_token='1446783631146373121-zP4qERHsnHTSRZBw2kpnm1K8QOf5ey'
    access_token_secret='uG2Bd7KtZymOeIv40k762PYparDiNWbRIY21RggrK9mmk'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    user = tweepy.API(auth)

        # Fetch user timeline instead of searching tweets
    #tweets = user.user_timeline(screen_name=symbol, count=20, tweet_mode='extended', lang='en', exclude_replies=True)
    tweets = tweepy.Cursor(user.search_tweets, q=symbol, tweet_mode='extended', lang='en', exclude_replies=True).items(20)


    query = "(from:stocks_AAPL) until:2023-01-01 since:2018-01-01"
    tweets = []
    limit = 50


    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
        if len(tweets) == limit:
            break
    else:
        tweets.append([tweet.date, tweet.username, tweet.content])


    tweet_list = []  # List of tweets alongside polarity
    global_polarity = 0  # Polarity of all tweets === Sum of polarities of individual tweets
    tw_list = []  # List of tweets only => to be displayed on the web page
    pos = 0  # Num of positive tweets
    neg = 1  # Num of negative tweets
    for tweett in tweets:
        tw2 = tweett.full_text
        tw = tweett.full_text
        tw = clean_tweet(tw)
        tw = tw.encode('ascii', 'ignore').decode('ascii')

    blob = TextBlob(tw)
    polarity = sum(sentence.sentiment.polarity for sentence in blob.sentences)

    global_polarity += polarity

    if polarity > 0:
        pos += 1
    elif polarity < 0:
        neg += 1

    tw_list.append(tw2)
    tweet_list.append({'text': tw, 'polarity': polarity})

    if len(tweet_list) != 0:
        global_polarity = global_polarity / len(tweet_list)

    neutral = 20 - pos - neg
    if neutral < 0:
        neg += neutral
        neutral = 20    

    
    
        # Handle the error gracefully, log, or display a user-friendly message

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg, neutral]
    colors = ['rgb(0,255,0)', 'rgb(255,0,0)', 'rgb(128,128,128)']

    pie_chart = go.Figure([go.Pie(labels=labels, values=sizes, marker=dict(colors=colors))])
    pie_chart.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    chart_html = pie_chart.to_html(full_html=False)

    tw_pol = "Overall Positive" if global_polarity > 0 else "Overall Negative"

    return render(request, 'sentimental.html', context={'tw_list': tw_list,
                                                    'tw_pol': tw_pol,
                                                    'pos': pos,
                                                    'neg': neg,
                                                    'neutral': neutral,
                                                    'chart_html': chart_html
                                                    })"""




"""def sentimental(request):
    # Assuming you have a pre-collected dataset stored in a variable named 'tweets'
    tweets = [
        "I love this company!",
        "The stock market is unpredictable today.",
        "Earnings report looks promising.",
        # ... more tweets
    ]

    tweet_list = []  # List of tweets alongside polarity
    global_polarity = 0  # Polarity of all tweets === Sum of polarities of individual tweets
    tw_list = []  # List of tweets only => to be displayed on the web page
    pos = 0  # Num of positive tweets
    neg = 0  # Num of negative tweets

    for tweet in tweets:
        analysis = TextBlob(tweet)
        polarity = analysis.sentiment.polarity
        global_polarity += polarity

        if polarity > 0:
            pos += 1
        elif polarity < 0:
            neg += 1

        tweet_list.append((tweet, polarity))

    # Calculate the overall sentiment of all tweets
    if global_polarity > 0:
        overall_sentiment = "Positive"
    elif global_polarity < 0:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"
    

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg]
    colors = ['rgb(0,255,0)', 'rgb(255,0,0)', 'rgb(128,128,128)']

    pie_chart = go.Figure([go.Pie(labels=labels, values=sizes, marker=dict(colors=colors))])
    pie_chart.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    chart_html = pie_chart.to_html(full_html=False)

    context = {
        'tweet_list': tweet_list,
        'overall_sentiment': overall_sentiment,
        'pos': pos,
        'neg': neg,
        'chart_html': chart_html
    }

    return render(request, 'sentimental.html', context)"""

"""from nltk.sentiment import SentimentIntensityAnalyzer

def calculate_sentiment_score(tweet):
    # Instantiate the SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Get the sentiment scores for the tweet
    sentiment_scores = sid.polarity_scores(tweet)

    # Extract the compound score which represents the overall sentiment
    sentiment_score = sentiment_scores['compound']

    # Return the sentiment score
    return sentiment_score


def sentimental(request):
    symbol = 'AAPL'  # Default symbol, you can uncomment the line below to get the symbol from the request
    # symbol = request.POST.get('ticker')

    stock_data = pd.read_csv('app/Data/Yahoo-Finance-Ticker-Symbols.csv')
    stock_ticker_map = pd.read_csv('app/Data/Yahoo-Finance-Ticker-Symbols.csv')
    stock_full_form = stock_ticker_map[stock_ticker_map['Ticker'] == symbol]
    symbol = stock_full_form['Name'].to_list()[0][0:12]

    consumer_key = 'gqCRnO1hyxPqweyTs36VVC0s1'
    consumer_secret = 'sCm02LX8NvWnM8JdPORMxh9UNhIaWfEaafhbMbHI82DevtqW46'
    access_token = '1446783631146373121-zP4qERHsnHTSRZBw2kpnm1K8QOf5ey'
    access_token_secret = 'uG2Bd7KtZymOeIv40k762PYparDiNWbRIY21RggrK9mmk'

    # Configure Twint to scrape tweets related to the stock symbol
    c = twint.Config()
    c.Search = symbol
    c.Lang = 'en'
    c.Limit = 10  # Number of tweets to scrape
    c.Store_csv = True
    c.Output = 'app/Data/tweets.csv'  # Output file for saving scraped tweets
    twint.run.Search(c)

    # Read the scraped tweets from the CSV file
    tweets = pd.read_csv('app/Data/tweets.csv')

    # Perform sentiment analysis on the tweets
    # Add your code for sentiment analysis here

    # Example: Calculate the sentiment score for each tweet
    sentiment_scores = []

    for tweet in tweets['tweet']:
        # Add your sentiment analysis code here to calculate the sentiment score for each tweet
        sentiment_scores.append(calculate_sentiment_score(tweet))

    # Add the sentiment scores to the tweets dataframe
    tweets['sentiment_score'] = sentiment_scores

    # Calculate the overall sentiment score for the stock
    overall_sentiment_score = tweets['sentiment_score'].mean()

    # Pass the tweets and overall sentiment score to the template
    context = {
        'tweets': tweets,
        'overall_sentiment_score': overall_sentiment_score
    }

    return render(request, 'sentimental.html', context)"""

#-------------------------------------Reddit------------------------------------------------------------------------------------------------



def get_top_headlines(stock_symbol):
    api_key_news = config('api_key_news')  # Replace with your News API key
    url = f'https://newsapi.org/v2/top-headlines?q={stock_symbol}&apiKey={api_key_news}'
    response = requests.get(url)
    data = response.json()
    return data['articles']

    # Load the sentiment analysis model
    #classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')


    """def get_sentiment(text):
        result = classifier(text)[0]
        return result['label']"""


def get_reddit_data(stock_symbol):
    reddit = Reddit(client_id=config('client_id'), client_secret=config('client_secret'), user_agent=config('user_agent'))
    posts = reddit.subreddit('StockMarket').search(stock_symbol, limit=1000)
    return posts

 

"""def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'POSITIVE'
    elif analysis.sentiment.polarity == 0:
        return 'NEUTRAL'
    else:
        return 'NEGATIVE'  
"""
    

def sentimental(request):
    #symbol = request.POST.get('ticker')
    stock_symbol = request.POST.get('ticker','AAPL')  # Default to 'AAPL' if no symbol is provided
    reddit_posts = get_reddit_data(stock_symbol)
    dipost = get_reddit_data(stock_symbol)
    posts_to_display = list(dipost)[:10] # Limit the number of posts to display to 100



    """roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    token = RobertaTokenizer.from_pretrained(roberta)
    mode = RobertaForSequenceClassification.from_pretrained(roberta)

    def get_sentiment(text):
        labels = ['Positive', 'Negative', 'Neutral']
        encoded_text = token(text, return_tensors='pt')
        output = mode(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sentiment = labels[np.argmax(scores)]
        return sentiment"""  
    



    # Load pre-trained model and tokenizer
    tokenizer = XLNetTokenizer.from_pretrained('minh21/XLNet-Reddit-Sentiment-Analysis')
    model = XLNetForSequenceClassification.from_pretrained('minh21/XLNet-Reddit-Sentiment-Analysis')

    def get_sentiment(text):
        # Encode the text
        inputs = tokenizer.encode_plus(text, return_tensors='pt')

        # Get model outputs
        outputs = model(**inputs)

        # Get sentiment prediction
        sentiment = torch.argmax(outputs[0]).item()

        if sentiment == 2:
            return 'Positive'
        elif sentiment == 1:
            return 'Neutral'
        else:
            return 'Negative'


    
    sentiments =  {'Positive': 0, 'Negative': 0, 'Neutral': 0}  #{'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    for post in reddit_posts:
        sentiment = get_sentiment(post.title)
        sentiments[sentiment] += 1

    # Pie chart
    labels = list(sentiments.keys())
    values = list(sentiments.values())
    colors = ['green', 'red', 'grey']  # Colors for 'Positive', 'Negative', 'Neutral'
    pie_chart = go.Figure([go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
    pie_chart.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    chart_html = pie_chart.to_html(full_html=False)

#--------------------------------------twitter---------------------------------
    







    consumer_key = config("consumer_key")
    consumer_secret = config("consumer_secret")
    access_token = config("access_token")
    access_token_secret = config("access_token_secret")

        # setup Twitter API
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = API(auth)

        # setup RoBERTa model for sentiment analysis
    roberta_model = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta_model)
    tokenizer = AutoTokenizer.from_pretrained(roberta_model)

    '''def get_sentiment_t(tweet):
        labels = ['Negative', 'Neutral', 'Positive']
        encoded_tweet = tokenizer(tweet, return_tensors='pt')
        output = model(**encoded_tweet)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sentiment_t = labels[np.argmax(scores)]
        return sentiment_t'''
    

    custom_tweets = [
            "Great news for Apple stock today!",
            "I think Apple is overvalued right now.",
            "Just bought some Apple stock. Excited!",
            "Apple's latest product launch was disappointing.",
            "I'm bullish on Apple's future.",
            "Apple's earnings report exceeded expectations.",
            "The competition is catching up to Apple.",
            "I'm considering selling my Apple shares.",
            "Apple's new iPhone is a game-changer.",
            "I'm not confident in Apple's management."
        ]
    
    # Perform sentiment analysis on the custom tweets
    sentiment_labels = ['Positive', 'Negative', 'Neutral' ]
    tweet_sentiments = []
    for tweet in custom_tweets:
        inputs = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        predictions = softmax(outputs.logits.detach().numpy(), axis=1)
        sentiment_t = sentiment_labels[np.argmax(predictions)]
        tweet_sentiments.append(sentiment_t)

    

    #public_tweets = api.search_tweets(stock_symbol).data
    #tweet_sentiments = [get_sentiment_t(tweet.text) for tweet in public_tweets]
    #tweet_sentiments = [get_sentiment_t(tweet) for tweet in custom_tweets]
    
    sentiment_values = [tweet_sentiments.count(label) for label in sentiment_labels]

        # Deploying the pie chart with custom colors and layout
    colors = ['green', 'red', 'grey']  # Colors for 'Positive', 'Negative', 'Neutral'
    pie_chart = go.Figure([go.Pie(labels=sentiment_labels, values=sentiment_values)])
    pie_chart.update_traces(marker=dict(colors=colors))
    pie_chart.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    chart_html_t = pie_chart.to_html(full_html=False)


          # newsapi      d855ab93b361433a843a0e4ff83136f9
    headlines = get_top_headlines(stock_symbol)






    # Reddit sentiments
    reddit_sentiments =  {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    for post in reddit_posts:
        sentiment = get_sentiment(post.title)
        reddit_sentiments[sentiment] += 1

    # Twitter sentiments
    #sentiment_labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL' ]
    twitter_sentiments = [tweet_sentiments.count(label) for label in sentiment_labels]
    twitter_sentiments_dict = dict(zip(sentiment_labels, twitter_sentiments))

    # Line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(labels), y=list(values),
                             mode='lines+markers', name='Reddit'))
    fig.add_trace(go.Scatter(x=list(twitter_sentiments_dict.keys()), y=list(twitter_sentiments_dict.values()),
                             mode='lines+markers', name='Twitter'))

    fig.update_layout(title='Sentiment Analysis Comparison',
                      xaxis_title='Sentiment',
                      yaxis_title='Count',
                      paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    line_chart_html = fig.to_html(full_html=False)












    return render(request, 'sentimental.html', context={'chart_html': chart_html,
                                                        'stock_symbol': stock_symbol,
                                                        'posts_to_display': posts_to_display,
                                                        'chart_html_t' : chart_html_t,
                                                        'custom_tweets' : custom_tweets,
                                                        'headlines': headlines,
                                                        'line_chart_html': line_chart_html
                                                        
                                                    })
    



#----------------------------pattern analyzer---------------------------------------------------------------------------------------



"""
def get_stock_data(symbol, period="12mo"):
    data = yf.download(symbol, period=period, interval='1d')
    return data

def analyze_patterns(symbol, pattern):
    data = get_stock_data(symbol)
    open = data['Open'].values
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    pattern_func = getattr(talib, pattern.upper())
    pattern_result = pattern_func(open, high, low, close)
    return pattern_result


def pattern(request):


    
    symbol = request.POST.get('ticker','AAPL')
    pattern = request.POST.get('xpattern','CDLHAMMER')
    data = get_stock_data(symbol)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close']))
    fig.update_layout(title=symbol, paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)

    # Add box annotation for pattern
    pattern_result = analyze_patterns(symbol, pattern)
    for i in range(len(pattern_result)):
        if pattern_result[i] > 0:
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=data.index[i-15] if i-15 >= 0 else data.index[0],
                y0=0,
                x1=data.index[i],
                y1=1,
                fillcolor="rgba(255, 255, 255, 0.2)",
                line=dict(color="white", width=1),
                layer="below"
            )


            fig.add_annotation(
                x=data.index[i],
                y=data['Low'][i],
                ax=data.index[i],
                ay=data['Low'][i-10] if i-10 >= 0 else data['Low'][0],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="white"
            )


    fig_html = fig.to_html(full_html=False)

















    return render(request, 'pattern.html', context={
                                                    'fig_html': fig_html,
                                                    'stock_symbol': symbol,
                                                    'pattern': pattern







    })

"""


def get_stock_data(symbol, period="1y", interval='1d'):  # Increase period and add interval parameter
    data = yf.download(symbol, period=period, interval=interval)
    return data
'''
def analyze_patterns(symbol, pattern, period="5y", interval='1d'):  # Add period and interval parameters
    data = get_stock_data(symbol, period, interval)
    open = data['Open'].values
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    pattern_func = getattr(talib, pattern.upper())
    pattern_result = pattern_func(open, high, low, close)
    return pattern_result
'''



def get_candlestick_image(data):
    fig, ax = plt.subplots()
    mpf.plot(data, type='candle', ax=ax, returnfig=True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img




def is_significant(data, i, volume_threshold=1000000):
    # Check if the trading volume is above the threshold
    if data['Volume'][i] > volume_threshold:
        return True
    else:
        return False


def pattern(request):
    symbol = request.POST.get('ticker','AAPL')
    pattern = request.POST.get('xpattern','CDLHAMMER')
    period = request.POST.get('period', '1y')  # Allow user to select period
    interval = request.POST.get('interval', '1d')  # Allow user to select interval
    data = get_stock_data(symbol, period, interval)








    
    open = data['Open'].values
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    pattern_func = getattr(talib, pattern.upper())
    pattern_result = pattern_func(open, high, low, close)

    """# Determine market sentiment based on the pattern
    bullish_patterns = ['CDLHAMMER', 'CDLBULLISHENGULFING', 'CDLMORNINGSTAR', 'CDLINVERTEDHAMMER', 'CDLPIERCING']
    bearish_patterns = ['CDLHANGINGMAN', 'CDLBEARISHENGULFING', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR', 'CDLDARKCLOUDCOVER']
    if pattern.upper() in bullish_patterns:
        market_sentiment = 'Bullish'
    elif pattern.upper() in bearish_patterns:
        market_sentiment = 'Bearish'
    else:
        market_sentiment = 'Neutral'


    """


#---------------CNN Model---------------------------------------------------------

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    img = get_candlestick_image(data)
    img = img.resize((64, 64))
    img = img.convert('RGB')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        market_sentiment = 'Bullish'
    else:
        market_sentiment = 'Bearish'
















    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close']))
    fig.update_layout(title=symbol, paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)

    # Add box annotation for pattern
    
    for i in range(len(pattern_result)):
        if pattern_result[i] > 0 and is_significant(data, i):  # Add a function to check if pattern is significant
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=data.index[i-15] if i-15 >= 0 else data.index[0],
                y0=0,
                x1=data.index[i],
                y1=1,
                fillcolor="rgba(255, 255, 255, 0.2)",
                line=dict(color="white", width=1),
                layer="below"
            )

            fig.add_annotation(
                x=data.index[i],
                y=data['Low'][i],
                ax=data.index[i],
                ay=data['Low'][i-10] if i-10 >= 0 else data['Low'][0],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="white"
            )

    fig_html = fig.to_html(full_html=False)

    return render(request, 'pattern.html', context={
                                                    'fig_html': fig_html,
                                                    'stock_symbol': symbol,
                                                    'pattern': pattern,
                                                    'market_sentiment' :market_sentiment
    })






#------------------------------------------technnical scan-----------------------------------------------------------------------------





"""

def technical(request):


    

    symbol = request.POST.get('ticker', 'AAPL')  # Default to Apple Inc. if no symbol provided
    df = yf.download(symbol, period="1y", interval='1d')


    #-------------------------------------------------------------------------------------------------



    df['Close'] = df['Close'].astype(float)

    df['RSI'] = RSI(df['Close'].values.astype(np.float64), timeperiod=14)
    df['CCI'] = CCI(df['High'].values.astype(np.float64), 
                    df['Low'].values.astype(np.float64),
                    df['Close'].values.astype(np.float64), 
                    timeperiod=14)
    df['ROC'] = ROC(df['Close'].astype(float), timeperiod=10)


    # Prepare data for kNN
    # Ensure that all NaN values are dropped to keep the lengths of features and labels equal
    df.dropna(inplace=True)
    features = df[['RSI', 'CCI', 'ROC']]
    labels = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1 if the price increased, 0 otherwise
    labels.dropna(inplace=True)  # Drop the last value which will be NaN after shifting

    # Ensure that features and labels have the same length
    min_length = min(len(features), len(labels))
    features = features.iloc[:min_length]
    labels = labels.iloc[:min_length]

    # Train kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)

    # Generate signals
    df['Signal'] = knn.predict(features)

        # Plot
    figg = goo.Figure()
    figg.add_trace(goo.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

    # Add buy and sell signals to the plot
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == 0]

    figg.add_trace(goo.Scatter(x=buy_signals.index, y=buy_signals['Low'], mode='markers', marker=dict(symbol='triangle-up', color='white', size=10), name='Buy Signal'))
    figg.add_trace(goo.Scatter(x=sell_signals.index, y=sell_signals['High'], mode='markers', marker=dict(symbol='triangle-down', color='blue', size=10), name='Sell Signal'))

    figg.update_layout(title=symbol, paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)
    plot = figg.to_html(full_html=False)











    # Calculate MA and SMA
    df['MA'] = talib.MA(df['Close'], timeperiod=20)
    df['SMA'] = talib.SMA(df['Close'], timeperiod=20)

    # Calculate RSI
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    # Calculate MACD
    df['MACD'], df['MACD Signal'], df['MACD Hist'] = talib.MACD(df['Close'])

    # Create a subplot with 3 rows
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('MACD', 'RSI'), vertical_spacing=0.3)

    # Add MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD Signal'], name='MACD Signal', line=dict(color='orange', width=1)), row=1, col=1)

    # Add RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=1)), row=2, col=1)

    # Update layout
    fig.update_layout(title_text="Technical Analysis", paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)

    # Convert plot to HTML
    plot_div = fig.to_html(full_html=False )



    
    
    
    


    return render(request, 'technical.html', context={
                                                        'plot_div': plot_div,
                                                        'plot': plot,
                                                        'stock_symbol':symbol
                                                    
    })
"""

"""
def technical(request):
    symbol = request.POST.get('ticker', 'AAPL')  # Default to Apple Inc. if no symbol provided
    df = yf.download(symbol, period="1y", interval='1d')

    # Calculate indicators
    df['Close'] = df['Close'].astype(float)
    df['RSI'] = RSI(df['Close'].values, timeperiod=14)
    df['CCI'] = CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    df['ROC'] = ROC(df['Close'].values, timeperiod=10)

    # Prepare data for kNN
    df.dropna(inplace=True)
    features = df[['RSI', 'CCI', 'ROC']]
    labels = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Ensure that features and labels have the same length
    min_length = min(len(features), len(labels))
    features = features.iloc[:min_length]
    labels = labels.iloc[:min_length]

    # Train kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)

    # Generate signals
    df['Signal'] = knn.predict(features)
    df['PrevSignal'] = df['Signal'].shift(1)
    df['BuySignal'] = np.where((df['Signal'] == 1) & (df['PrevSignal'] == 0), 1, 0)
    df['SellSignal'] = np.where((df['Signal'] == 0) & (df['PrevSignal'] == 1), 1, 0)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

    # Add buy and sell signals to the plot
    buy_signals = df[df['BuySignal'] == 1]
    sell_signals = df[df['SellSignal'] == 1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'))

    fig.update_layout(title=symbol, paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)
    plot = fig.to_html(full_html=False)

    return render(request, 'technical.html', context={'plot': plot, 'stock_symbol': symbol})
"""



"""

def technical(request):
    symbol = request.POST.get('ticker', 'AAPL')
    df = yf.download(symbol, period="1y", interval='1d')

    # Calculate indicators
    df['RSI'] = RSI(df['Close'], timeperiod=14)
    df['CCI'] = CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ROC'] = ROC(df['Close'], timeperiod=10)
    df['Volume'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(df['Volume'].values.reshape(-1, 1)).flatten()

    # Prepare data for kNN
    df.dropna(inplace=True)
    features = df[['RSI', 'CCI', 'ROC', 'Volume']]
    labels = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

    # Train kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features[:-1], labels[:-1])  # Exclude the last row for which we don't have a label

    # Generate signals
    df['Signal'] = knn.predict(features)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

    # Add buy and sell signals to the plot
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'))

    fig.update_layout(title=symbol, paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)
    plot = fig.to_html(full_html=False)

    return render(request, 'technical.html', context={'plot': plot, 'stock_symbol': symbol})
"""


"""

def technical(request):
    symbol = request.POST.get('ticker', 'AAPL')
    df = yf.download(symbol, period="1y", interval='1d')

    # Calculate indicators
    df['RSI'] = RSI(df['Close'], timeperiod=14)
    df['CCI'] = CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ROC'] = ROC(df['Close'], timeperiod=10)
    df['Volume'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(df['Volume'].values.reshape(-1, 1)).flatten()

    # Prepare data for kNN
    df.dropna(inplace=True)
    features = df[['RSI', 'CCI', 'ROC', 'Volume']]
    labels = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

    # Train kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features[:-1], labels[:-1])  # Exclude the last row for which we don't have a label

    # Generate signals
    df['Signal'] = knn.predict(features)
    df['PrevSignal'] = df['Signal'].shift(1)

    # Define a threshold for generating signals
    threshold = 0.5  # This is an arbitrary value; adjust based on your strategy's needs

    # Generate buy and sell signals based on the change in prediction and threshold
    df['BuySignal'] = np.where((df['Signal'] > threshold) & (df['PrevSignal'] <= threshold), 1, 0)
    df['SellSignal'] = np.where((df['Signal'] < -threshold) & (df['PrevSignal'] >= -threshold), -1, 0)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

    # Add buy and sell signals to the plot
    buy_signals = df[df['BuySignal'] == 1]
    sell_signals = df[df['SellSignal'] == -1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'], mode='markers', marker=dict(symbol='triangle-up', color='white', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'], mode='markers', marker=dict(symbol='triangle-down', color='blue', size=10), name='Sell Signal'))

    fig.update_layout(title=symbol, paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)
    plot = fig.to_html(full_html=False)

    return render(request, 'technical.html', context={'plot': plot, 'stock_symbol': symbol})
"""




"""
def technical(request):
    symbol = request.POST.get('ticker', 'AAPL')
    df = yf.download(symbol, period="1y", interval='1d')

    # Calculate indicators
    df['RSI'] = RSI(df['Close'], timeperiod=14)
    df['CCI'] = CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ROC'] = ROC(df['Close'], timeperiod=10)
    df['Volume'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(df['Volume'].values.reshape(-1, 1)).flatten()

    # Prepare data for kNN and scale all features
    df.dropna(inplace=True)
    features = df[['RSI', 'CCI', 'ROC', 'Volume']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    labels = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

    # Hyperparameter tuning for kNN
    k_values = range(1, 21)
    scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, features_scaled[:-1], labels[:-1], cv=5).mean()
        scores.append(score)
    
    best_k = k_values[scores.index(max(scores))]

    # Train kNN with best_k and bagging
    knn = KNeighborsClassifier(n_neighbors=best_k)
    bagging = BaggingClassifier(knn, max_samples=0.5, max_features=0.5)
    bagging.fit(features_scaled[:-1], labels[:-1])

    # Generate signals
    df['Signal'] = bagging.predict(features_scaled)
    df['PrevSignal'] = df['Signal'].shift(1)

    # Define a threshold for generating signals
    threshold = 0.5  # This is an arbitrary value; adjust based on your strategy's needs

    # Generate buy and sell signals based on the change in prediction and threshold
    df['BuySignal'] = np.where((df['Signal'] > threshold) & (df['PrevSignal'] <= threshold), 1, 0)
    df['SellSignal'] = np.where((df['Signal'] < -threshold) & (df['PrevSignal'] >= -threshold), -1, 0)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

    # Add buy and sell signals to the plot
    buy_signals = df[df['BuySignal'] == 1]
    sell_signals = df[df['SellSignal'] == -1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'], mode='markers', marker=dict(symbol='triangle-up', color='white', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'], mode='markers', marker=dict(symbol='triangle-down', color='blue', size=10), name='Sell Signal'))

    fig.update_layout(title=symbol, paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)
    plot = fig.to_html(full_html=False)

    return render(request, 'technical.html', context={'plot': plot, 'stock_symbol': symbol})
"""


"""

def technical(request):
    symbol = request.POST.get('ticker', 'AAPL')
    df = yf.download(symbol, period="1y", interval='1d')

    # Calculate indicators
    df['RSI'] = RSI(df['Close'], timeperiod=14)
    df['CCI'] = CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ROC'] = ROC(df['Close'], timeperiod=10)
    df['Volume'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(df['Volume'].values.reshape(-1, 1)).flatten()
    df['SMA_52'] = SMA(df['Close'], timeperiod=52)  # Add 52-period Simple Moving Average

    # Prepare data for kNN and scale all features
    df.dropna(inplace=True)
    features = df[['RSI', 'CCI', 'ROC', 'Volume']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    labels = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

    # Hyperparameter tuning for kNN
    parameters = {'n_neighbors': range(1, 21)}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters)
    clf.fit(features_scaled[:-1], labels[:-1])

    best_k = clf.best_params_['n_neighbors']

    # Train kNN with best_k and bagging
    knn = KNeighborsClassifier(n_neighbors=best_k)
    bagging = BaggingClassifier(knn, max_samples=0.5, max_features=0.5)
    bagging.fit(features_scaled[:-1], labels[:-1])

    # Generate signals
    df['Signal'] = bagging.predict(features_scaled)
    df['PrevSignal'] = df['Signal'].shift(1)

    # Define a threshold for generating signals
    threshold = 0.5  # This is an arbitrary value; adjust based on your strategy's needs  0.5

    # Generate buy and sell signals based on the change in prediction and threshold
    df['BuySignal'] = np.where((df['Signal'] > threshold) & (df['PrevSignal'] <= threshold), 1, 0)
    df['SellSignal'] = np.where((df['Signal'] < -threshold) & (df['PrevSignal'] >= -threshold), -1, 0)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

    # Add buy and sell signals to the plot
    buy_signals = df[df['BuySignal'] == 1]
    sell_signals = df[df['SellSignal'] == -1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'], mode='markers', marker=dict(symbol='triangle-up', color='white', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'], mode='markers', marker=dict(symbol='triangle-down', color='blue', size=10), name='Sell Signal'))
        # Add the SMA_52 to the plot
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_52'], mode='lines', line=dict(color='orange'), name='SMA 52'))
    fig.update_layout(title=symbol, paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)
    plot = fig.to_html(full_html=False)

    return render(request, 'technical.html', context={'plot': plot, 'stock_symbol': symbol})
"""




def technical(request):

    symbol = request.POST.get('ticker', 'AAPL')
    df = yf.download(symbol, period="1y", interval='1d')  # Adjust interval to 4 hours

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Calculate indicators
    df['RSI'] = RSI(df['Close'], timeperiod=14)  # Adjust timeperiod for RSI to match 4-hour interval
    df['CCI'] = CCI(df['High'], df['Low'], df['Close'], timeperiod=14)  # Adjust timeperiod for CCI to match 4-hour interval
    df['ROC'] = ROC(df['Close'], timeperiod=10)  # Adjust timeperiod for ROC to match 4-hour interval
    df['SMA_52'] = SMA(df['Close'], timeperiod=52)  # Adjust timeperiod for SMA to match 4-hour interval

    # Scale all features
    df['Volume'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(df['Volume'].values.reshape(-1, 1)).flatten()

    # Prepare data for kNN and scale all features
    df.dropna(inplace=True)
    features = df[['RSI', 'CCI', 'ROC', 'Volume']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    labels = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

    # Hyperparameter tuning for kNN
    parameters = {'n_neighbors': range(1, 21)}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters)
    clf.fit(features_scaled[:-1], labels[:-1])

    best_k = clf.best_params_['n_neighbors']

    # Train kNN with best_k and bagging
    knn = KNeighborsClassifier(n_neighbors=best_k)
    bagging = BaggingClassifier(knn, max_samples=0.5, max_features=0.5)
    bagging.fit(features_scaled[:-1], labels[:-1])

    # Generate signals
    df['Signal'] = bagging.predict(features_scaled)
    df['PrevSignal'] = df['Signal'].shift(1)

    # Define a threshold for generating signals
    threshold = 0.5  # This is an arbitrary value; adjust based on your strategy's needs

    # Generate buy and sell signals based on the change in prediction and threshold
    df['BuySignal'] = np.where((df['Signal'] > threshold) & (df['PrevSignal'] <= threshold), 1, 0)
    df['SellSignal'] = np.where((df['Signal'] < -threshold) & (df['PrevSignal'] >= -threshold), -1, 0)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

    # Add buy and sell signals to the plot
    buy_signals = df[df['BuySignal'] == 1]
    sell_signals = df[df['SellSignal'] == -1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'], mode='markers', marker=dict(symbol='triangle-up', color='white', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'], mode='markers', marker=dict(symbol='triangle-down', color='blue', size=10), name='Sell Signal'))

    # Add the SMA_52 to the plot
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_52'], mode='lines', line=dict(color='orange'), name='SMA 52'))

    fig.update_layout(title=symbol, paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)
    plot = fig.to_html(full_html=False)




















     # Calculate MA and SMA
    df['MA'] = talib.MA(df['Close'], timeperiod=20)
    df['SMA'] = talib.SMA(df['Close'], timeperiod=20)

    # Calculate RSI
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    # Calculate MACD
    df['MACD'], df['MACD Signal'], df['MACD Hist'] = talib.MACD(df['Close'])

    # Create a subplot with 3 rows
    fig_t = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('MACD', 'RSI'), vertical_spacing=0.3)

    # Add MACD
    fig_t.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df.index, y=df['MACD Signal'], name='MACD Signal', line=dict(color='orange', width=1)), row=1, col=1)

    # Add RSI
    fig_t.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=1)), row=2, col=1)

    # Update layout
    fig_t.update_layout(title_text="Technical Analysis", paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)

    # Convert plot to HTML
    plot_div = fig_t.to_html(full_html=False )

    return render(request, 'technical.html', context={'plot': plot, 
                                                      'stock_symbol': symbol,
                                                      'plot_div' : plot_div
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      })

"""
def algo(request):

    symbol = request.POST.get('ticker', 'AAPL')
    df = yf.download(symbol, period="1y", interval='1d')



    # Define the strategy class
    class MyStrat(Strategy):
        # Define strategy parameters

        initsize = 0.99
        mysize = initsize
        slatr = None
        TPSLRatio = 2

        def init(self):
            super().init()
            # Define indicators
            self.EMA = self.I(ta.EMA, self.data.Close, timeperiod=100)
            # Calculate VWAP manually since TA-Lib does not have a VWAP indicator
            typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
            self.VWAP = self.I(lambda h, l, c, v: (v * typical_price).cumsum() / v.cumsum(),
                                self.data.High, self.data.Low, self.data.Close, self.data.Volume)



        def next(self):
            super().next()
            # Define the trading logic
            if len(self.trades) == 0:
                # Calculate the ATR using the correct function name and uppercase
                slatr = 0.8 * ta.ATR(self.data.High, self.data.Low, self.data.Close, timeperiod=5)[-1]
                if self.data.Close[-1] > self.EMA[-1] and self.data.Close[-1] > self.VWAP[-1]:
                    sl1 = self.data.Close[-1] - slatr
                    tp1 = self.data.Close[-1] + slatr * self.TPSLRatio
                    self.buy(sl=sl1, tp=tp1, size=self.mysize)
                elif self.data.Close[-1] < self.EMA[-1] and self.data.Close[-1] < self.VWAP[-1]:
                    sl1 = self.data.Close[-1] + slatr
                    tp1 = self.data.Close[-1] - slatr * self.TPSLRatio
                    self.sell(sl=sl1, tp=tp1, size=self.mysize)

    # Create the view
    
        # Get the stock symbol from the user input

        
        # Apply the trading strategy
    bt = Backtest(df, MyStrat, cash=100000, commission=.002)
    stats = bt.run()
        
        # Generate the chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'])])
    fig.update_layout(title=f'{symbol} Candlestick Chart')
    chart = plot(fig, output_type='div')
        
        # Pass the chart and stats to the template

    





    return render(request, 'algo.html', context={ #'chart': chart,
                                                  #'stats': stats,
                                                 
                                                      
                                                      
                                                      
                                                      
                                                      
                                                      })

"""




def algo(request):
    symbol = request.POST.get('ticker', 'AAPL')
    df = yf.download(symbol, period="60d", interval='15m')

    df=df[df.High!=df.Low]

    df["VWAP"]=ta.vwap(df.High, df.Low, df.Close, df.Volume)
    df["EMA"]=ta.ema(df.Close, length=100)

    emasignal = [0]*len(df)
    backcandles = 6

    for row in range(backcandles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-backcandles, row+1):
            ema_value = df.EMA[i]
            if ema_value is not None and df.High[i] >= ema_value:
                dnt = 0
            if ema_value is not None and df.Low[i] <= ema_value:
                upt=0
        if upt==1 and dnt==1:
            emasignal[row]=3
        elif upt==1:
            emasignal[row]=2
        elif dnt==1:
            emasignal[row]=1

    df['EMASignal'] = emasignal

    VWAPsignal = [0]*len(df)
    backcandles = 3

    for row in range(backcandles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-backcandles, row+1):
            if df.High[i]>=df.VWAP[i]:
                dnt=0
            if df.Low[i]<=df.VWAP[i]:
                upt=0
        if upt==1 and dnt==1:
            VWAPsignal[row]=3
        elif upt==1:
            VWAPsignal[row]=2
        elif dnt==1:
            VWAPsignal[row]=1

    df['VWAPSignal'] = VWAPsignal

    def TotalSignal(l):
        myclosedistance = 100
        if (df.EMASignal[l]==2 and df.VWAPSignal[l]==2
            and min(abs(df.VWAP[l]-df.High[l]),abs(df.VWAP[l]-df.Low[l]))<=myclosedistance):
                return 2
        if (df.EMASignal[l]==1 and df.VWAPSignal[l]==1
            and min(abs(df.VWAP[l]-df.High[l]),abs(df.VWAP[l]-df.Low[l]))<=myclosedistance):
                return 1

    TotSignal = [0]*len(df)
    for row in range(0, len(df)):
        TotSignal[row] = TotalSignal(row)
    df['TotalSignal'] = TotSignal

    def pointposbreak(x):
        if x['TotalSignal']==1:
            return x['High']+1e-3
        elif x['TotalSignal']==2:
            return x['Low']-1e-3
        else:
            return np.nan

    df['pointposbreak'] = df.apply(lambda row: pointposbreak(row), axis=1)

    dfpl = df[1500:1900]
    dfpl.reset_index(inplace=True)
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close']),
                    go.Scatter(x=dfpl.index, y=dfpl.EMA, line=dict(color='orange', width=1), name="EMA"),
                    go.Scatter(x=dfpl.index, y=dfpl.VWAP, line=dict(color='blue', width=1), name="VWAP")])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointposbreak'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="Signal")
    fig.update_layout(title_text="T", paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white", xaxis_rangeslider_visible=False)

    # Convert plot to HTML
    plot_div = fig.to_html(full_html=False )   



    dfpl = df[:].copy()
    dfpl.reset_index(inplace=True)

    dfpl['ATR']=ta.atr(dfpl.High, dfpl.Low, dfpl.Close, length=5)

    def SIGNAL():
        return dfpl.TotalSignal

    class MyStrat(Strategy):
        initsize = 0.99
        mysize = initsize
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            slatr = 0.8*self.data.ATR[-1]
            TPSLRatio = 2

            if self.signal1==2 and len(self.trades)==0:
                sl1 = self.data.Close[-1] - slatr
                tp1 = self.data.Close[-1] + slatr*TPSLRatio
                self.buy(sl=sl1, tp=tp1, size=self.mysize)

            elif self.signal1==1 and len(self.trades)==0:
                sl1 = self.data.Close[-1] + slatr
                tp1 = self.data.Close[-1] - slatr*TPSLRatio
                self.sell(sl=sl1, tp=tp1, size=self.mysize)

    bt = Backtest(dfpl, MyStrat, cash=100000, margin=1/5, commission=.00)
    stat = bt.run()
    
    #driver = webdriver.Chrome(executable_path='"C:\My Folder\stock_web_app\stock-master-pro\app\chromedriver.exe"')


# Specify the path to chromedriver using the Service object
    service = ChromeService(executable_path='chromedriver.exe')
    driver = webdriver.Chrome(service=service)

    #bt.plot()
    figp = bt.plot()

    # Save the figure to the static directory
    #figp.savefig('app/static/trade_plot.png')
    export_png(figp, filename="app/static/trade_plot.png")


    # Close the figure to free up memory
    #plt.close(figp)













    return render(request, 'algo.html', context={ 'plot_div': plot_div,
                                                  'stats': stat,  
                                                  'symbol': symbol,

                                                
                                                      })
