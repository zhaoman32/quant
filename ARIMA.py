#导入包
import pandas as pd
import pandas_datareader
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import datetime
from pylab import mpl
from statsmodels.tsa.arima_model import ARIMA

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

#爬取08-18十年数据
start = datetime.datetime(2008,1,1)
end = datetime.datetime(2018,1,1)
#必须要调用这个函数
yf.pdr_override()
#股票数据
stock_data = pandas_datareader.data.get_data_yahoo("SPY",start,end)
print(stock_data)

#数据可视化
stock_data['Close'].plot()
plt.title('股票每日收盘价')
plt.show()


stock_week = stock_data['Close'].resample('W-MON').mean()
#划分训练集
train_data = stock_week['2008':'2016']
#可视化
train_data.plot()
plt.title('股票每周收盘价均值')
plt.show()


#这里进行一阶差分
diff_data = train_data.diff()
diff_data.dropna(inplace=True)
diff_data.plot()
plt.show()


#查看acf与pacf确定q和p
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
acf = plot_acf(diff_data,lags=25)
plt.title('ACF')
acf.show()
pacf = plot_pacf(diff_data,lags=25)
plt.title('PACF')
pacf.show()
plt.show()


model = ARIMA(train_data,order=(1,1,1),freq='W-MON')
arima = model.fit()
pre_data = arima.predict('2017-01-02','2017-12-30',dynamic=True,typ='levels')

#可视化对比
stock_con = pd.concat([stock_week,pre_data],axis=1,keys=['ori','pre'])
stock_con.plot()
plt.title('预测情况')
plt.show()




