# oss: https://perfiliev.co.uk/market-commentary/how-to-calculate-gamma-exposure-and-zero-gamma-level/
# data source: https://www.cboe.com/delayed_quotes/spx/quote_table
import pandas as pd
import numpy as np
# import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
# from matplotlib import dates
# from itertools import accumulate

pd.options.display.float_format = '{:,.4f}'.format

#====== START functions =======
# Black-Scholes European-Options Gamma
def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    if T == 0 or vol == 0:
        return 0

    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T) 

    if optType == 'call':
        gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma 
    else: # Gamma is same for calls and puts. This is just to cross-check
        gamma = K * np.exp(-r*T) * norm.pdf(dm) / (S * S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma 

def isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21

def create_xticks(n1, n2):
  n1 = int(n1 / 10)
  n2 = int(n2 / 10)

  arr = []
  for x in range (n1 * 10, n2 * 10, 10):
    arr.append(x)

  return arr
#====== END of functions =======

today = date.today()
# Expiration Date Input
# expiry_date = '2022-12-08'
expiry_date = input('Enter SPX Expiry Date (' + today.strftime('%Y-%m-%d') + '): ')
if not expiry_date:
    expiry_date = today.strftime('%Y-%m-%d')
print('You entered: ' + expiry_date)

# Inputs and Parameters
filename = 'spx_quotedata.csv'
# https://www.cboe.com/delayed_quotes

# This assumes the CBOE file format hasn't been edited, i.e. table beginds at line 4
optionsFile = open(filename)
optionsFileData = optionsFile.readlines()
optionsFile.close()

# Get SPX Spot
spotLine = optionsFileData[1]
spotPrice = float(spotLine.split('Last:')[1].split(',')[0])
fromStrike = 0.95 * spotPrice # 0.8
toStrike = 1.05 * spotPrice # 1.2

# Get Today's Date
dateLine = optionsFileData[2]
todayDate = dateLine.split('Date: ')[1].split(',')
monthDay = todayDate[0].split(' ')

# Get File Datetime
fileDatetime = dateLine.split('Date: ')[1].split('",')[0]
fileDatetime = fileDatetime[0:-4]
# print('Date: ' + fileDatetime)
generated_datetime = datetime.strptime(fileDatetime, '%B %d, %Y at %I:%M %p')

# Handling of US/EU date formats
if len(monthDay) == 2:
    year = int(todayDate[1].split(' ')[1])
    month = monthDay[0]
    day = int(monthDay[1])
else:
    year = int(monthDay[2].split(' ')[1])
    month = monthDay[1]
    day = int(monthDay[0])

todayDate = datetime.strptime(month,'%B')
todayDate = todayDate.replace(day=day, year=year)

# Get SPX Options Data
df = pd.read_csv(filename, sep=",", header=None, skiprows=4)
df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
              'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
              'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']

df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y')
df['ExpirationDate'] = df['ExpirationDate'] + timedelta(hours=16)
df['StrikePrice'] = df['StrikePrice'].astype(float)
df['CallIV'] = df['CallIV'].astype(float)
df['PutIV'] = df['PutIV'].astype(float)
df['CallGamma'] = df['CallGamma'].astype(float)
df['PutGamma'] = df['PutGamma'].astype(float)
df['CallOpenInt'] = df['CallOpenInt'].astype(float)
df['PutOpenInt'] = df['PutOpenInt'].astype(float)
# Add Vol data
df['CallVol'] = df['CallVol'].astype(float)
df['PutVol'] = df['PutVol'].astype(float)

# Filter df for a specific expiry date, and create a copy of original df
fdf = df.loc[df.ExpirationDate == expiry_date  + ' 16:00:00'].copy()
# print(fdf)

# ---=== CALCULATE SPOT GAMMA ===---
# Gamma Exposure = Unit Gamma * Open Interest * Contract Size * Spot Price 
# To further convert into 'per 1% move' quantity, multiply by 1% of spotPrice
df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10**9
dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)
strikes = dfAgg.index.values

fdf['CallGEX'] = fdf['CallGamma'] * fdf['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
fdf['PutGEX'] = fdf['PutGamma'] * fdf['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

fdf['CallGEXVol'] = fdf['CallGamma'] * fdf['CallVol'] * 100 * spotPrice * spotPrice * 0.01
fdf['PutGEXVol'] = fdf['PutGamma'] * fdf['PutVol'] * 100 * spotPrice * spotPrice * 0.01 * -1

fdf['TotalGamma'] = (fdf.CallGEX + fdf.PutGEX) / 10**9
fdfAgg = fdf.groupby(['StrikePrice']).sum(numeric_only=True)
strikes = fdfAgg.index.values

# Print selected columns
# print(fdf.filter(items=['StrikePrice', 'TotalGamma']))

#-==== Calculate Zero Gamma ====-
# ---=== CALCULATE GAMMA PROFILE ===---
levels = np.linspace(fromStrike, toStrike) # 60

# For 0DTE options, I'm setting DTE = 1 day, otherwise they get excluded
fdf['daysTillExp'] = [1/262 if (np.busday_count(todayDate.date(), x.date())) == 0 \
                           else np.busday_count(todayDate.date(), x.date())/262 for x in fdf.ExpirationDate]
nextExpiry = fdf['ExpirationDate'].min()

fdf['IsThirdFriday'] = [isThirdFriday(x) for x in fdf.ExpirationDate]
thirdFridays = fdf.loc[fdf['IsThirdFriday'] == True]
nextMonthlyExp = thirdFridays['ExpirationDate'].min()

totalGamma = []
callGammaEx = []
putGammaEx = []
totalGammaExNext = []
totalGammaExFri = []

# For each spot level, calc gamma exposure at that point
for level in levels:
    fdf['callGammaEx'] = fdf.apply(lambda row : calcGammaEx(level, row['StrikePrice'], row['CallIV'], 
                                                          row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis = 1)

    fdf['putGammaEx'] = fdf.apply(lambda row : calcGammaEx(level, row['StrikePrice'], row['PutIV'], 
                                                         row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis = 1)    
    callGammaEx.append(fdf['callGammaEx'].sum())
    putGammaEx.append(fdf['putGammaEx'].sum())
    totalGamma.append(fdf['callGammaEx'].sum() - fdf['putGammaEx'].sum())

    exNxt = fdf.loc[fdf['ExpirationDate'] != nextExpiry]
    totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())

    exFri = fdf.loc[fdf['ExpirationDate'] != nextMonthlyExp]
    totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())

totalGamma = np.array(totalGamma) / 10**9
totalGammaExNext = np.array(totalGammaExNext) / 10**9
totalGammaExFri = np.array(totalGammaExFri) / 10**9

# Calculate Major Call GEX and Put GEX VOL
callGEXInd = np.argmax(fdfAgg['CallGEXVol'].to_numpy() / 10**9)
callGEXMajor = strikes[callGEXInd]
putGEXInd = np.argmin(fdfAgg['PutGEXVol'].to_numpy() / 10**9)
putGEXMajor = strikes[putGEXInd]

# Find Gamma Flip Point
zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]
negGamma = totalGamma[zeroCrossIdx]
posGamma = totalGamma[zeroCrossIdx+1]
negStrike = levels[zeroCrossIdx]
posStrike = levels[zeroCrossIdx+1]
# Might use Vol data to calculate...
# print('totalGamma', totalGamma)
# print('totalGamma', np.diff(np.sign(totalGamma)))
# print('zeroCrossIdx', zeroCrossIdx)
# print('negGamma', negGamma)
# print('posGamma', posGamma)
# print('negStrike', negStrike)
# print('posStrike', posStrike)

# print(levels)
# Writing and sharing this code is only possible with your support! 
# If you find it useful, consider supporting us at perfiliev.com/support :)
zeroGamma = posStrike - ((posStrike - negStrike) * posGamma/(posGamma-negGamma)) # -15.09 delta adjustment
if len(zeroGamma) == 0:
    zeroGamma = spotPrice
else:
    zeroGamma = zeroGamma[0]
print('zeroGamma', zeroGamma)

# Chart 1: Absolute Gamma Exposure by OI
plt.grid()
plt.bar(strikes, fdfAgg['TotalGamma'].to_numpy(), width=6, linewidth=0.1, edgecolor='k', label="Gamma Exposure")
plt.xlim([fromStrike, toStrike])
chartTitle = 'SPX Total GEX by OI for ' + expiry_date + ' Expiration' # todayDate.strftime('%b-%d-%Y')
chartTitle = "Total Gamma: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " Bn per 1% SPX Move"
plt.title(chartTitle, fontweight="bold", fontsize=20)
plt.xlabel('Strike  (data from CBOE on ' +generated_datetime.strftime('%m-%d-%Y %I:%M %p')+ ')', fontweight="bold")
plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
# plt.axvline(x=zeroGamma, color='g', lw=1, label="Zero Gamma: " + str("{:,.0f}".format(zeroGamma)))
plt.xticks(create_xticks(fromStrike, toStrike), rotation=45, fontsize=8)
plt.legend()
plt.show()

# Chart 2: Absolute Gamma Exposure by Calls and Puts OI
# plt.grid()
# plt.bar(strikes, fdfAgg['CallGEXVol'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma VOL", color='#cefcb8')
# plt.bar(strikes, fdfAgg['PutGEXVol'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma VOL", color='#e69b6a')
# plt.bar(strikes, fdfAgg['CallGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma OI", color='#4dba1a')
# plt.bar(strikes, fdfAgg['PutGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma OI", color='#f23030')
# plt.xlim([fromStrike, toStrike])
# # chartTitle = "Total Gamma: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " Bn per 1% SPX Move"
# chartTitle = 'SPX GEX for ' + expiry_date + ' Expiration' # todayDate.strftime('%b-%d-%Y')
# plt.title(chartTitle, fontweight="bold", fontsize=20)
# plt.xlabel('Strike  (data from CBOE on ' +generated_datetime.strftime('%m-%d-%Y %I:%M %p')+ ')', fontweight="bold")
# plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
# plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot:" + str("{:,.2f}".format(spotPrice)))
# # plt.axvline(x=zeroGamma, color='b', lw=1, label="Zero Gamma: " + str("{:,.2f}".format(zeroGamma)))
# plt.axvline(x=callGEXMajor, color='#005b96', lw=1, label="Call Major Gamma: " + str("{:,.0f}".format(callGEXMajor)))
# plt.axvline(x=putGEXMajor, color='#005b96', lw=1, label="Put Major Gamma: " + str("{:,.0f}".format(putGEXMajor)))
# plt.xticks(create_xticks(fromStrike, toStrike), rotation=45, fontsize=8)
# plt.legend()
# plt.show()

# Chart 3: Absolute Gamma Exposure by Calls and Puts VOL
# plt.grid()
# plt.bar(strikes, fdfAgg['CallGEXVol'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma")
# plt.bar(strikes, fdfAgg['PutGEXVol'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma")
# plt.xlim([fromStrike, toStrike])
# # chartTitle = "Total Gamma: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " Bn per 1% SPX Move"
# chartTitle = 'SPX GEX by Calls and Puts VOL for ' + expiry_date + ' Expiration' # todayDate.strftime('%b-%d-%Y')
# plt.title(chartTitle, fontweight="bold", fontsize=20)
# plt.xlabel('Strike  (data from CBOE on ' +generated_datetime.strftime('%m-%d-%Y %I:%M %p')+ ')', fontweight="bold")
# plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
# plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot:" + str("{:,.0f}".format(spotPrice)))
# plt.axvline(x=zeroGamma, color='g', lw=1, label="Zero Gamma: " + str("{:,.0f}".format(zeroGamma)))
# plt.xticks(create_xticks(fromStrike, toStrike), rotation=45, fontsize=8)
# plt.legend()
# plt.show()
