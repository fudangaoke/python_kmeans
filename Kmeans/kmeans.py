import pandas as pd
import numpy as np
import operator
output_address = "C:\\Users\\HP\\Desktop\\HW4\\output.txt"

def OutputTXT(file_address,string):
	with open(file_address,'w') as f:
		f.write(string)

def Outputdf(df):
	df.to_csv('C:\\Users\\HP\\Desktop\\HW4\\output.csv')

def OpenCSV(tickers,address):
	dict={}
	for ticker in tickers:
		dict[ticker]=pd.read_csv(address+"\\"+str(ticker)+".csv",index_col=0)
	return dict

consumer_tickers=['AMT', 'AMZN', 'BAM', 'CCI', 'CMCSA', 'COST', 'DIS', 'HD', 'LOW', 'LVS', 'MAR', 'MCD', 'NFLX', 'PLD', 'RELX', 'SBUX', 'SPG', 'TGT', 'TJX', 'WMT']
finance_tickers=['AXP', 'BAC', 'BK', 'BLK', 'C', 'CME', 'COF', 'GS', 'HDB', 'ITUB', 'JPM', 'LFC', 'MS', 'PNC', 'RY', 'SAN', 'SPGI', 'TD', 'USB', 'WFC']
healthcare_tickers=['ABBV', 'ABT', 'AMGN', 'ANTM', 'AZN', 'BDX', 'BMY', 'CVS', 'GILD', 'GSK', 'JNJ', 'LLY', 'MDT', 'MMM', 'MRK', 'NVO', 'NVS', 'PFE', 'SYK', 'UNH']
technology_tickers=['AAPL', 'ADBE', 'ADP', 'AVGO', 'CRM', 'CSCO', 'FB', 'GOOG', 'IBM', 'INTC', 'INTU', 'ITW', 'MSFT', 'NVDA', 'ORCL', 'QCOM', 'SAP', 'TSM', 'TXN', 'VMW']
consumer_dict=OpenCSV(consumer_tickers,'C:\\Users\\HP\\Desktop\\HW4\\Consumer')
finance_dict=OpenCSV(finance_tickers,'C:\\Users\\HP\\Desktop\\HW4\\Finance')
healthcare_dict=OpenCSV(healthcare_tickers,'C:\\Users\\HP\\Desktop\\HW4\\Healthcare')
technology_dict=OpenCSV(technology_tickers,'C:\\Users\\HP\\Desktop\\HW4\\Technology')
columns=['Gross Margin', 'Operating Margin', 'Payout Ratio', 'Tax Rate',
       'Net Margin', 'Asset Turnover', 'Return on Assets',
       'Financial Leverage', 'Return on Equity', 'Return on Invested Capital',
       'Interest Coverage', 'Current Ratio', 'Quick Ratio',
       'Financial Leverage', 'Debt to Equity Ratio', 'Fixed Assets Turnover',
       'Asset Turnover']
tickers=consumer_tickers+finance_tickers+healthcare_tickers+technology_tickers

dict={**consumer_dict,**finance_dict,**healthcare_dict,**technology_dict}

def getData(ticker, date, type):
	return float(dict[ticker].loc[date,type])

def getDict(date, type):
	new_dict={}
	for ticker in tickers:
		new_dict[ticker]=getData(ticker,date,type)
	return new_dict

def getDataframe(date):
	list1=[]
	for ticker in tickers:
		list2=[]
		for type in columns:
			list2.append(getData(ticker,date,type))
		list1.append(list2)
	df=pd.DataFrame(list1,index=tickers,columns=columns)
	return df


#normalize data
def ScaleDataframe(df):
	scaled_df=(df-df.min())/(df.max()-df.min())
	return scaled_df

def RandomArray(dimension):
	return np.random.random(dimension)

class Centroid:      #a single centroid point
	def __init__(self,position):
		self.position=position
		self.ticker_list=[]
		self.point_list=[]      #Assign different points to point_list for each centroid
		self.previous_ticker_list=[]     #Determine when the algorithm is finished.


class Kmeans:
	def __init__(self, k,df, maximum_iters):			#k: k value, 
		self.k = k
		self.centroid_list = []   #list of centroids
		self.df = df
		self.data=df.values
		self.dimension = df.values.shape[1]   #df.values.shape[1] is dimension of centroid coordinate.
		self.maximum_iters=maximum_iters
		self.count = 0

		#initialize random centroid_list 
		for _ in range(k):
			position_array=np.array(RandomArray(self.dimension))
			self.centroid_list.append(Centroid(position_array))
		
	def assign_centroid(self,x):
		"""
		return centroid closest to a certain point
		"""
		distances = {}
		for centroid in self.centroid_list:
			distances[centroid] = np.linalg.norm(centroid.position - x)  #This is a dictionary, key is centroid, value is distance from data to centroid
		closest_centroid = min(distances.items(), key = operator.itemgetter(1))[0]  #return centroid which has min of distance
		return closest_centroid

	def fit(self):
		"""
		Fit point_list in self.data
		Assign the point_list the centroid
		Update centroid location based on assigned point_list
		"""
		terminate = 0
		while self.count < self.maximum_iters and terminate < self.k:
			terminate = 0
			#Erase last allocation result.
			for centroid in self.centroid_list:
				centroid.previous_ticker_list = centroid.ticker_list
				centroid.point_list=[]
				centroid.ticker_list=[]

			#add points to different centroids.
			for i, point in enumerate(self.data):
				closest_centroid = self.assign_centroid(point)
				closest_centroid.point_list.append(point)
				closest_centroid.ticker_list.append(self.df.index[i])

			#determine terminate condition
			for i in range(self.k):
				if (self.centroid_list[i].previous_ticker_list == self.centroid_list[i].ticker_list and len(self.centroid_list[i].ticker_list)>0):
					terminate += 1
				

			for i in range(self.k):
				array = np.array([0.0]*self.dimension)
				if len(self.centroid_list[i].point_list)>0:
					for j in range(len(self.centroid_list[i].point_list)):
						array += self.centroid_list[i].point_list[j]
					array = array/(len(self.centroid_list[i].point_list))
					self.centroid_list[i].position = array
				elif len(self.centroid_list[i].point_list)==0:
					for j in range(self.df.values.shape[0]):
						array += self.data[j]
					array = array/self.df.values.shape[0]
					self.centroid_list[i].position = array
			self.count += 1
	def __str__(self):
		string=""
		string+="Iteration count ="+str(self.count)+"\n"
		for i in range(self.k):
			string+="Group "+str(i)+" with "+str(len(self.centroid_list[i].ticker_list))+" element"+"\n"
			string+="Ticker list : "+str(self.centroid_list[i].ticker_list)+"\n\n"
		return string
	def __repr__(self):
		string=""
		string+="Iteration count ="+str(self.count)+"\n"
		for i in range(self.k):
			string+="Group "+str(i)+" with "+str(len(self.centroid_list[i].ticker_list))+" element"+"\n"
			string+="Ticker list : "+str(self.centroid_list[i].ticker_list)+"\n\n"
		return string

	def inSameGroup(self,ticker1,ticker2):
		for i in range(self.k):
			if ticker1 in self.centroid_list[i].ticker_list:
				if ticker2 in self.centroid_list[i].ticker_list:
					return True
				else:
					return False


def start(year,k_value,max_iter):
	df=getDataframe(year)
	df.drop('Financial Leverage',axis =1, inplace=True)
	df=ScaleDataframe(df)
	a=Kmeans(k_value,df,max_iter)
	a.fit()
	string=''
	print("Year = ",year, ", K = ",k_value)
	print(a)
	string+="Year = "+str(year)+", K = "+str(k_value)+"\n"
	string+=str(a)
	return string, a



#main program
string =''
kmeans_list=[]
for year in range(2013,2016):
	for k_value in range(2,9):
		s, kmeans_result= start(year,k_value,100)
		string+=s
		kmeans_list.append(kmeans_result)
	string+="********************************\n"


OutputTXT(output_address,string)

#Similarity
for i, ticker1 in enumerate(tickers):
	for j, ticker2 in enumerate(tickers):
		if i<j:
			count = 0
			threshold = 0.8*len(kmeans_list)
			for a in kmeans_list:
				if a.inSameGroup(ticker1, ticker2):
					count+=1
			if count > threshold:
				print("[",ticker1,",",ticker2,"]  Similarity = ",count/len(kmeans_list))
