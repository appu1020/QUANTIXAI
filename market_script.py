from bs4 import BeautifulSoup
import pandas as pd
import requests
import yfinance as yf

url = "https://finance.yahoo.com/markets/stocks/most-active/?start=0&count=100"

# Get the webpage content
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

table = soup.find("table")

# Extract data into a list
data = []
for row in table.find_all("tr"):
    cols = [col.text.strip() for col in row.find_all("td")]
    if cols:
        data.append(cols)

# Convert to Pandas DataFrame
df = pd.DataFrame(data)

df.head()

df.columns = ["Symbol",
    "Stock Name",
    "Price",
    "Change",
    "Change %",
    "Volume",
    "Avg Vol (3M)",
    "Market Cap",
    "P/E Ratio",
    "52 wk Change %",
    "a","b"
]


df.columns

df.head(200)

df.isna().sum()

df = df[['Symbol', 'Stock Name', 'Change','Avg Vol (3M)','Market Cap']]

df.rename(columns={
    'Change': 'Price Change'
}, inplace=True)

df

