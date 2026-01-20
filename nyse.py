import yfinance as yf
import pandas as pd
import requests
from get_all_tickers import get_tickers
import time
from tqdm import tqdm

def get_nyse_tickers():
    """Get NYSE tickers directly from NASDAQ API - NO external dependencies"""
    try:
        url = "https://api.nasdaq.com/api/screener/stocks?exchange=nyse&limit=0&offset=0"
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        response = requests.get(url, headers=headers)
        data = response.json()
        nyse_tickers = []
        for row in data['data']['table']['rows']:
            symbol = row['symbol']
            if symbol != 'N/A' and '.' not in symbol:  # Filter valid NYSE tickers
                nyse_tickers.append(symbol)
        print(len(nyse_tickers))
        quit()
        
        return nyse_tickers[:5]  # Limit to avoid rate limits
    except Exception as e:
        print(f"API failed, using fallback list: {e}")
        # Fallback NYSE tickers
        return [
            'JPM', 'UNH', 'V', 'MA', 'JNJ', 'HD', 'PG', 'CVX', 'ABBV', 'BAC',
            'KO', 'XOM', 'CRM', 'WMT', 'NFLX', 'TXN', 'AVGO', 'ACN', 'COST', 'ABT'
        ]

def fetch_financial_data(ticker):
    """Fetch comprehensive financial data for a single ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Basic info with financial metrics
        info = stock.info
        
        top_keys = [
        "marketCap",           # 1.69T - Overall company size [1st priority]
        "trailingPE",          # 33.84 - Price-to-earnings valuation
        "forwardPE",           # 24.53 - Forward-looking valuation  
        "totalRevenue",        # 3.63T - Revenue scale
        "netIncomeToCommon",   # 1.57T - Bottom line profitability
        "profitMargins",       # 43.29% - Profit efficiency
        "revenueGrowth",       # 30.3% - Revenue growth rate
        "earningsGrowth",      # 39.1% - Earnings growth rate
        "currentPrice",        # 325.25 - Current valuation point
        "priceToBook",         # 53.32 - Book value comparison
        "industryKey",         # Industry classification
        "sectorKey",              # Sector classification
        "longName",          # User-friendly name
        "country"              # Country of origin
        ]
        data = {key: info[key] for key in top_keys if key in info}
        data['ticker'] = ticker
        return data

    except Exception as e:
        return {'ticker': ticker, 'error': str(e)}

def main():
    print("Fetching NYSE tickers...")
    tickers = get_nyse_tickers()
    print(f"Found {len(tickers)} NYSE tickers (limited to first 500 for rate limiting)")
    
    results = []
    for ticker in tickers:
        data = fetch_financial_data(ticker)
        results.append(data)
        time.sleep(.1)  # Rate limiting
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv('nyse_financial_data.csv', index=False)
    

if __name__ == "__main__":
    # Install required packages first:
    # pip install yfinance pandas get-all-tickers tqdm openpyxl
    main()
