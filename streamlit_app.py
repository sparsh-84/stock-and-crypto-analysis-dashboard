import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sb
import yfinance as yf
from PIL import Image
import cufflinks as cf
import datetime
from streamlit_option_menu import option_menu
with st.sidebar:
    selected = option_menu(
        menu_title="main menu",
        options=["stock market", "cryptocurrencies"],
        icons=["bar-chart-fill", "currency-bitcoin"],
        menu_icon="cast"
    )

if selected == "stock market":
    st.title('S&P 500 Dashboard')
    st.sidebar.header('Select one or more sector')
    image = Image.open('logo1.jpg')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.image(image, width=600)

    st.markdown("""
        Retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)
        """)

    excel_file = "book1.xlsx"
    sheet_name = "data1"
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
    st.line_chart(df[:25], x="Date", y="High")

    # Web scraping of S&P 500 data
    #

    @st.cache
    def load_data():
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        html = pd.read_html(url, header=0)
        df = html[0]
        return df

    df = load_data()
    sector = df.groupby('GICS Sector')

    # Sidebar - Sector selection
    sorted_sector_unique = sorted(df['GICS Sector'].unique())
    selected_sector = st.sidebar.multiselect(
        'Sector', sorted_sector_unique, sorted_sector_unique)

    # Filtering data
    df_selected_sector = df[(df['GICS Sector'].isin(selected_sector))]

    st.header('Display Companies in Selected Sector')
    st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(
        df_selected_sector.shape[1]) + ' columns.')
    st.dataframe(df_selected_sector)

    # Download S&P500 data
    # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806

    def filedownload(df):
        csv = df.to_csv(index=False)
        # strings <-> bytes conversions
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
        return href

    st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

    # https://pypi.org/project/yfinance/

    data = yf.download(
        tickers=list(df_selected_sector.Symbol),
        period="ytd",
        interval="1d",
        group_by='ticker',
        auto_adjust=True,
        prepost=True,
        threads=True,
        proxy=None
    )
    data.dropna(axis=0)
    st.markdown(
        """ ## The opening price,high price,low price and total volume of shares""")
    st.dataframe(data)

    # downloading opening closing prices of company

    def filedownload(df):
        csv = df.to_csv(index=False)
        # strings <-> bytes conversions
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
        return href

    st.markdown(filedownload(data), unsafe_allow_html=True)
    # Plot Closing Price of Query Symbol
    col1, col2, col3, col4 = st.columns(4)

    def price_plot1(symbol):
        df = pd.DataFrame(data[symbol].Close)
        df['Date'] = df.index
        fig = plt.figure(figsize=(4, 3))
        plt.plot(df.Date, df.Close, c='blue', )
        plt.xticks(rotation=45)
        plt.title(symbol, fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Closing Price', fontweight='bold')
        plt.grid()
        return st.pyplot(fig)
    # using pairplot func to see histogram and scatterplot

    def price_plot2(symbol):
        df = pd.DataFrame(data[symbol])
        df['Date'] = df.index
        sb.pairplot(
            df,
            y_vars=["Volume"],
            x_vars=["Open", "High"],
        )
        plt.xticks(rotation=45)
        plt.title(symbol, fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Closing Price', fontweight='bold')
        plt.grid()
        return st.pyplot()

    def see_outliers(symbol):
        df = pd.DataFrame(data[symbol])
        df['Date'] = df.index
        sb.boxplot(data=df[['High', 'Low']], orient="h")
        return st.pyplot()

    num_company = st.sidebar.slider('Number of Companies', 1, 10)
    if st.button('Show Plots'):
        for i in list(df_selected_sector.Symbol)[:num_company]:
            st.title(i)
            price_plot1(i)
            price_plot2(i)
            see_outliers(i)
# -----------------------------------------------------------------------------------------------
    st.sidebar.subheader('Query parameters')
    start_date = st.sidebar.date_input("Start date", datetime.date(2022, 1, 1))
    end_date = st.sidebar.date_input("End date", datetime.date(2022, 1, 31))

    # Retrieving tickers data
    ticker_list = pd.read_excel(
        'ticker.xlsx')
    tickerSymbol = st.sidebar.selectbox(
        'Stock ticker', ticker_list)  # Select ticker symbol
    tickerData = yf.Ticker(tickerSymbol)  # Get ticker data
    # get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

    # Ticker information
    string_logo = '<img src=%s>' % tickerData.info['logo_url']
    st.markdown(string_logo, unsafe_allow_html=True)

    string_name = tickerData.info['longName']
    st.header('**%s**' % string_name)

    string_summary = tickerData.info['longBusinessSummary']
    st.info(string_summary)
    st.write(tickerDf)

    # Bollinger bands
    st.header('**Bollinger Bands**')
    qf = cf.QuantFig(tickerDf, title='First Quant Figure',
                     legend='top', name='GS')
    qf.add_bollinger_bands()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)
    ####
    # st.write('---')
    # st.write(tickerData.info)

    def price_plot3():
        plt.plot(tickerDf.Close, c='blue')
        plt.xticks(rotation=45)
        plt.title(tickerSymbol, fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Closing Price', fontweight='bold')
        plt.grid()
        return st.pyplot()
    # using pairplot func to see histogram and scatterplot

    def price_plot4():
        sb.pairplot(
            tickerDf,
            y_vars=["Volume"],
            x_vars=["Open", "High"],
        )
        plt.xticks(rotation=45)
        plt.title(tickerSymbol, fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Closing Price', fontweight='bold')
        plt.grid()
        return st.pyplot()

    def see_outliers1(tickerdf):
        sb.boxplot(data=tickerdf[['High', 'Low']], orient="h")
        return st.pyplot()

    chart_options = ["Line chart", "Pairplot", "Boxplot"]
    charts = st.selectbox(
        "**what chart would you like to see?**", options=chart_options)
    if charts == "line chart":
        price_plot3()
    if charts == "Pairplot":
        price_plot4()
    if charts == "Boxplot":
        see_outliers1(tickerDf)


if selected == "cryptocurrencies":
    st.markdown('''# **Crypto Hub**
    A cryptocurrency price app pulling price data from Binance API.
    ''')
    image = Image.open('logo.jpg')

    st.image(image, width=600)

    st.header('**Selected Price**')

    # Load market data from Binance API
    df = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')

    # Custom function for rounding values

    def round_value(input_value):
        if input_value.values > 1:
            a = float(round(input_value, 2))
        else:
            a = float(round(input_value, 8))
        return a

    col1, col2, col3 = st.columns(3)

    # Cryptocurrency selection box
    col1_selection = st.sidebar.selectbox(
        'Price 1', df.symbol, list(df.symbol).index('BTCBUSD'))
    col2_selection = st.sidebar.selectbox(
        'Price 2', df.symbol, list(df.symbol).index('ETHBUSD'))
    col3_selection = st.sidebar.selectbox(
        'Price 3', df.symbol, list(df.symbol).index('BNBBUSD'))
    col4_selection = st.sidebar.selectbox(
        'Price 4', df.symbol, list(df.symbol).index('XRPBUSD'))
    col5_selection = st.sidebar.selectbox(
        'Price 5', df.symbol, list(df.symbol).index('ADABUSD'))
    col6_selection = st.sidebar.selectbox(
        'Price 6', df.symbol, list(df.symbol).index('DOGEBUSD'))
    col7_selection = st.sidebar.selectbox(
        'Price 7', df.symbol, list(df.symbol).index('SHIBBUSD'))
    col8_selection = st.sidebar.selectbox(
        'Price 8', df.symbol, list(df.symbol).index('DOTBUSD'))
    col9_selection = st.sidebar.selectbox(
        'Price 9', df.symbol, list(df.symbol).index('MATICBUSD'))

    # DataFrame of selected Cryptocurrency
    col1_df = df[df.symbol == col1_selection]
    col2_df = df[df.symbol == col2_selection]
    col3_df = df[df.symbol == col3_selection]
    col4_df = df[df.symbol == col4_selection]
    col5_df = df[df.symbol == col5_selection]
    col6_df = df[df.symbol == col6_selection]
    col7_df = df[df.symbol == col7_selection]
    col8_df = df[df.symbol == col8_selection]
    col9_df = df[df.symbol == col9_selection]

    # function to round values
    col1_price = round_value(col1_df.weightedAvgPrice)
    col2_price = round_value(col2_df.weightedAvgPrice)
    col3_price = round_value(col3_df.weightedAvgPrice)
    col4_price = round_value(col4_df.weightedAvgPrice)
    col5_price = round_value(col5_df.weightedAvgPrice)
    col6_price = round_value(col6_df.weightedAvgPrice)
    col7_price = round_value(col7_df.weightedAvgPrice)
    col8_price = round_value(col8_df.weightedAvgPrice)
    col9_price = round_value(col9_df.weightedAvgPrice)

    # Select the priceChangePercent column
    col1_percent = f'{float(col1_df.priceChangePercent)}%'
    col2_percent = f'{float(col2_df.priceChangePercent)}%'
    col3_percent = f'{float(col3_df.priceChangePercent)}%'
    col4_percent = f'{float(col4_df.priceChangePercent)}%'
    col5_percent = f'{float(col5_df.priceChangePercent)}%'
    col6_percent = f'{float(col6_df.priceChangePercent)}%'
    col7_percent = f'{float(col7_df.priceChangePercent)}%'
    col8_percent = f'{float(col8_df.priceChangePercent)}%'
    col9_percent = f'{float(col9_df.priceChangePercent)}%'

    # Create a metrics price box
    col1.metric(col1_selection, col1_price, col1_percent)
    col2.metric(col2_selection, col2_price, col2_percent)
    col3.metric(col3_selection, col3_price, col3_percent)
    col1.metric(col4_selection, col4_price, col4_percent)
    col2.metric(col5_selection, col5_price, col5_percent)
    col3.metric(col6_selection, col6_price, col6_percent)
    col1.metric(col7_selection, col7_price, col7_percent)
    col2.metric(col8_selection, col8_price, col8_percent)
    col3.metric(col9_selection, col9_price, col9_percent)

    st.header('**All Price**')
    st.dataframe(df)

    def filedownload(df):
        csv = df.to_csv(index=False)
        # strings <-> bytes conversions
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
        return href

    st.markdown(filedownload(df), unsafe_allow_html=True)

    # st.markdown("""
    # <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    # <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    # <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    # """, unsafe_allow_html=True)
    df1 = pd.DataFrame([df.symbol, df.priceChangePercent])
    result = df1.transpose()

  #  def lastPrice_plot():  # viewing plots of specific company
  #      df1 = pd.DataFrame([df.symbol, df.priceChangePercent])
  #      result = df1.transpose()
  #      plt.plot(result.Date)
  #      plt.xticks(rotation=45)
  #      return st.pyplot
  #  currency = st.text_input("enter the currency", ' ')
  #  if currency in list(result.symbol):
  #      lastPrice_plot()
  #   else:
  #      st.write("wrong data entered")
    df1 = pd.DataFrame([df.symbol, df.priceChangePercent])
    result = df1.transpose()
    fig = plt.figure(figsize=(9, 7))
    company = result.symbol.values
    change = result.priceChangePercent.values
    plt.bar(company[:30], change[:30])
    plt.xticks(rotation=90)
    plt.xlabel("Cryptocurrencies", fontweight='bold')
    plt.ylabel("Percent change in prices", fontweight='bold')
    st.pyplot(fig)
    df2 = pd.DataFrame([df.symbol, df.volume])
    result1 = df2.transpose()
    fig1 = plt.figure(figsize=(9, 7))
    change2 = result1.volume.values
    plt.bar(company[:30], change2[:30], color='r')
    plt.xticks(rotation=90)
    plt.xlabel("Cryptocurrencies", fontweight='bold')
    plt.ylabel("Volume", fontweight='bold')
    st.pyplot(fig1)
    df3 = pd.DataFrame([df.symbol, df.weightedAvgPrice])
    result2 = df3.transpose()
    fig2 = plt.figure(figsize=(9, 7))
    change3 = result2.weightedAvgPrice.values
    plt.bar(company[:30], change3[:30])
    plt.xticks(rotation=90)
    plt.xlabel("Cryptocurrencies", fontweight='bold')
    plt.ylabel("weightedAvgPrice", fontweight='bold')
    st.pyplot(fig2)
