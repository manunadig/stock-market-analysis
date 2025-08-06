import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.linear_model import LinearRegression
import os
plt.style.use("fivethirtyeight")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock') or 'POWERGRID.NS'
        
        # Download stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2024, 10, 1)
        df = yf.download(stock, start=start, end=end)

        if df.empty:
            return render_template('index.html', error="Invalid stock symbol or no data available.")

        # Descriptive statistics
        data_desc = df.describe()

        # Compute EMAs
        ema20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema50 = df['Close'].ewm(span=50, adjust=False).mean()
        ema100 = df['Close'].ewm(span=100, adjust=False).mean()
        ema200 = df['Close'].ewm(span=200, adjust=False).mean()

        # Prepare data for Linear Regression (Trend Line)
        df = df.reset_index()
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        X = df[['Days']]
        y = df['Close']

        model = LinearRegression()
        model.fit(X, y)
        df['Prediction'] = model.predict(X)

        # Plot 1: EMA 20 & 50
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df['Close'], 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (EMA 20 & 50)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        # Plot 2: EMA 100 & 200
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df['Close'], 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (EMA 100 & 200)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        # Plot 3: Linear Regression Prediction
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(df['Close'], label='Actual Price', color='blue')
        ax3.plot(df['Prediction'], label='Linear Trend Prediction', color='orange')
        ax3.set_title("Actual vs Predicted Trend (Linear Regression)")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Save dataset
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path, index=False)

        return render_template('index.html',
                       stock=stock,  # <- pass the stock symbol to HTML
                       plot_path_ema_20_50=ema_chart_path,
                       plot_path_ema_100_200=ema_chart_path_100_200,
                       plot_path_prediction=prediction_chart_path,
                       data_desc=data_desc.to_html(classes='table table-bordered'),
                       dataset_link=csv_file_path)


    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
