from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Load the test predictions DataFrame
    test_predictions_df = pd.read_csv(r"C:\Users\manil\ROSSMAN_SALES_PREDICTION_BY_MANILA\test_df.csv")

    # Generate a plot and convert it to a base64-encoded image
    plt.figure(figsize=(12, 6))
    plt.plot(test_predictions_df['Date'], test_predictions_df['PredictedSalesPerCustomer'], label='Predicted SalesPerCustomer', color='blue')
    plt.title('Predicted Sales Per Customer Over Time')
    plt.xlabel('Date')
    plt.ylabel('Predicted SalesPerCustomer')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object and encode it as base64
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()

    # Pass the predictions and base64-encoded plot to the template
    return render_template('index.html', predictions=test_predictions_df.to_html(), plot_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
