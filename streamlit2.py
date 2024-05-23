import requests
import streamlit as st
import json

def get_data_from_api():
    url = 'http://192.168.0.7:5000/get_data'  # Update the URL
    response = requests.get(url)
    return response.json()

def main():
    st.title('Streamlit API Example')

    saved_data = get_data_from_api()
    if saved_data:
        # Initialize lists to store data for the table
        headers = []
        rows = []

        # Iterate over each item in the 'data' key
        for item in saved_data['data']:
            # Extract the data value
            data_value = item['data']
            # Split the data value by ','
            pairs = data_value.split(',')
            # Iterate over each pair
            for pair in pairs:
                # Split each pair by ':'
                key, value = pair.strip().split(':')
                # Append the key and value to the headers and rows lists
                headers.append(key.strip())
                rows.append(value.strip())

        # Create HTML for the table with row styles
        table_html = f"<table><tr><th>Key</th><th>Value</th></tr>"
        for header, row in zip(headers, rows):
            # Apply row style if value is 'yes'
            row_style = 'background-color: lightcoral;' if row.strip().lower() == 'yes' else ''
            table_html += f"<tr><td>{header}</td><td style='{row_style}'>{row}</td></tr>"
        table_html += "</table>"

        # Display the table
        st.write(table_html, unsafe_allow_html=True)
    else:
        st.write('No data saved yet')

if __name__ == '__main__':
    main()
