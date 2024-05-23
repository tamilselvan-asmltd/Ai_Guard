from flask import Flask, request

app_flask = Flask(__name__)

saved_data = []

@app_flask.route('/save_data', methods=['POST'])
def save_data():
    data = request.json
    saved_data.append(data)
    return {'message': 'Data saved successfully'}

@app_flask.route('/get_data', methods=['GET'])
def get_data():
    return {'data': saved_data}

if __name__ == '__main__':
    app_flask.run(host='192.168.0.7', debug=True)
