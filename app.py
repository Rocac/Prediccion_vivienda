from flask import Flask, request, send_from_directory, jsonify
import joblib
import numpy as np

app = Flask(__name__)
modelo = joblib.load('modelo.pkl')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def style():
    return send_from_directory('.', 'style.css')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos = request.json['datos']
        datos = np.array(datos).reshape(1, -1)
        prediccion = modelo.predict(datos)
        return jsonify({'resultado': f"El precio estimado es: ${int(prediccion[0]):,}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
