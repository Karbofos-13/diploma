from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import os


app = Flask(__name__)


model_path = 'dnn_model.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Файл модели '{model_path}' не найден. Пожалуйста, убедитесь, что файл существует.")

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    prediction_result = None
    
    if request.method == 'POST':
        try:
            density = float(request.form.get('density', 0))
            elastic_modulus = float(request.form.get('elastic_modulus', 0))
            hardener_amount = float(request.form.get('hardener_amount', 0))
            epoxy_group = float(request.form.get('epoxy_group', 0))
            flash_point = float(request.form.get('flash_point', 0))
            surface_density = float(request.form.get('surface_density', 0))
            elastic_modulus_tension = float(request.form.get('elastic_modulus_tension', 0))
            tensile_strength = float(request.form.get('tensile_strength', 0))
            resin_consumption = float(request.form.get('resin_consumption', 0))
            angle = float(request.form.get('angle', 0))
            step = float(request.form.get('step', 0))
            density_wave = float(request.form.get('density_wave', 0))
            
            input_data = np.array([[density, elastic_modulus, hardener_amount, epoxy_group, flash_point,
                                    surface_density, elastic_modulus_tension, tensile_strength,
                                    resin_consumption, angle, step, density_wave]], dtype=np.float32)
            
            
            prediction_result = float(model.predict(input_data)[0][0])
            
        except Exception as e:
            error = f"Ошибка предсказания: {str(e)}"
        
    return render_template('index.html', prediction_result=prediction_result, error=error)
        
if __name__ == '__main__':
    app.run()