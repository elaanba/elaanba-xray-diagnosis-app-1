import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import base64
from io import BytesIO, StringIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FOLDER'] = 'model'

class Tubes:
    def __init__(self, model_path, image):
        self.model_path = model_path
        self.image = image

        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def Detect(self):
        labels = [
            'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'ETT - Abnormal',
            'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged',
            'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal'
        ]

        img = cv2.resize(self.image, (380, 380)) / 255.0
        input_data = np.expand_dims(img.astype('float32'), axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output_data[0]

        result = [[label, round(float(prob), 3)] for label, prob in zip(labels, predictions)]
        return result

def highlight_max(s):
    return ['background-color: lightgreen' if v >= 0.5 else '' for v in s]

@app.route("/", methods=["GET", "POST"])
def index():
    models = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.endswith('.tflite')]
    if request.method == "POST":
        selected_model = request.form.get("model_name")
        model_path = os.path.join(app.config['MODEL_FOLDER'], selected_model)

        f = request.files.get("image")
        if not f:
            return render_template("index.html", models=models, selected_model=selected_model)

        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        image = cv2.imread(filepath)
        case = Tubes(model_path, image)
        data = case.Detect()

        df = pd.DataFrame(data, columns=["Class", "Probability"])
        styled_df = df.style.apply(highlight_max, subset=["Probability"])
        table_html = styled_df.to_html(classes="styled-table")

        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # Save CSV to memory
        csv_buf = StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        global last_csv
        last_csv = csv_buf

        return render_template("index.html", table=table_html, img_data=img_str, models=models, selected_model=selected_model)

    return render_template("index.html", table=None, img_data=None, models=models, selected_model=None)

@app.route("/download", methods=["POST"])
def download():
    global last_csv
    if last_csv:
        return send_file(BytesIO(last_csv.getvalue().encode()), mimetype='text/csv',
                         as_attachment=True, download_name='diagnosis_results.csv')
    return "No data", 400

if __name__ == "__main__":
    last_csv = None
    app.run(debug=True)
