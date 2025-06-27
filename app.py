from flask import Flask,render_template,request
import joblib
from src.logging.logger import my_logger


app = Flask('__name__')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/validate-url', methods=['POST'])
def check_url():
    if request.method == 'POST':
        url = request.form['url']
        my_logger.info(f"input url is {url}")
        model = joblib.load('best_model/ML_model.joblib')
        prediction = model.predict([url])
        my_logger.info(f"prediction is {prediction}")

        return render_template('result.html', prediction="SAFE URL " if prediction[0] == 1 else "UNSAFE URL")
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)
