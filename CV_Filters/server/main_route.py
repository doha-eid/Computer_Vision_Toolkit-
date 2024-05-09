from flask import render_template
from server import app


@app.route('/')
def main_route():
    return render_template('index.html')
