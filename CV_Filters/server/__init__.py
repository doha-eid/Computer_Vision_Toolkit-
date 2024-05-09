from flask import Flask

app = Flask(__name__)

from server import main_route, routes