from flask import Flask
from ui import ui

app = Flask(__name__)
app.register_blueprint(ui)
if __name__ == '__main__':
    app.run(port=5001)
