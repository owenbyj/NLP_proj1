from flask import Flask

from web import key_extractor

app = Flask(__name__, instance_relative_config=True, template_folder='static/templates')

app.register_blueprint(key_extractor.bp)

if __name__ == '__main__':
    app.run()
