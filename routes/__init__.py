from flask import Flask

app = Flask(__name__)
import routes.square
import routes.investigate
import routes.ticket
import routes.trivia
import routes.safeguard
import routes.blanks
