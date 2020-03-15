import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

from summarize.method1.summerize import summarize_from_file, summarize_from_text
from web.preloading import get_wv


bp = Blueprint('key_extractor', __name__, url_prefix='/key_extractor')


@bp.route('/', methods=['GET', 'POST'])
def extractor_page():
    if request.method == 'GET':
        print(url_for('static', filename='templates/key_extractor.html'))
        outdict = {'output_list': '',
                   'input_str': '',
                   'input_title': '',
                   'smooth_method': ''}
        return render_template('key_extractor.html', output=outdict)
    elif request.method == 'POST':
        print(request.form)
        output_list = summarize_from_text(request.form['input-title'], request.form['input-content'], request.form['smooth_method'])
        outdict = {'output_list': output_list,
                   'input_str': request.form['input-content'],
                   'input_title': request.form['input-title'],
                   'smooth_method': request.form['smooth_method']}
        return render_template('key_extractor.html', output=outdict)
