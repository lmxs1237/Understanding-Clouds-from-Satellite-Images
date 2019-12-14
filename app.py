from flask import Flask, render_template, request, flash
from wtforms import SubmitField
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
import os
import uuid

import segementation


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'upload'
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'static', 'upload')
bootstrap = Bootstrap(app)

class UploadForm(FlaskForm):
    img = FileField('Upload Image', validators=[FileRequired(), FileAllowed(['jpg','jpeg','png','gif'])])
    submit = SubmitField()


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/summary_report')
def about():
    return render_template('summary.html')

@app.route('/classification_models', methods=['GET', 'POST'])
def model():
    form = UploadForm()
    if request.method == 'GET':
        return render_template('model.html', form=form)
    else:
        if form.validate_on_submit():
            prefix = ''.join(str(uuid.uuid4()).split('-'))
            ext = form.img.data.filename.split('.')[-1]
            img_name = prefix + '.' + ext
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            form.img.data.save(img_path)

            new_img = segementation.process(img_path)
            print(new_img)
        return render_template('model_left.html', form=form, old_img=img_name, new_img=new_img)

if __name__ == '__main__':
    app.run(debug=True)