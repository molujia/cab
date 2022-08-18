from flask import Flask

from datetime import timedelta
import config
from blueprints import se_bp
from blueprints import home_bp
import ImageProcess as IP


app = Flask(__name__)

app.config.from_object(config)
app.register_blueprint(se_bp)
app.register_blueprint(home_bp)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)



if __name__ == '__main__':
    app.run()
