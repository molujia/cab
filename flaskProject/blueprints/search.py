from flask import Blueprint,render_template

bp=Blueprint("search",__name__,url_prefix='/search')

@bp.route("/result")
def search():
    return "搜索列表"