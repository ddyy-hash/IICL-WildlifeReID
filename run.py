from app import create_app, db
from app.models import User, Video
from app.routes import main_bp

app = create_app()

# 创建数据库表
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    print(app.template_folder)
    print(app.static_folder)
    print(main_bp.static_folder)
    print(main_bp.template_folder)
    app.run(debug=True)