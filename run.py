from app import create_app, db

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        # 创建数据库表
        db.create_all()
    
    app.run(debug=True, host='::', port=5000)
