from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import sqlite3
import json
import os

app = Flask(__name__)
CORS(app)

DB_FOLDER = 'db'
DB_PATH = os.path.join(DB_FOLDER, 'traffic.db')

def init_db():
    """Initialize database and create tables if they don't exist"""
    os.makedirs(DB_FOLDER, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lokasi TEXT NOT NULL,
            waktu TEXT NOT NULL,
            jumlah_kendaraan INTEGER NOT NULL,
            congestion_index REAL NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            UNIQUE(lokasi)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        
        required_fields = ['lokasi', 'waktu', 'jumlah_kendaraan', 'congestion_index', 'status']
        if not all(k in data for k in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO traffic_data 
            (lokasi, waktu, jumlah_kendaraan, congestion_index, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['lokasi'],
            data['waktu'],
            data['jumlah_kendaraan'],
            data['congestion_index'],
            data['status'],
            timestamp
        ))
        
        conn.commit()
        
        cursor.execute('SELECT * FROM traffic_data WHERE lokasi = ?', (data['lokasi'],))
        row = cursor.fetchone()
        conn.close()
        
        result = dict(row)
        print(f"Received: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        return jsonify({
            'success': True,
            'message': 'Data received',
            'data': result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/traffic/latest', methods=['GET'])
def get_latest():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM traffic_data ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return jsonify([dict(row) for row in rows]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/traffic/location/<location_name>', methods=['GET'])
def get_by_location(location_name):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM traffic_data WHERE lokasi = ?', (location_name,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return jsonify(dict(row)), 200
        return jsonify({'error': 'Location not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/traffic/status/<status>', methods=['GET'])
def get_by_status(status):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM traffic_data WHERE status = ?', (status,))
        rows = cursor.fetchall()
        conn.close()
        
        return jsonify([dict(row) for row in rows]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) as count FROM traffic_data')
        count = cursor.fetchone()['count']
        conn.close()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'active_locations': count
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/traffic/clear', methods=['DELETE'])
def clear_all():
    """Clear all traffic data (useful for testing)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM traffic_data')
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'All traffic data cleared'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)