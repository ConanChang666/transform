import os
import pymysql
from queue import Queue, Empty
from threading import Lock
from dotenv import load_dotenv
load_dotenv()

_POOL_SIZE = int(os.getenv("MYSQL_POOL_SIZE", "5"))          
_POOL_GET_TIMEOUT = float(os.getenv("MYSQL_POOL_TIMEOUT", "10"))  
_POOL_PING = os.getenv("MYSQL_POOL_PING", "true").lower() == "true" 

_pools = {}            
_pools_lock = Lock()   


def _make_key(host: str, user: str, db: str) -> tuple[str, str, str]:
    return (host, user, db)


def _create_connection(host: str, user: str, pwd: str, db: str) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=host,
        user=user,
        password=pwd,
        db=db,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False, 
    )


def _get_pool(host: str, user: str, db: str) -> Queue:
    key = _make_key(host, user, db)
    with _pools_lock:
        if key not in _pools:
            _pools[key] = Queue(maxsize=_POOL_SIZE)
        return _pools[key]


class MySQLConn:
    def __init__(self, db: str):
        self.host = os.getenv("MYSQL_HOST")
        self.user = os.getenv("MYSQL_USER")
        self.password = os.getenv("MYSQL_PASSWORD")
        self.db = db
        self._pool = _get_pool(self.host, self.user, self.db)
        self.conn = None

    def __enter__(self):
        try:
            conn = self._pool.get(timeout=_POOL_GET_TIMEOUT)
            if _POOL_PING:
                try:
                    conn.ping(reconnect=True)
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
        except Empty:
            conn = None

        if conn is None:
            conn = _create_connection(self.host, self.user, self.password, self.db)

        self.conn = conn
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            try:
                if exc_type is not None:
                    try:
                        self.conn.rollback()
                    except Exception:
                        pass
                try:
                    if _POOL_PING:
                        self.conn.ping(reconnect=False)
                    try:
                        self._pool.put_nowait(self.conn)
                    except Exception:
                        try:
                            self.conn.close()
                        except Exception:
                            pass
                except Exception:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
            finally:
                self.conn = None


# 測試
if __name__ == "__main__":
    with MySQLConn("stock_market_data_lake") as conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM three_major_investors LIMIT 5;")
                print(cursor.fetchall())
            conn.commit()
        except Exception as e:
            print("查詢出錯:", e)
