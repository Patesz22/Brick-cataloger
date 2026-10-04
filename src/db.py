import os
import time
import gzip
import csv
import sqlite3
import urllib.request
from datetime import datetime, timedelta

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_FILE = os.path.join(ROOT_DIR, "rebrickable.db")
BASE_URL = "https://cdn.rebrickable.com/media/downloads/"

TABLES = {
    "themes": {
        "file": "themes.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS themes
               (
                   id        INTEGER PRIMARY KEY,
                   name      TEXT,
                   parent_id INTEGER
               );
               """
    },
    "colors": {
        "file": "colors.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS colors
               (
                   id        INTEGER PRIMARY KEY,
                   name      TEXT,
                   rgb       TEXT,
                   is_trans  INTEGER,
                   num_parts INTEGER,
                   num_sets  INTEGER,
                   y1        INTEGER,
                   y2        INTEGER
               );
               """
    },
    "part_categories": {
        "file": "part_categories.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS part_categories
               (
                   id   INTEGER PRIMARY KEY,
                   name TEXT
               );
               """
    },
    "parts": {
        "file": "parts.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS parts
               (
                   part_num      TEXT PRIMARY KEY,
                   name          TEXT,
                   part_cat_id   INTEGER,
                   part_material TEXT,
                   FOREIGN KEY (part_cat_id) REFERENCES part_categories (id)
               );
               """
    },
    "part_relationships": {
        "file": "part_relationships.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS part_relationships
               (
                   rel_type        TEXT,
                   child_part_num  TEXT,
                   parent_part_num TEXT
               );
               """
    },
    "elements": {
        "file": "elements.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS elements
               (
                   element_id TEXT PRIMARY KEY,
                   part_num   TEXT,
                   color_id   INTEGER,
                   design_id  TEXT,
                   FOREIGN KEY (part_num) REFERENCES parts (part_num),
                   FOREIGN KEY (color_id) REFERENCES colors (id)
               );
               """
    },
    "sets": {
        "file": "sets.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS sets
               (
                   set_num   TEXT PRIMARY KEY,
                   name      TEXT,
                   year      INTEGER,
                   theme_id  INTEGER,
                   num_parts INTEGER,
                   img_url   TEXT,
                   FOREIGN KEY (theme_id) REFERENCES themes (id)
               );
               """
    },
    "minifigs": {
        "file": "minifigs.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS minifigs
               (
                   fig_num   TEXT PRIMARY KEY,
                   name      TEXT,
                   num_parts INTEGER,
                   img_url   TEXT
               );
               """
    },
    "inventories": {
        "file": "inventories.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS inventories
               (
                   id      INTEGER PRIMARY KEY,
                   version INTEGER,
                   set_num TEXT
               );
               """
    },
    "inventory_parts": {
        "file": "inventory_parts.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS inventory_parts
               (
                   inventory_id INTEGER,
                   part_num     TEXT,
                   color_id     INTEGER,
                   quantity     INTEGER,
                   is_spare     INTEGER,
                   img_url      TEXT
               );
               """
    },
    "inventory_sets": {
        "file": "inventory_sets.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS inventory_sets
               (
                   inventory_id INTEGER,
                   set_num      TEXT,
                   quantity     INTEGER
               );
               """
    },
    "inventory_minifigs": {
        "file": "inventory_minifigs.csv.gz",
        "sql": """
               CREATE TABLE IF NOT EXISTS inventory_minifigs
               (
                   inventory_id INTEGER,
                   fig_num      TEXT,
                   quantity     INTEGER
               );
               """
    }
}


class RebrickableOfflineDB:
    """
    Manages local SQLite storage and queries for Rebrickable database dumps.
    """

    def __init__(self, db_path: str = DB_FILE):
        """
        Initializes the database instance and metadata structures.

        @parameters:
            @param db_path: str - Path to the SQLite database file.
        @returns:
            None
        """
        self.db_path = db_path
        self._init_meta()

    def get_connection(self) -> sqlite3.Connection:
        """
        Opens a database connection configured with foreign key constraints.

        @parameters:
            None
        @returns:
            sqlite3.Connection - Active database connection.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_meta(self) -> None:
        """
        Initializes the internal synchronization tracking table.

        @parameters:
            None
        @returns:
            None
        """
        with self.get_connection() as conn:
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS _db_sync_meta
                         (
                             id                    INTEGER PRIMARY KEY CHECK (id = 1),
                             last_synced_timestamp REAL
                         );
                         """)

    def is_update_needed(self, cooldown_hours: int = 24) -> bool:
        """
        Determines whether synchronization is permitted based on elapsed hours.

        @parameters:
            @param cooldown_hours: int - Minimum time interval required between downloads.
        @returns:
            bool - True if cooldown has expired or no record exists; False otherwise.
        """
        with self.get_connection() as conn:
            row = conn.execute("SELECT last_synced_timestamp FROM _db_sync_meta WHERE id = 1;").fetchone()
            if not row or row[0] is None:
                return True
            last_sync = datetime.fromtimestamp(row[0])
            elapsed = datetime.now() - last_sync
            if elapsed < timedelta(hours=cooldown_hours):
                remaining = timedelta(hours=cooldown_hours) - elapsed
                print(f"[CACHE] Database is fresh (updated {elapsed.seconds // 3600}h ago).")
                print(f"[CACHE] Next allowed update in: {str(remaining).split('.')[0]}.")
                return False
            return True

    def sync_database(self, force: bool = False) -> None:
        """
        Streams compressed CSV files from Rebrickable and regenerates SQLite tables.

        @parameters:
            @param force: bool - Bypasses the 24-hour rate limit check when True.
        @returns:
            None
        """
        if not force and not self.is_update_needed():
            return

        print("[SYNC] Starting Rebrickable database download and refresh...")
        start_time = time.time()

        conn = self.get_connection()
        cur = conn.cursor()

        cur.execute("PRAGMA foreign_keys = OFF;")
        cur.execute("PRAGMA synchronous = OFF;")
        cur.execute("PRAGMA journal_mode = MEMORY;")

        for table_name, meta in TABLES.items():
            print(f" -> Downloading & populating '{table_name}'...")

            cur.execute(f"DROP TABLE IF EXISTS {table_name};")
            cur.execute(meta["sql"])

            file_url = BASE_URL + meta["file"]
            req = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})

            with urllib.request.urlopen(req) as response:
                with gzip.GzipFile(fileobj=response) as gz:
                    reader = csv.reader(line.decode("utf-8") for line in gz)
                    headers = next(reader)

                    placeholders = ", ".join(["?"] * len(headers))
                    insert_sql = f"INSERT INTO {table_name} ({', '.join(headers)}) VALUES ({placeholders});"

                    batch = []
                    for row in reader:
                        cleaned = [
                            1 if val in ("t", "True", "true") else 0 if val in ("f", "False", "false") else (
                                None if val == "" else val)
                            for val in row
                        ]
                        batch.append(cleaned)
                        if len(batch) >= 50000:
                            cur.executemany(insert_sql, batch)
                            batch.clear()

                    if batch:
                        cur.executemany(insert_sql, batch)

            conn.commit()

        print("[INDEX] Creating query acceleration indexes...")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_parts_name ON parts(name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_parts_cat ON parts(part_cat_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_elements_lookup ON elements(part_num, color_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_elements_design ON elements(design_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_inv_parts_lookup ON inventory_parts(part_num, color_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_colors_name ON colors(name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_colors_rgb ON colors(rgb);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sets_theme ON sets(theme_id);")

        cur.execute("INSERT OR REPLACE INTO _db_sync_meta (id, last_synced_timestamp) VALUES (1, ?);", (time.time(),))
        conn.commit()
        conn.close()

        print(f"[DONE] Complete database synchronized in {time.time() - start_time:.1f}s.")

    def get_part_info(self, part_num: str) -> tuple | None:
        """
        Retrieves part description, category, and plastic material type.

        @parameters:
            @param part_num: str - Rebrickable part number.
        @returns:
            tuple | None - (part_num, name, category_name, part_material) or None.
        """
        with self.get_connection() as conn:
            query = """
                    SELECT p.part_num, p.name AS part_name, c.name AS category_name, p.part_material
                    FROM parts p
                             LEFT JOIN part_categories c ON p.part_cat_id = c.id
                    WHERE p.part_num = ?; \
                    """
            return conn.execute(query, (part_num,)).fetchone()

    def resolve_element_id(self, part_num: str, color_id: int) -> tuple | None:
        """
        Finds the official element ID and design ID for a part and color pairing.

        @parameters:
            @param part_num: str - Part number.
            @param color_id: int - Rebrickable color identifier.
        @returns:
            tuple | None - (element_id, design_id) or None.
        """
        with self.get_connection() as conn:
            query = "SELECT element_id, design_id FROM elements WHERE part_num = ? AND color_id = ?;"
            return conn.execute(query, (part_num, color_id)).fetchone()

    def get_color_by_name(self, name: str) -> tuple | None:
        """
        Finds a color record using an exact or partial name match.

        @parameters:
            @param name: str - Target color name.
        @returns:
            tuple | None - (id, name, rgb, is_trans, num_parts, num_sets, y1, y2) or None.
        """
        with self.get_connection() as conn:
            query = """
                    SELECT id, \
                           name, \
                           rgb, \
                           is_trans, \
                           num_parts, \
                           num_sets, \
                           y1, \
                           y2
                    FROM colors
                    WHERE LOWER(name) = ?; \
                    """
            row = conn.execute(query, (name.lower(),)).fetchone()
            if not row:
                query_like = """
                             SELECT id, \
                                    name, \
                                    rgb, \
                                    is_trans, \
                                    num_parts, \
                                    num_sets, \
                                    y1, \
                                    y2
                             FROM colors
                             WHERE LOWER(name) LIKE ?
                             LIMIT 1; \
                             """
                row = conn.execute(query_like, (f"%{name.lower()}%",)).fetchone()
            return row

    def search_color(self, term: str) -> list[tuple]:
        """
        Matches colors by name fragment or exact hexadecimal string.

        @parameters:
            @param term: str - Color name substring or 6-digit RGB hex code.
        @returns:
            list[tuple] - List of matching records (id, name, rgb, is_trans, num_parts, num_sets, y1, y2).
        """
        with self.get_connection() as conn:
            query = """
                    SELECT id, \
                           name, \
                           rgb, \
                           is_trans, \
                           num_parts, \
                           num_sets, \
                           y1, \
                           y2
                    FROM colors
                    WHERE LOWER(name) LIKE ? \
                       OR LOWER(rgb) = ?; \
                    """
            like_term = f"%{term.lower()}%"
            return conn.execute(query, (like_term, term.lower().lstrip("#"))).fetchall()

    def get_parts_in_set(self, set_num: str) -> list[tuple]:
        """
        Retrieves inventory parts, color names, quantities, and image URLs for a set.

        @parameters:
            @param set_num: str - Set number with or without suffix.
        @returns:
            list[tuple] - Rows containing (part_num, name, color_name, quantity, is_spare, img_url).
        """
        with self.get_connection() as conn:
            query = """
                    SELECT ip.part_num, p.name, c.name AS color_name, ip.quantity, ip.is_spare, ip.img_url
                    FROM sets s
                             JOIN inventories i ON i.set_num = s.set_num
                             JOIN inventory_parts ip ON ip.inventory_id = i.id
                             JOIN parts p ON ip.part_num = p.part_num
                             JOIN colors c ON ip.color_id = c.id
                    WHERE s.set_num = ? \
                       OR s.set_num LIKE ?; \
                    """
            return conn.execute(query, (set_num, f"{set_num}-%")).fetchall()
