import sqlite3

class DatabaseManager:
    def __init__(self, db_name='sensor_experiments.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        schema_sql = """
        -- Table: Sensor Information
        CREATE TABLE IF NOT EXISTS sensors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            sensitivity REAL,
            sinr_threshold REAL,
            freq_min REAL,
            freq_max REAL,
            sector_count INTEGER DEFAULT 1
        );
        
        --Table: Sensor Gain Sectors
        CREATE TABLE IF NOT EXISTS sensor_gain_sector (
            sensor_id INTEGER,
            exp_id INTEGER,
            sector_id INTEGER,
            angle_start REAL,    -- degrees, e.g. 0.0
            angle_end REAL,      -- degrees, e.g. 60.0
            gain_db REAL,        -- dBi
            PRIMARY KEY(sensor_id, exp_id, sector_id),
            FOREIGN KEY(sensor_id) REFERENCES sensors(id),
            FOREIGN KEY (exp_id)    REFERENCES experiments(id)
        );

        -- Table: Sensor Experiments
        CREATE TABLE IF NOT EXISTS sensor_exp (
            exp_id INTEGER,
            sensor_id INTEGER,
            sensor_exp_id INTEGER,
            x REAL,
            y REAL,
            z REAL,
            PRIMARY KEY (sensor_exp_id),
            FOREIGN KEY (exp_id) REFERENCES experiments(id),
            FOREIGN KEY (sensor_id) REFERENCES sensors(id)
        );

        -- Table: Sensor Up Times
        CREATE TABLE IF NOT EXISTS sensor_up_time (
            exp_id INTEGER,
            sensor_id INTEGER,
            sensor_up_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            sensor_noise REAL,
            received_thermal_noise_power REAL,
            sinr REAL,
            active INTEGER,
            FOREIGN KEY (exp_id) REFERENCES experiments(id),
            FOREIGN KEY (sensor_id) REFERENCES sensors(id)
        );

        -- Table: Sensor Detections
        CREATE TABLE IF NOT EXISTS sensor_detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_up_id INTEGER,
            drone_id INTEGER,
            received_power REAL,
            received_signal_bandwidth REAL,
            FOREIGN KEY (sensor_up_id) REFERENCES sensor_up_time(sensor_up_id),
            FOREIGN KEY (drone_id) REFERENCES drones(id)
        );

        -- Table: Drones
        CREATE TABLE IF NOT EXISTS drones (
            id INTEGER PRIMARY KEY,
            drone_type TEXT NOT NULL,
            drone_transmitter_power REAL,
            transmitter_antenna_gain REAL,
            properties TEXT, 
            freq_min REAL,
            freq_max REAL,
            bandwidth REAL
        );

        -- Table: Drone Experiments
        CREATE TABLE IF NOT EXISTS drone_exp (
            exp_id INTEGER,
            drone_id INTEGER,
            drone_exp_id INTEGER,
            PRIMARY KEY (drone_exp_id),
            FOREIGN KEY (exp_id) REFERENCES experiments(id),
            FOREIGN KEY (drone_id) REFERENCES drones(id)
        );

        -- Table: Drone Radio Frequency
        CREATE TABLE IF NOT EXISTS drone_radio_freq (
            exp_id INTEGER,
            drone_id INTEGER,
            drone_exp_id INTEGER,
            radio_freq REAL,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            PRIMARY KEY (exp_id, drone_id, radio_freq),
            FOREIGN KEY (exp_id) REFERENCES experiments(id),
            FOREIGN KEY (drone_id) REFERENCES drones(id),
            FOREIGN KEY (drone_exp_id) REFERENCES drone_exp(drone_exp_id)
        );

        -- Table: Drone Experiment Time (Drone Trajectory)
        CREATE TABLE IF NOT EXISTS drone_exp_time (
            id INTEGER PRIMARY KEY,
            drone_id INTEGER,
            exp_id INTEGER,
            time TIMESTAMP,
            speed REAL,
            start_x REAL,
            start_y REAL,
            start_z REAL,
            end_x REAL,
            end_y REAL,
            end_z REAL,
            FOREIGN KEY (drone_id) REFERENCES drones(id),
            FOREIGN KEY (exp_id) REFERENCES experiments(id)
        );

        -- Table: Experiments
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY,
            date TIMESTAMP NOT NULL,
            description TEXT,
            parameters TEXT
        );

        -- Table: Experiment Environment
        CREATE TABLE IF NOT EXISTS exp_env (
            exp_id INTEGER,
            topography_id INTEGER,
            env_id INTEGER,
            PRIMARY KEY (exp_id, topography_id, env_id),
            FOREIGN KEY (exp_id) REFERENCES experiments(id),
            FOREIGN KEY (topography_id) REFERENCES topography(id),
            FOREIGN KEY (env_id) REFERENCES environment(id)
        );

        -- Table: Topography
        CREATE TABLE IF NOT EXISTS topography (
            id INTEGER PRIMARY KEY,
            x REAL,
            y REAL
        );

        -- Table: Environment
        CREATE TABLE IF NOT EXISTS environment (
            id INTEGER PRIMARY KEY,
            disturbance REAL,
            visibility REAL
        );

        -- Table: Mountains
        CREATE TABLE IF NOT EXISTS mountains (
            id INTEGER PRIMARY KEY,
            x REAL,
            y REAL,
            side_length REAL,
            height REAL
        );

        -- Table: Topography Mountains
        CREATE TABLE IF NOT EXISTS topo_mountains (
            topography_id INTEGER,
            mountain_id INTEGER,
            PRIMARY KEY (topography_id, mountain_id),
            FOREIGN KEY (topography_id) REFERENCES topography(id),
            FOREIGN KEY (mountain_id) REFERENCES mountains(id)
        );
        
        --Table: Sensor Distances
        CREATE TABLE IF NOT EXISTS sensor_distances (
                exp_id        INTEGER,
                sensor1_id    INTEGER,
                sensor1_x REAL,
                sensor1_y REAL,
                sensor1_z REAL,
                sensor2_id    INTEGER,
                sensor2_x REAL,
                sensor2_y REAL,
                sensor2_z REAL,
                distance      REAL,
                PRIMARY KEY(exp_id, sensor1_id, sensor2_id),
                FOREIGN KEY(sensor1_id) REFERENCES sensors(id),
                FOREIGN KEY(sensor2_id) REFERENCES sensors(id),
                FOREIGN KEY(exp_id) REFERENCES experiments(id)
            );
        """
        self.cursor.executescript(schema_sql)
        self.conn.commit()

    # Getters and Setters for Sensors
    def get_sensors(self):
        self.cursor.execute("SELECT * FROM sensors")
        return self.cursor.fetchall()

    def set_sensor(self, sensor_type, sensitivity, sinr_threshold, freq_min, freq_max, sector_count=1):
        self.cursor.execute("""
            INSERT OR REPLACE INTO sensors (type, sensitivity, sinr_threshold, freq_min, freq_max, sector_count) 
            VALUES (?, ?, ?, ?, ?, ?)""",
            (sensor_type, sensitivity, sinr_threshold, freq_min, freq_max, sector_count)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_sensor_info_by_experiment(self, exp_id):
        self.cursor.execute("""
            SELECT s.id, s.type, s.sensitivity, s.sinr_threshold, s.freq_min, s.freq_max, s.sector_count
              FROM sensors AS s
              JOIN sensor_exp AS se 
                ON s.id = se.sensor_id
             WHERE se.exp_id = ?
        """, (exp_id,))
        return self.cursor.fetchall()

    def get_sensor_gain_sectors(self):
        self.cursor.execute("SELECT * FROM sensor_gain_sector")
        return self.cursor.fetchall()

    def set_sensor_gain_sector(self, sensor_id, exp_id, sector_id,
                               angle_start, angle_end, gain_db):
        self.cursor.execute("""
            INSERT OR REPLACE INTO sensor_gain_sector
                (sensor_id, exp_id, sector_id, angle_start, angle_end, gain_db)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (sensor_id, exp_id, sector_id, angle_start, angle_end, gain_db))
        self.conn.commit()

    def load_sensor_patterns(self, exp_id):
        """
        Returns: { sensor_id: [(start,end,gain), …] }
        """
        self.cursor.execute("""
          SELECT sensor_id, angle_start, angle_end, gain_db
            FROM sensor_gain_sector
           WHERE exp_id = ?
           ORDER BY sensor_id, sector_id
        """, (exp_id,))
        patterns = {}
        for sid, a0, a1, g in self.cursor.fetchall():
            patterns.setdefault(sid, []).append((a0, a1, g))
        return patterns

    # Getters and Setters for Sensor Experiments
    def get_sensor_exp(self):
        self.cursor.execute("SELECT * FROM sensor_exp")
        return self.cursor.fetchall()

    def get_sensors_by_exp_id(self, exp_id):
        self.cursor.execute("SELECT * FROM sensor_exp WHERE exp_id = ?", (exp_id,))
        return self.cursor.fetchall()

    def get_sensor_exp_by_exp_id_and_sensor_id(self, exp_id, sensor_id):
        self.cursor.execute("""
            SELECT * FROM sensor_exp
            WHERE exp_id = ?
            AND sensor_id = ?
        """, (exp_id, sensor_id))
        return self.cursor.fetchall()

    def set_sensor_exp(self, exp_id, sensor_id, sensor_exp_id, x, y, z):
        self.cursor.execute("""
            INSERT OR REPLACE INTO sensor_exp (exp_id, sensor_id, sensor_exp_id, x, y, z)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (exp_id, sensor_id, sensor_exp_id, x, y, z))
        self.conn.commit()

    # Getters and Setters for Sensor Up Time
    def get_sensor_up_time(self):
        self.cursor.execute("SELECT * FROM sensor_up_time")
        return self.cursor.fetchall()

    def get_sensor_up_time_by_experiment(self, exp_id):
        self.cursor.execute("""
            SELECT * FROM sensor_up_time WHERE exp_id = ?
        """, (exp_id,))
        return self.cursor.fetchall()

    def get_active_sensor_up_times(self, exp_id, timestamp):
        self.cursor.execute("""
            SELECT * FROM sensor_up_time
            WHERE exp_id = ?
            AND start_time <= ?
            AND end_time >= ?
        """, (exp_id, timestamp, timestamp))
        return self.cursor.fetchall()

    def set_sensor_up_time(self, exp_id, sensor_id, start_time, end_time, sensor_noise,
                           received_thermal_noise_power, sinr, active):
        self.cursor.execute("""
            INSERT OR REPLACE INTO sensor_up_time (exp_id, sensor_id, start_time, end_time, sensor_noise, 
            received_thermal_noise_power, sinr, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (exp_id, sensor_id, start_time, end_time, sensor_noise, received_thermal_noise_power, sinr,
              active))
        self.conn.commit()

    def update_sensor_metrics(self, sensor_up_id, thermal_noise, sinr, active):
        self.cursor.execute("""
            UPDATE sensor_up_time
            SET received_thermal_noise_power = ?, sinr = ?, active = ?
            WHERE sensor_up_id = ?
        """, (thermal_noise, sinr, active, sensor_up_id))
        self.conn.commit()

    # Getters and Setters for Sensor Detections
    def get_sensor_detections(self):
        self.cursor.execute("SELECT * FROM sensor_detections")
        return self.cursor.fetchall()

    def get_sensor_detections_by_sensor_up_id(self, sensor_up_id):
        self.cursor.execute("""
            SELECT * FROM sensor_detections
            WHERE sensor_up_id = ?
        """, (sensor_up_id,))
        return self.cursor.fetchall()

    def set_sensor_detection(self, sensor_up_id, drone_id, received_power, received_signal_bandwidth):
        self.cursor.execute("""
            INSERT INTO sensor_detections (sensor_up_id, drone_id, received_power, received_signal_bandwidth)
            VALUES (?, ?, ?, ?)
        """, (sensor_up_id, drone_id, received_power, received_signal_bandwidth))
        self.conn.commit()

    # Getters and Setters for Drones
    def get_drones(self):
        self.cursor.execute("SELECT * FROM drones")
        return self.cursor.fetchall()

    def get_drone_by_id(self, drone_id):
        self.cursor.execute("SELECT * FROM drones WHERE id = ?", (drone_id,))
        return self.cursor.fetchone()  # Use fetchone() since there should only be one drone per ID

    def set_drone(self, drone_id, drone_type, drone_transmitter_power, transmitter_antenna_gain, properties,
                  freq_min, freq_max, bandwidth):
        self.cursor.execute("""
            INSERT OR REPLACE INTO drones (id, drone_type, drone_transmitter_power, transmitter_antenna_gain, 
            properties, freq_min, freq_max, bandwidth)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (drone_id, drone_type, drone_transmitter_power, transmitter_antenna_gain, properties,
              freq_min, freq_max, bandwidth))
        self.conn.commit()

    # Getters and Setters for Drone Experiments
    def get_drone_exp(self):
        self.cursor.execute("SELECT * FROM drone_exp")
        return self.cursor.fetchall()

    def get_drones_by_exp_id(self, exp_id):
        self.cursor.execute("SELECT * FROM drone_exp WHERE exp_id = ?", (exp_id,))
        return self.cursor.fetchall()

    def set_drone_exp(self, exp_id, drone_id, drone_exp_id):
        self.cursor.execute("INSERT OR REPLACE INTO drone_exp (exp_id, drone_id, drone_exp_id) VALUES (?, ?, ?)",
                            (exp_id, drone_id, drone_exp_id))
        self.conn.commit()

    # Getters and Setters for Drone Radio Frequency
    def get_drone_radio_freq(self):
        self.cursor.execute("SELECT * FROM drone_radio_freq")
        return self.cursor.fetchall()

    def set_drone_radio_freq(self, exp_id, drone_id, drone_exp_id, radio_freq, start_time, end_time):
        self.cursor.execute("""
            INSERT OR REPLACE INTO drone_radio_freq (exp_id, drone_id, drone_exp_id, radio_freq, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (exp_id, drone_id, drone_exp_id, radio_freq, start_time, end_time))
        self.conn.commit()

    # Getters and Setters for Drone Experiment Time (Trajectory)
    def get_drone_exp_time(self):
        self.cursor.execute("SELECT * FROM drone_exp_time")
        return self.cursor.fetchall()

    def set_drone_exp_time(self, id, drone_id, exp_id, time, speed, start_x, start_y, start_z, end_x, end_y, end_z):
        self.cursor.execute("""
            INSERT OR REPLACE INTO drone_exp_time (id, drone_id, exp_id, time, speed, start_x, start_y, start_z, end_x, end_y, end_z)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (id, drone_id, exp_id, time, speed, start_x, start_y, start_z, end_x, end_y, end_z))
        self.conn.commit()

    def get_drone_exp_time_by_drone_and_experiment(self, exp_id, drone_id):
        """
        Return all trajectory records for a specific drone in a specific experiment.
        """
        self.cursor.execute(
            "SELECT * FROM drone_exp_time WHERE exp_id = ? AND drone_id = ? ORDER BY time",
            (exp_id, drone_id)
        )
        return self.cursor.fetchall()

    # Getters and Setters for Experiments
    def get_experiments(self):
        self.cursor.execute("SELECT * FROM experiments")
        return self.cursor.fetchall()

    def set_experiment(self, exp_id, date, description, parameters):
        self.cursor.execute(
            """INSERT OR REPLACE INTO experiments (id, date, description, parameters) VALUES (?, ?, ?, ?)""",
            (exp_id, date, description, parameters))
        self.conn.commit()

    def get_mountains_by_experiment(self, exp_id):
        query = """
        SELECT m.id, m.x, m.y, m.side_length, m.height
        FROM mountains m
        JOIN topo_mountains tm ON m.id = tm.mountain_id
        JOIN exp_env ee ON tm.topography_id = ee.topography_id
        WHERE ee.exp_id = ?
        """
        self.cursor.execute(query, (exp_id,))
        return self.cursor.fetchall()

    def get_sensors_distances(self):
        self.cursor.execute("SELECT * FROM sensor_distances")
        return self.cursor.fetchall()

    def set_sensors_distances(self, exp_id, sid_i, si_x, si_y, si_z, sid_j, sj_x, sj_y, sj_z, d):
        self.cursor.execute(
            """INSERT INTO sensor_distances (exp_id, sensor1_id, sensor1_x, sensor1_y, sensor1_z, 
            sensor2_id, sensor2_x, sensor2_y, sensor2_z, distance) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (exp_id, sid_i, si_x, si_y, si_z, sid_j, sj_x, sj_y, sj_z, d))
        self.conn.commit()

    def get_next_experiment_id(self):
        """
        Return the next experiment ID by taking MAX(id) + 1, or 1 if no experiments exist.
        """
        self.cursor.execute("SELECT MAX(id) FROM experiments")
        row = self.cursor.fetchone()
        max_id = row[0] if row and row[0] is not None else 0
        return max_id + 1

    def get_next_topography_id(self):
        """
        Return the next topography ID by taking MAX(id) + 1, or 1 if none exist.
        """
        self.cursor.execute("SELECT MAX(id) FROM topography")
        row = self.cursor.fetchone()
        max_id = row[0] if row and row[0] is not None else 0
        return max_id + 1

    def get_next_environment_id(self):
        """
        Return the next environment ID by taking MAX(id) + 1, or 1 if none exist.
        """
        self.cursor.execute("SELECT MAX(id) FROM environment")
        row = self.cursor.fetchone()
        max_id = row[0] if row and row[0] is not None else 0
        return max_id + 1