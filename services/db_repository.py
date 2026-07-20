import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


class DBRepository:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )
        self.cursor = self.conn.cursor()
        self._ensure_agent_interactions_table()

    def _ensure_agent_interactions_table(self) -> None:
        """Create the agent_interactions table if it does not yet exist."""
        query = (
            "CREATE TABLE IF NOT EXISTS agent_interactions ("
            "  id INT AUTO_INCREMENT PRIMARY KEY,"
            "  day INT NOT NULL,"
            "  sender_id INT NOT NULL,"
            "  receiver_id INT NOT NULL,"
            "  interaction_type VARCHAR(50) NOT NULL,"
            "  amount DECIMAL(10,2) NOT NULL DEFAULT 0.00,"
            "  message TEXT,"
            "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        self.cursor.execute(query)
        self.conn.commit()

    def add_daily_metrics(self, day: int, metrics: list[dict]) -> None:
        if not metrics:
            return
        query = "INSERT INTO daily_metrics (agent_id, day, wealth, happiness, integrity, reputation, action_type, fear_index) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        values = [
            (
                m["agent_id"],
                day,
                m["wealth"],
                m["happiness"],
                m["integrity"],
                m.get("reputation", 1.0),
                m.get("action_type", ""),
                m.get("fear_index", 0.0),
            )
            for m in metrics
        ]
        self.cursor.executemany(query, values)
        self.conn.commit()

    def add_memory(self, day: int, memories: list[dict]) -> None:
        if not memories:
            return
        query = "INSERT INTO memories (agent_id, day, event_description, agent_reflection, family_status_snapshot, action_type) VALUES (%s, %s, %s, %s, %s, %s)"
        values = [
            (
                m["agent_id"],
                day,
                m["event_description"],
                m["agent_reflection"],
                m.get("family_status_snapshot", ""),
                m.get("action_type", ""),
            )
            for m in memories
        ]
        self.cursor.executemany(query, values)
        self.conn.commit()

    def add_agent_interactions(self, day: int, interactions: list[dict]) -> None:
        """Persist agent-to-agent social and economic interactions."""
        if not interactions:
            return
        query = (
            "INSERT INTO agent_interactions "
            "(day, sender_id, receiver_id, interaction_type, amount, message) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        values = [
            (
                day,
                i["sender_id"],
                i["receiver_id"],
                i["interaction_type"],
                i.get("amount", 0.0),
                i.get("message", ""),
            )
            for i in interactions
        ]
        self.cursor.executemany(query, values)
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()
