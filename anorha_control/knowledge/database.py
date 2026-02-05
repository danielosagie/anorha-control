"""
SQLite database for experience storage
Stores exploration experiences for training
"""
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

import aiosqlite


@dataclass
class Experience:
    """Single exploration experience."""
    screenshot_before_path: str
    screenshot_after_path: str
    action_x: float  # Normalized 0-1
    action_y: float  # Normalized 0-1
    action_type: int  # 0=click, 1=right_click, 2=double_click, 3=type, 4=scroll
    reward: float
    state_hash_before: str
    state_hash_after: str
    instruction: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))
    success: bool = False


class ExperienceDB:
    """
    Async SQLite database for storing exploration experiences.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
    
    async def connect(self):
        """Open database connection and create tables."""
        self._conn = await aiosqlite.connect(self.db_path)
        await self._create_tables()
    
    async def close(self):
        """Close database connection."""
        if self._conn:
            await self._conn.close()
    
    async def _create_tables(self):
        """Create experience and knowledge tables."""
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screenshot_before_path TEXT NOT NULL,
                screenshot_after_path TEXT NOT NULL,
                action_x REAL NOT NULL,
                action_y REAL NOT NULL,
                action_type INTEGER NOT NULL,
                reward REAL NOT NULL,
                state_hash_before TEXT NOT NULL,
                state_hash_after TEXT NOT NULL,
                instruction TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                timestamp INTEGER NOT NULL,
                success BOOLEAN DEFAULT FALSE
            );
            
            CREATE INDEX IF NOT EXISTS idx_experiences_timestamp ON experiences(timestamp);
            CREATE INDEX IF NOT EXISTS idx_experiences_success ON experiences(success);
            CREATE INDEX IF NOT EXISTS idx_experiences_state_before ON experiences(state_hash_before);
            
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_hash TEXT UNIQUE NOT NULL,
                visit_count INTEGER DEFAULT 1,
                transitions TEXT DEFAULT '[]',
                first_seen INTEGER NOT NULL,
                last_seen INTEGER NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_knowledge_state ON knowledge(state_hash);
            
            CREATE TABLE IF NOT EXISTS exploration_stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_actions INTEGER DEFAULT 0,
                total_successes INTEGER DEFAULT 0,
                unique_states INTEGER DEFAULT 0,
                start_time INTEGER,
                last_update INTEGER
            );
            
            INSERT OR IGNORE INTO exploration_stats (id, start_time, last_update)
            VALUES (1, strftime('%s', 'now'), strftime('%s', 'now'));
        """)
        await self._conn.commit()
    
    async def add_experience(self, exp: Experience) -> int:
        """Add an experience to the database. Returns ID."""
        cursor = await self._conn.execute("""
            INSERT INTO experiences (
                screenshot_before_path, screenshot_after_path,
                action_x, action_y, action_type, reward,
                state_hash_before, state_hash_after,
                instruction, metadata, timestamp, success
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exp.screenshot_before_path,
            exp.screenshot_after_path,
            exp.action_x,
            exp.action_y,
            exp.action_type,
            exp.reward,
            exp.state_hash_before,
            exp.state_hash_after,
            exp.instruction,
            json.dumps(exp.metadata),
            exp.timestamp,
            exp.success,
        ))
        await self._conn.commit()
        
        # Update stats
        await self._conn.execute("""
            UPDATE exploration_stats 
            SET total_actions = total_actions + 1,
                total_successes = total_successes + ?,
                last_update = ?
            WHERE id = 1
        """, (1 if exp.success else 0, int(time.time())))
        await self._conn.commit()
        
        return cursor.lastrowid
    
    async def get_successful_experiences(self, limit: int = 1000) -> List[Experience]:
        """Get successful experiences for training."""
        cursor = await self._conn.execute("""
            SELECT * FROM experiences 
            WHERE success = TRUE 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [self._row_to_experience(row) for row in rows]
    
    async def get_recent_experiences(self, limit: int = 100) -> List[Experience]:
        """Get most recent experiences."""
        cursor = await self._conn.execute("""
            SELECT * FROM experiences 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [self._row_to_experience(row) for row in rows]
    
    async def get_experiences_for_state(self, state_hash: str) -> List[Experience]:
        """Get all experiences from a specific state."""
        cursor = await self._conn.execute("""
            SELECT * FROM experiences 
            WHERE state_hash_before = ?
        """, (state_hash,))
        rows = await cursor.fetchall()
        return [self._row_to_experience(row) for row in rows]
    
    async def count_experiences(self) -> int:
        """Total number of experiences."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM experiences")
        row = await cursor.fetchone()
        return row[0]
    
    async def count_successful(self) -> int:
        """Number of successful experiences."""
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM experiences WHERE success = TRUE"
        )
        row = await cursor.fetchone()
        return row[0]
    
    async def get_experiences_by_reward(self, min_reward: float = 0.0, limit: int = 10000) -> List[Experience]:
        """Get experiences with reward >= min_reward for training export."""
        cursor = await self._conn.execute("""
            SELECT * FROM experiences 
            WHERE reward >= ?
            ORDER BY reward DESC, timestamp DESC
            LIMIT ?
        """, (min_reward, limit))
        rows = await cursor.fetchall()
        return [self._row_to_experience(row) for row in rows]
    
    async def update_knowledge(self, state_hash: str, transition_info: Dict[str, Any]):
        """Update knowledge about a state."""
        now = int(time.time())
        
        # Check if state exists
        cursor = await self._conn.execute(
            "SELECT id, transitions, visit_count FROM knowledge WHERE state_hash = ?",
            (state_hash,)
        )
        row = await cursor.fetchone()
        
        if row:
            # Update existing
            transitions = json.loads(row[1])
            transitions.append(transition_info)
            transitions = transitions[-100:]  # Keep last 100
            
            await self._conn.execute("""
                UPDATE knowledge 
                SET visit_count = visit_count + 1,
                    transitions = ?,
                    last_seen = ?
                WHERE state_hash = ?
            """, (json.dumps(transitions), now, state_hash))
        else:
            # Insert new
            await self._conn.execute("""
                INSERT INTO knowledge (state_hash, transitions, first_seen, last_seen)
                VALUES (?, ?, ?, ?)
            """, (state_hash, json.dumps([transition_info]), now, now))
            
            # Update unique state count
            await self._conn.execute("""
                UPDATE exploration_stats 
                SET unique_states = unique_states + 1
                WHERE id = 1
            """)
        
        await self._conn.commit()
    
    async def get_least_visited_states(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get states with fewest visits (for curiosity-driven exploration)."""
        cursor = await self._conn.execute("""
            SELECT state_hash, visit_count, transitions 
            FROM knowledge 
            ORDER BY visit_count ASC 
            LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [
            {"state_hash": r[0], "visit_count": r[1], "transitions": json.loads(r[2])}
            for r in rows
        ]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        cursor = await self._conn.execute("SELECT * FROM exploration_stats WHERE id = 1")
        row = await cursor.fetchone()
        if row:
            return {
                "total_actions": row[1],
                "total_successes": row[2],
                "unique_states": row[3],
                "start_time": row[4],
                "last_update": row[5],
                "success_rate": row[2] / max(1, row[1]),
            }
        return {}
    
    async def get_category_stats(self) -> Dict[str, Any]:
        """Get task category breakdown from experience metadata."""
        cursor = await self._conn.execute("SELECT metadata FROM experiences")
        rows = await cursor.fetchall()
        
        category_counts = {}
        category_successes = {}
        site_counts = {}
        
        for row in rows:
            try:
                meta = json.loads(row[0]) if row[0] else {}
                category = meta.get("category", "unknown")
                site = meta.get("site", "unknown")
                success = meta.get("success", False)
                
                # Count by category
                category_counts[category] = category_counts.get(category, 0) + 1
                if success:
                    category_successes[category] = category_successes.get(category, 0) + 1
                
                # Count by site
                site_counts[site] = site_counts.get(site, 0) + 1
            except (json.JSONDecodeError, AttributeError):
                pass
        
        return {
            "by_category": category_counts,
            "successes_by_category": category_successes,
            "by_site": site_counts,
            "total_experiences": len(rows),
        }

    
    def _row_to_experience(self, row) -> Experience:
        """Convert database row to Experience object."""
        return Experience(
            id=row[0],
            screenshot_before_path=row[1],
            screenshot_after_path=row[2],
            action_x=row[3],
            action_y=row[4],
            action_type=row[5],
            reward=row[6],
            state_hash_before=row[7],
            state_hash_after=row[8],
            instruction=row[9],
            metadata=json.loads(row[10]) if row[10] else {},
            timestamp=row[11],
            success=bool(row[12]),
        )


# Quick test
if __name__ == "__main__":
    async def test():
        db = ExperienceDB(Path("/tmp/test_experiences.db"))
        await db.connect()
        
        # Add test experience
        exp = Experience(
            screenshot_before_path="/tmp/before.png",
            screenshot_after_path="/tmp/after.png",
            action_x=0.5,
            action_y=0.3,
            action_type=0,
            reward=0.8,
            state_hash_before="abc123",
            state_hash_after="def456",
            instruction="click button",
            success=True,
        )
        
        exp_id = await db.add_experience(exp)
        print(f"Added experience with ID: {exp_id}")
        
        # Get stats
        stats = await db.get_stats()
        print(f"Stats: {stats}")
        
        # Count
        count = await db.count_experiences()
        print(f"Total experiences: {count}")
        
        await db.close()
    
    asyncio.run(test())
