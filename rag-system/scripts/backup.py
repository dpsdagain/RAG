"""SQLite backup script with rotation.

Creates timestamped backups using sqlite3's backup API and retains
only the most recent N backups.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def backup_database(
    db_path: str = "./data/rag.db",
    backup_dir: str = "./data/backups",
    keep_last: int = 5,
) -> Path:
    """Create a timestamped SQLite backup.

    Args:
        db_path: Path to the source database.
        backup_dir: Directory for backup files.
        keep_last: Number of recent backups to keep.

    Returns:
        Path to the created backup file.
    """
    db_path = Path(db_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"rag_backup_{timestamp}.db"

    print(f"Backing up {db_path} → {backup_path}")

    source = sqlite3.connect(str(db_path))
    dest = sqlite3.connect(str(backup_path))

    source.backup(dest)

    source.close()
    dest.close()

    size_mb = backup_path.stat().st_size / (1024 * 1024)
    print(f"Backup created: {backup_path} ({size_mb:.1f} MB)")

    # Rotate old backups
    backups = sorted(backup_dir.glob("rag_backup_*.db"), reverse=True)
    for old_backup in backups[keep_last:]:
        old_backup.unlink()
        print(f"Deleted old backup: {old_backup.name}")

    print(f"Backups retained: {min(len(backups), keep_last)}")
    return backup_path


if __name__ == "__main__":
    backup_database()
