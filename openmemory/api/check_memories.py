#!/usr/bin/env python3
"""
OpenMemory 跨 App 記憶體可見性診斷腳本
用法 (在 VPS 的 API container 裡面跑):
  docker exec <container_name> python3 /usr/src/openmemory/check_memories.py

或者直接在 VPS 上跑 (需要能連到 DB):
  cd /path/to/api && python3 check_memories.py
"""
import os, sys

# Ensure the app modules are importable
for p in ["/usr/src/openmemory", "/app", os.path.dirname(os.path.abspath(__file__))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.database import SessionLocal
from app.models import Memory, App, User, AccessControl, MemoryState
from app.utils.permissions import check_memory_access_permissions

db = SessionLocal()

# ── 1. 列出所有 Users ──
users = db.query(User).all()
print("=" * 70)
print(f"USERS ({len(users)})")
print("=" * 70)
for u in users:
    print(f"  user_id={u.user_id:20s}  id={u.id}")

# ── 2. 列出所有 Apps ──
apps = db.query(App).all()
print(f"\n{'=' * 70}")
print(f"APPS ({len(apps)})")
print("=" * 70)
for a in apps:
    owner = db.query(User).filter(User.id == a.owner_id).first()
    owner_name = owner.user_id if owner else "?"
    print(f"  {a.name:20s}  active={a.is_active}  owner={owner_name}  id={a.id}")

# ── 3. 列出所有 Memories (按 app 分組) ──
memories = db.query(Memory).all()
print(f"\n{'=' * 70}")
print(f"ALL MEMORIES ({len(memories)})")
print("=" * 70)

by_app = {}
for m in memories:
    app = db.query(App).filter(App.id == m.app_id).first()
    app_name = app.name if app else "unknown"
    by_app.setdefault(app_name, []).append(m)

for app_name, mems in sorted(by_app.items()):
    active_count = sum(1 for m in mems if m.state == MemoryState.active)
    archived_count = sum(1 for m in mems if m.state == MemoryState.archived)
    other_count = len(mems) - active_count - archived_count
    print(f"\n  App: {app_name} ({len(mems)} total: {active_count} active, {archived_count} archived, {other_count} other)")
    for m in mems:
        print(f"    [{str(m.state):8s}] {m.content[:65]}")

# ── 4. ACL Rules ──
acl_rules = db.query(AccessControl).all()
print(f"\n{'=' * 70}")
print(f"ACL RULES ({len(acl_rules)})")
print("=" * 70)
if not acl_rules:
    print("  (none) → all active memories visible to all apps")
else:
    for rule in acl_rules:
        print(f"  {rule.subject_type}:{rule.subject_id} → {rule.object_type}:{rule.object_id} = {rule.effect}")

# ── 5. 跨 App 可見性測試 ──
print(f"\n{'=' * 70}")
print("CROSS-APP VISIBILITY TEST")
print("=" * 70)

# Group by user
user_ids = set(m.user_id for m in memories)
for uid in user_ids:
    user = db.query(User).filter(User.id == uid).first()
    user_name = user.user_id if user else str(uid)
    user_apps = db.query(App).filter(App.owner_id == uid).all()
    user_mems = [m for m in memories if m.user_id == uid]

    if not user_apps or not user_mems:
        continue

    print(f"\n  User: {user_name}")
    print(f"  Apps: {', '.join(a.name for a in user_apps)}")
    print(f"  Memories: {len(user_mems)}")

    for app in user_apps:
        visible = []
        denied = []
        for m in user_mems:
            if check_memory_access_permissions(db, m, app.id):
                visible.append(m)
            else:
                denied.append(m)

        print(f"\n    '{app.name}' sees {len(visible)}/{len(user_mems)} memories:")
        if denied:
            for m in denied:
                src = db.query(App).filter(App.id == m.app_id).first()
                src_name = src.name if src else "?"
                print(f"      DENIED  [{str(m.state):8s}] from={src_name:12s} | {m.content[:50]}")

print(f"\n{'=' * 70}")
print("DONE")
print("=" * 70)

db.close()
