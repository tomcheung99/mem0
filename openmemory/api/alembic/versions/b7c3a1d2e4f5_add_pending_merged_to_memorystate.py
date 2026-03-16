"""add_pending_merged_to_memorystate

Revision ID: b7c3a1d2e4f5
Revises: afd00efbd06b
Create Date: 2026-03-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7c3a1d2e4f5'
down_revision: Union[str, None] = 'afd00efbd06b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # PostgreSQL ENUMs require ALTER TYPE to add new values.
    # These are non-transactional DDL, so execute outside a transaction block.
    op.execute("ALTER TYPE memorystate ADD VALUE IF NOT EXISTS 'pending'")
    op.execute("ALTER TYPE memorystate ADD VALUE IF NOT EXISTS 'merged'")


def downgrade() -> None:
    # PostgreSQL does not support removing values from an ENUM type.
    # A full migration would require recreating the type + column.
    # For safety we leave the values in place on downgrade.
    pass
