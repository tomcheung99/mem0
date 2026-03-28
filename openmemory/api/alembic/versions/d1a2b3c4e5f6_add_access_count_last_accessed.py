"""add_access_count_last_accessed_to_memories

Revision ID: d1a2b3c4e5f6
Revises: b7c3a1d2e4f5
Create Date: 2026-03-28 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd1a2b3c4e5f6'
down_revision: Union[str, None] = 'b7c3a1d2e4f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('memories', sa.Column('access_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('memories', sa.Column('last_accessed', sa.DateTime(), nullable=True))
    op.create_index('idx_memory_last_accessed', 'memories', ['last_accessed'])


def downgrade() -> None:
    op.drop_index('idx_memory_last_accessed', table_name='memories')
    op.drop_column('memories', 'last_accessed')
    op.drop_column('memories', 'access_count')
