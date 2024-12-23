"""added project id to datasets

Revision ID: 0c154a390bf3
Revises: a3c33691cf0d
Create Date: 2024-12-22 17:43:21.965806

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0c154a390bf3'
down_revision: Union[str, None] = 'a3c33691cf0d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('datasets', sa.Column('project_id', sa.Integer(), nullable=True))
    op.create_foreign_key(None, 'datasets', 'projects', ['project_id'], ['id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'datasets', type_='foreignkey')
    op.drop_column('datasets', 'project_id')
    # ### end Alembic commands ###