"""empty message

Revision ID: 66c568a6cfa7
Revises: a6a2cca4b1f0
Create Date: 2019-09-09 16:12:12.959532

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '66c568a6cfa7'
down_revision = 'a6a2cca4b1f0'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('movies', sa.Column('thumbnail', sa.String(length=64), nullable=True))
    op.add_column('movies', sa.Column('watchlink', sa.String(length=256), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('movies', 'watchlink')
    op.drop_column('movies', 'thumbnail')
    # ### end Alembic commands ###
