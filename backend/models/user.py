from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime
)

import datetime

from database.db import Base


class User(Base):

    __tablename__ = "users"

    # PRIMARY ID
    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    # USER NAME
    name = Column(
        String,
        nullable=False,
        index=True
    )

    # PROFILE IMAGE
    image_path = Column(
        String,
        nullable=True
    )

    # ALL FACE EMBEDDINGS
    # STORED AS JSON STRING
    embeddings = Column(
        Text,
        nullable=False
    )

    # MASTER AVERAGE EMBEDDING
    master_embedding = Column(
        Text,
        nullable=True
    )

    # CREATED TIME
    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )